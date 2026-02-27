import copy
import logging
import re

import torch
import time
import random
import os
import math
import torch.distributed as dist

from typing import Union, List, Dict, Tuple, Optional
from transformers import Trainer, TrainerCallback, PreTrainedTokenizerFast
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import has_length, IntervalStrategy, speed_metrics
from wandb import Table
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model

# Local imports
from .config import BabyLMConfig
from .continual_pretraining.learning_rate_reset_callback import LearningRateResetCallback
# Data curriculum related
from .data_curriculum.datasampler import CurriculumSampler, DistributedCurriculumSampler
from .data_curriculum.difficulty_scorer import get_difficulty_scorer
from .data_curriculum.pacing_fn import get_pacing_fn
from .dataloader import CurriculumDataLoader

from .evaluator import ZeroShotEvaluator, SuperGlueEvaluator
from src.helper.dataset_preprocessor import base_collate_fn
from src.helper.inference import compute_trainer_perplexity, prepare_dataset_for_ppl_inference
from .gradual_stacking.stacking_callback import GradualStackingCallback
from .helper.visualization import calculate_and_save_layer_similarity_plot

# Set up logging for different components of the trainer
logger = logging.getLogger(__name__)
data_cl_logger = logging.getLogger("Data Curriculum")

class FinalLayerSimilarityCallback(TrainerCallback):
    """
    Saves the layer similarity matrix at the very end of training for any model.
    """
    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            results_dir = args.output_dir.replace("checkpoints/", "results/")
            # Save in the root output directory (results/run_name/)
            calculate_and_save_layer_similarity_plot(
                model,
                output_dir=os.path.join(results_dir, f"checkpoint-{state.global_step}"),
                stage_name=None,
                step=state.global_step
            )


class FLOPTrainingLimitCallback(TrainerCallback):
    """
    A callback that monitors training progress and signals to stop
    if step or FLOP limits are reached.
    """

    def __init__(self, max_flops: Optional[float] = None):
        self.max_flops = max_flops

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.max_flops is not None and state.total_flos >= self.max_flops:
            logger.info(f"FLOP limit reached: {state.total_flos} >= {self.max_flops}. Stopping training.")
            control.should_training_stop = True


class StagedEvaluationCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self._boundaries = set()
        self._initialized = False
        self._dry_run = trainer.dry_run

    def on_step_end(self, args, state, control, **kwargs):
        # Initialize boundaries by reading from other callbacks
        if not self._initialized:
            train_loader = kwargs.get("train_dataloader")
            self._initialize_boundaries(train_loader=train_loader)
        # Check if next step is a boundary
        if (state.global_step + 1) in self._boundaries:
            logger.info(f"Stage boundary will be reached in the next step ({state.global_step}). Performing evaluation...")
            if self._dry_run:
                logger.info(f"Evaluation at stage end will be skipped in dry run")
            else:
                # Evaluate & save the model
                self.trainer.evaluate()

    def _initialize_boundaries(self, train_loader):

        self._update_with_gradual_stacking_boundaries()
        self._update_with_staged_data_curriculum_boundaries(train_loader=train_loader)

        if self._boundaries:
            self._initialized = True
            logger.info(f"StagedEvaluationCallback: Unified boundaries detected: {sorted(list(self._boundaries))}")
        else:
            raise RuntimeError("StagedEvaluationCallback: No boundaries could be detected, "
                               "even though StagedEvaluationCallback was created.")

    def _update_with_gradual_stacking_boundaries(self):
        # Check for Gradual Stacking Callback and get its growth steps
        for callback in self.trainer.callback_handler.callbacks:
            # Check by type name to avoid strict import dependencies
            if "GradualStackingCallback" in str(type(callback)):
                self._boundaries.update(callback.steps_at_which_model_should_be_grown)

    def _update_with_staged_data_curriculum_boundaries(self, train_loader):
        # Check for Staged Data Curriculum and calculate steps from percentiles
        is_staged = (self.trainer.data_curriculum_cfg is not None and
                     self.trainer.data_curriculum_cfg.difficulty_scorer_name == "staged_data_split")
        if is_staged:
            # Retrieve the scorer from the curriculum sampler
            scorer = train_loader.sampler.difficulty_scorer
            pacing_fn = train_loader.sampler.pacing_fn
            threshold_percentiles = scorer.transition_thresholds
            total_steps = self.trainer.args.max_steps

            curriculum_steps = []
            for p in threshold_percentiles:
                low = 0
                high = total_steps
                boundary_step = high

                while low <= high:
                    mid = (low + high) // 2
                    if pacing_fn(mid) >= p:
                        boundary_step = mid
                        high = mid - 1
                    else:
                        low = mid + 1

                curriculum_steps.append(boundary_step)

            self._boundaries.update(curriculum_steps)


class CurriculumLearningCallback(TrainerCallback):
    """
    A TrainerCallback that updates the data sampler and data collator with the current global step of training.
    This is crucial for curriculum learning where the data sampling or processing might depend on the training progress.
    """

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        train_dataloader,
        **kwargs,
    ):
        """
        Called at the beginning of training. Sets the initial global step for the DataLoader.
        """
        train_dataloader.global_stepnum = state.global_step

    def on_step_end(self, *_, train_dataloader, **kwargs) -> None:
        """
        Called at the end of each training step. Increments the global step for the sampler
        (if it's a curriculum sampler) and the DataLoader. This ensures curriculum components
        are aware of the current training stage.
        """
        if isinstance(
            train_dataloader.sampler,
            (CurriculumSampler, DistributedCurriculumSampler),
        ):
            train_dataloader.sampler.global_stepnum += 1

        train_dataloader.global_stepnum += 1



class CustomTrainer(Trainer):
    """
    CustomTrainer extends Hugging Face's Trainer to incorporate various curriculum learning strategies
    (now focusing on data curriculum) and specialized evaluation routines for the BabyLM Challenge
    """
    def __init__(
        self,
        hydra_config: BabyLMConfig,
        dry_run: bool,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizerFast,
        curriculum_learning_table: Union[Table, None] = None,
        **kwargs,
    ) -> None:
        """
        Overrides the __init__ method of the base Trainer to add specific configurations
        and curriculum learning components

        Args:
            * hydra_config (BabyLMConfig): The configuration object loaded using Hydra. It contains all experimental settings
            * dry_run (bool): Whether the experiment is being run in dry run mode
            * args (TrainingArguments): The Hugging Face training arguments
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for the current training run
            * curriculum_learning_table (wandb.Table, optional): A Weights & Biases table
                                                                 used to log detailed curriculum
                                                                 learning progress (e.g., difficulty scores,
                                                                 pacing function values, sampled data)
        """

        # Get configurations from the hydra config file
        self.hydra_config = hydra_config
        self.dry_run = dry_run

        # Extract experiment-specific identifiers from the configuration
        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name

        # Evaluation flags from the configuration
        self.eval_blimp = hydra_config.trainer.eval_blimp
        self.eval_glue = hydra_config.trainer.eval_glue
        self.eval_perplexity = hydra_config.trainer.eval_perplexity
        self.skip_execution_of_eval_scripts_for_debugging = (
            hydra_config.experiment.skip_execution_of_eval_scripts_for_debugging
        )

        # Call the parent Trainer's __init__ method to set up standard training components
        super().__init__(args=args, **kwargs)

        # Initialize curriculum learning configurations
        self.data_curriculum_cfg = hydra_config.data_curriculum

        # Logging for curriculum usage
        if self.data_curriculum_cfg:
            data_cl_logger.info(
                f"Using data curriculum configuration {self.data_curriculum_cfg}"
            )

        # Store tokenizer and optional W&B curriculum table
        self.processing_class = tokenizer
        self.curriculum_learning_table = curriculum_learning_table

        # Add custom callbacks to the Trainer's callback handler
        self.add_callback(CurriculumLearningCallback())  # Manages global step for curriculum

        # Add Gradual Stacking Callback
        if self.hydra_config.gradual_stacking.enabled:
            stacking_callback = GradualStackingCallback(total_training_steps = self.hydra_config.trainer.max_training_steps,
                                                        k_number_of_stages = self.hydra_config.gradual_stacking.k_number_of_stages,
                                                        alpha = self.hydra_config.gradual_stacking.alpha,
                                                        layer_per_block = self.hydra_config.gradual_stacking.layer_per_block,
                                                        align_with_staged_data_curriculum=self.hydra_config.gradual_stacking.align_with_staged_data_curriculum)
            self.add_callback(stacking_callback)

        is_staged_data_curriculum = (
                self.data_curriculum_cfg is not None and
                self.data_curriculum_cfg.difficulty_scorer_name == "staged_data_split"
        )

        # Conditional addition of the ContinualPretrainingCallback
        if self.data_curriculum_cfg:
            # Check if 'lr_reset' exists in data curriculum kwargs and if it is True
            scorer_kwargs = self.data_curriculum_cfg.difficulty_scorer_kwargs or {}
            should_lr_reset = hydra_config.continual_pretraining.enable_lr_reset

            if is_staged_data_curriculum and should_lr_reset:
                logger.info("Continual Pre-training enabled: Adding LearningRateResetCallback")
                # The callback will handle boundary calculation internally as discussed
                self.add_callback(LearningRateResetCallback(trainer=self, cfg=hydra_config))

        # If FLOP limit is provided, add callback to monitor that
        if self.hydra_config.trainer.max_flops is not None:
            # Add the limit callback
            self.add_callback(
                FLOPTrainingLimitCallback(max_flops=self.hydra_config.trainer.max_flops)
            )

        if is_staged_data_curriculum or self.hydra_config.gradual_stacking.enabled:
            logger.info("Adding StagedEvaluationCallback based on experiment configuration")
            self.add_callback(StagedEvaluationCallback(trainer=self))

        # Coordination flag to prevent duplicate evaluations at the same step
        self._last_eval_step_stacking_and_data_curriculum = -1

        # Add the general similarity callback (runs for all models at the end)
        self.add_callback(FinalLayerSimilarityCallback())

        # Flag indicating whether training is distributed across multiple GPUs/processes
        self.is_distributed = self.args.world_size > 1

    def _get_train_sampler(self):
        """
        Overriding this method to use custom samplers that enable data-driven curriculum pacing.
        """
        # Validate dataset
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = self._create_generator()
        seed = self._get_seed()

        if self.data_curriculum_cfg:
            return self._get_curriculum_sampler(generator, seed)
        else:
            return self._get_default_sampler(generator, seed)

    def _create_generator(self):
        """
        Create and seed a torch random number generator for single-process training.
        """
        if self.is_distributed:
            return None  # Multi-process training handles seeding differently
        else:
            generator = torch.Generator()

            # Determine seed: use data_seed if provided, else fallback
            if self.args.data_seed is None:
                # Sample a random integer seed
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed

            generator.manual_seed(seed)
            return generator

    def _get_seed(self):
        """
        Determine the seed for distributed samplers
        """
        return (
            self.args.data_seed
            if self.args.data_seed is not None
            else self.args.seed
        )

    def _get_curriculum_sampler(self, generator, seed):
        """
        Create a curriculum-aware sampler using difficulty scoring and pacing
        """
        pacing_fn = get_pacing_fn(
            self.data_curriculum_cfg.pacing_fn_name,
            self.args.max_steps,
            **self.data_curriculum_cfg.pacing_fn_kwargs,
        )
        difficulty_scorer = get_difficulty_scorer(
            self.data_curriculum_cfg.difficulty_scorer_name,
            self.data_curriculum_cfg.difficulty_scorer_kwargs,
            trainer=self,
        )
        if self.is_distributed:
            # Multi-process distributed curriculum sampler
            return DistributedCurriculumSampler(
                self.train_dataset,
                difficulty_scorer=difficulty_scorer,
                pacing_fn=pacing_fn,
                batch_size=self.args.per_device_train_batch_size,
                generator=generator,
                global_stepnum=self.state.global_step,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )
        else:
            # Single-process curriculum sampler
            return CurriculumSampler(
                self.train_dataset,
                difficulty_scorer=difficulty_scorer,
                pacing_fn=pacing_fn,
                batch_size=self.args.per_device_train_batch_size,
                generator=generator,
                global_stepnum=self.state.global_step,
            )

    def _get_default_sampler(self, generator, seed):
        """
        Fallback to default random or distributed sampler.
        """
        if self.is_distributed:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )
        else:
            return RandomSampler(self.train_dataset, generator=generator)


    def _get_ignore_columns(self, dataset) -> List[str]:
        """
        Returns the list of columns to ignore when training. This is used to remove columns that
        are not used for training, but are used for curriculum pacing.

        Args:
            * dataset (:class:`~datasets.Dataset`): The dataset to use for training.

        Returns:
            * (List[str]): The list of columns to ignore when training.
        """
        # Determine relevant columns for training / forward pass of the model
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        if signature_columns is None:
            signature_columns = []
        # Determine columns to ignore by removing the relevant columns for training from all dataset columns
        ignore_columns = list(
            set(dataset.column_names) - set(signature_columns)
        )
        return ignore_columns

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        logs["model_total_parameters"] = sum(p.numel() for p in self.model.parameters())
        logs["train_cumulative_flops"] = self.state.total_flos
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen))

        output = {**logs, **{"step": self.state.global_step}}
        if "curriculum_learning_table" not in logs:
            # NOTE: Everything in state will be serialized to a json file when we save
            # a checkpoint - alas a wandb.Table is not
            self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        The dataset is sorted by the scoring function, if provided.

        Returns:
            * (CustomDataLoader): The custom training dataloader, a subclass instance of the torch
                Dataloader.
        """

        # Ensure the training dataset is set before proceeding
        assert self.train_dataset is not None
        # Ensure tokenizer is available
        assert (
                self.processing_class is not None
        ), "Tokenizer is not set. Please set the tokenizer before calling the train method."

        # NOTE: The standard Trainer.get_train_dataloader() method removes unused columns for
        # training, but here we only remove the 'filename' column explicitly.
        # Other columns in the dataset should now be either float or int (filename is the only string column).
        # Other irrelevant columns will be removed later in a postprocessing step (after objective collation).

        # Get the sampler, potentially curriculum-aware or distributed
        train_sampler = self._get_train_sampler()
        # Remove the 'filename' column as it's not used for training
        train_dataset = self.train_dataset.remove_columns("filename")

        # NOTE: Some columns still need to be ignored during batching because they are not part of the model's input signature.
        # We obtain these columns here to exclude them from the batch, but they may still be used in objective generation.
        ignore_columns = self._get_ignore_columns(train_dataset)

        # Return an instance of the custom CurriculumDataLoader initialized with all relevant parameters
        # TODO: Why will always a CurriculumDataLoader be returned, even when Curriculum isn't used?
        return CurriculumDataLoader(
            global_stepnum=self.state.global_step,  # Current global step for curriculum pacing
            # tokenizer=self.tokenizer,  # Tokenizer instance for data processing
            ignore_columns=ignore_columns,  # Columns to exclude from training batches
            dataset=train_dataset,  # Dataset with 'filename' column removed
            sampler=train_sampler,  # Sampler that controls data iteration order
            batch_size=self._train_batch_size,  # Batch size to use
            drop_last=self.args.dataloader_drop_last,  # Whether to drop last incomplete batch
            num_workers=self.args.dataloader_num_workers,  # Number of worker processes for data loading
            pin_memory=self.args.dataloader_pin_memory,  # Whether to use pinned memory for faster transfers to GPU
        )

    def compute_loss(self, model, inputs, **kwargs):
        """
        Compute and return the loss for the current training step.
        This method also logs curriculum-related metrics.
        """

        # Standard Hugging Face loss computation
        loss = super().compute_loss(model, inputs, **kwargs)

        # Needed for logging
        loss_metric = {"loss": loss.item()}

        # Safety check, stop if max steps exceeded
        self._check_max_steps()

        # Log curriculum learning metrics
        if self._should_log():
            self.log(loss_metric)
            self._log_curriculum_metrics(inputs)

        return loss

    def _check_max_steps(self):
        """Raise an error if global_step exceeds max_steps."""
        if self.state.global_step >= self.args.max_steps:
            raise Exception(
                """
                Reached max_steps already - training should have stopped.
                NOTE: You are probably using a resume_from_checkpoint flag with max_steps 
                set to a value smaller than the number of steps in the checkpoint.
                """
            )

    def _should_log(self):
        """Determine if logging should be done at the current step."""
        return (
                self.args.logging_strategy == IntervalStrategy.STEPS
                and self.state.global_step % self.args.logging_steps == 0
        )

    def _log_curriculum_metrics(self, inputs):
        """
        Log curriculum learning metrics and sample data if applicable.
        """
        # Check if curriculum learning table is initialized aka some form of curriculum learning is done
        if self.curriculum_learning_table is not None:
            # Skip redundant logging if already logged for this step
            if not self._check_if_curriculum_metrics_were_logged():
                # Prepare data samples from current batch
                data_samples = self._decode_sample_inputs(inputs)

                # If data curriculum is done
                if self.data_curriculum_cfg:
                    (
                        data_difficulty_percentile,
                        data_sampled_percentile,
                        num_samples,
                        max_difficulty_score,
                        min_difficulty_score,
                        median_difficulty_score,
                        current_stage,
                    ) = self._compute_data_curriculum_difficulty_metrics_for_logging()
                else:
                    # Default metrics if no data curriculum is done
                    data_difficulty_percentile = 1.0
                    data_sampled_percentile = 1.0
                    num_samples = len(self.callback_handler.train_dataloader.sampler) * self.args.world_size
                    max_difficulty_score, min_difficulty_score, median_difficulty_score = 0.0, 0.0, 0.0
                    current_stage = 1.0

                # Log additionally as standard metrics
                self.log({
                    "curriculum/data_difficulty_percentile": data_difficulty_percentile,
                    "curriculum/data_sampled_percentile": data_sampled_percentile,
                    "curriculum/num_samples": num_samples,
                    "curriculum/max_difficulty_score": max_difficulty_score,
                    "curriculum/min_difficulty_score": min_difficulty_score,
                    "curriculum/median_difficulty_score": median_difficulty_score,
                    "curriculum/current_stage": current_stage,
                })

                # Add collected curriculum learning data into the tracking table
                self.curriculum_learning_table.add_data(
                    self.state.global_step,
                    data_difficulty_percentile,
                    data_sampled_percentile,
                    num_samples,
                    max_difficulty_score,
                    min_difficulty_score,
                    median_difficulty_score,
                    data_samples,
                    current_stage,
                )

                # If at evaluation step, log the curriculum learning table
                if self._check_if_curriculum_table_should_be_logged():
                    _curriculum_learning_table = Table(
                        columns=self.curriculum_learning_table.columns,
                        data=self.curriculum_learning_table.data,
                    )
                    self.log({"curriculum_learning_table": _curriculum_learning_table})

    def _check_if_curriculum_metrics_were_logged(self):
        """Check if curriculum metrics are already logged for this step."""
        if len(self.curriculum_learning_table.data) > 0:
            # Get the last recorded step in the table
            max_table_step = int(
                self.curriculum_learning_table.data[-1][0]
            )

            # If we've already logged metrics for this step, skip redundant logging
            if max_table_step >= self.state.global_step:
                return True
            else:
                return False
        else:
            return False

    def _decode_sample_inputs(self, inputs, num_samples: int = 5):
        """Decode the first few input_ids from the current batch."""
        decoded_samples = ""
        for i in range(min(num_samples, len(inputs["input_ids"]))):
            decoded_text = self.processing_class.decode(
                inputs["input_ids"][i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Remove artifacts
            decoded_text = decoded_text.replace("Ġ", " ")
            decoded_text = decoded_text.replace("ĉ", "\n")
            decoded_text = decoded_text.replace("Ċ", "\n")
            decoded_text = decoded_text.replace(" ", " ")
            decoded_text = decoded_text.replace("âĢĻ", "'")  # Apostrophes
            decoded_text = decoded_text.replace("âĢĵ", "-")  # Dashes
            decoded_text = decoded_text.replace("âĢľ", '"')  # Left double quote
            decoded_text = decoded_text.replace("âĢĿ", '"')  # Right double quote
            decoded_text = decoded_text.replace("âĢĶ", "...")  # Ellipsis
            decoded_text = re.sub(r' +', ' ', decoded_text).strip()

            decoded_samples += f"{i + 1}: " + decoded_text + "\n\n"
        return decoded_samples

    def _compute_data_curriculum_difficulty_metrics_for_logging(self):
        """
        Compute dynamic difficulty metrics during data curriculum learning for logging
        """
        # Get pacing function and difficulty scorer
        sampler = self.callback_handler.train_dataloader.sampler
        pacing_fn = self.callback_handler.train_dataloader.sampler.pacing_fn
        difficulty_scorer = self.callback_handler.train_dataloader.sampler.difficulty_scorer

        current_sampler_step = getattr(sampler, 'global_stepnum', self.state.global_step)

        # Percentile of current pacing
        # Calculates the currently aimed for difficulty percentile based on the current global step
        data_difficulty_percentile = pacing_fn(current_sampler_step)

        # Filtered difficulty scores for current batch
        difficulty_scores = torch.tensor(
            difficulty_scorer.filtered_difficulty_scores
        )
        # Just keep non-zero scores for further calculations
        difficulty_scores = difficulty_scores[difficulty_scores != 0]

        # Adjust for distributed training
        # Calculate the actual difficulty percentile
        num_samples = difficulty_scores.shape[0] * self.args.world_size

        data_sampled_percentile = num_samples / self.train_dataset.num_rows
        max_score = difficulty_scores.max().item()
        min_score = difficulty_scores.min().item()
        median_score = difficulty_scores.median().item()

        current_stage = getattr(difficulty_scorer, "current_stage", 1.0)

        return (
            data_difficulty_percentile,
            data_sampled_percentile,
            num_samples,
            max_score,
            min_score,
            median_score,
            current_stage
        )

    def _check_if_curriculum_table_should_be_logged(self):
        """Check if we should log the curriculum learning table at this step."""
        return (
                self.args.eval_strategy == IntervalStrategy.STEPS
                and self.state.global_step % self.args.eval_steps == 0  # type: ignore
        )

    def evaluate(
            self,
            metric_key_prefix: str = "eval",  # Optional prefix for metric keys (e.g., "eval_bleu" vs. "test_bleu")
            **kwargs,  # Accepts additional keyword arguments for compatibility
    ) -> Dict[str, float]:
        """
        Override the Trainer.evaluate() method to evaluate on BLIMP and possibly other tasks.

        Args:
            metric_key_prefix (str): Prefix for all metric names (default "eval").
            **kwargs: Extra arguments for flexibility.

        Returns:
            Dict[str, float]: Metrics like loss, perplexity, task scores, speed, etc.
        """
        # Start tracking memory usage as early as possible
        # Hugging Face has a built-in memory tracker (_memory_tracker) to record GPU/CPU RAM consumption
        self._memory_tracker.start()
        # Record the time this evaluation starts
        start_time = time.time()
        # Flag to indicate whether this is a "best" run, i.e. final evaluation of best model
        is_best_run = "best" in metric_key_prefix

        metrics = {}
        metrics = self.evaluate_on_perplexity(metrics, metric_key_prefix)
        self._save_and_sync_model()
        metrics = self._evaluate_on_additional_tasks(metrics, metric_key_prefix, is_best_run)
        metrics = self._compute_speed_metrics(metrics, metric_key_prefix, start_time)
        metrics = self._record_best_model_step(metrics, metric_key_prefix, is_best_run)

        self.log(metrics)
        # Trigger callbacks for evaluation, notifies any registered callbacks (like EarlyStoppingCallback)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        # Stop memory tracking and update metrics
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def evaluate_on_perplexity(self, metrics: Dict[str, float], metric_key_prefix: str):
        """Evaluate and record perplexity metrics if enabled."""
        # If perplexity evaluation is disabled just return the initial metrics
        if not self.eval_perplexity:
            return metrics

        # Simulate perplexity calculation during debugging
        if self.skip_execution_of_eval_scripts_for_debugging:
            ppl_mean = self._simulate_perplexity_metrics()
        # Calculate perplexity on eval subset
        else:
            ppl_mean = self._compute_perplexity_from_dataset()

        # Add perplexity values to the metrics result dict
        metrics[f"{metric_key_prefix}_perplexity_mean"] = ppl_mean
        return metrics

    def _simulate_perplexity_metrics(self) -> float:
        """Generate simulated perplexity metrics for debugging purposes."""
        ppl_mean = random.uniform(10.0, 50.0)
        return ppl_mean

    def _compute_perplexity_from_dataset(self) -> float:
        """Compute mean and std of perplexity from evaluation dataset."""
        eval_subset = self._get_eval_subset()
        logging.info("Evaluating perplexity...")
        logging.info(f" ** Number of samples: {eval_subset.num_rows}")

        total_nll, total_tokens = self._run_perplexity_inference(eval_subset)

        # When the training is run in a distributed environment
        if self.is_distributed:
            total_nll, total_tokens = self._gather_distributed_metrics(total_nll, total_tokens)

        ppl_mean = math.exp(total_nll / total_tokens) if total_tokens > 0 else 0.0

        return ppl_mean

    def _run_perplexity_inference(self, eval_subset) -> Tuple[float, int]:
        """Run inference to compute perplexity scores for each batch."""
        eval_subset = prepare_dataset_for_ppl_inference(self, eval_subset)

        dataloader = DataLoader(
            eval_subset,  # type: ignore
            batch_size=4,
            shuffle=False,
            collate_fn=base_collate_fn,
            pin_memory=True,
        )

        total_nll = 0.0
        total_tokens = 0
        self.model.eval()

        for batch in tqdm(dataloader, desc="Evaluating Perplexity"):
            batch_nll, batch_tokens = compute_trainer_perplexity(
                batch, self.processing_class, self
            )
            total_nll += batch_nll
            total_tokens += batch_tokens

        return total_nll, total_tokens

    def _gather_distributed_metrics(self, total_nll, total_tokens):
        # Wait for all GPUs to finish their loops
        dist.barrier()

        # Sum everything together
        stats = torch.tensor([total_nll, float(total_tokens)], device=self.args.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total_nll, total_tokens = stats[0].item(), int(stats[1].item())

        # Final sync to ensure everyone has the reduced numbers before moving on
        dist.barrier()

        return total_nll, total_tokens

    def _get_eval_subset(self):
        """
        Retrieve a subset of the evaluation dataset for perplexity computation

        Returns:
            Subset of the evaluation dataset
        """
        eval_subset = self.eval_dataset.select(  # type: ignore
            range(
                self.args.process_index,  # local process rank
                self.eval_dataset.num_rows,  # type: ignore
                self.eval_dataset.num_rows // ((100 if self.dry_run else 10_000) // self.args.world_size),  # type: ignore
            )
        )
        return eval_subset

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override Trainer._save() to save the model and tokenizer.
        """
        if self.args.should_save:
            super()._save(output_dir=output_dir, state_dict=state_dict)

            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)

            # Save the (unwrapped) base model directly
            model_to_save = unwrap_model(self.model)
            model_to_save.save_pretrained(output_dir)

            # Save tokenizer
            self.processing_class.save_pretrained(output_dir)

    def _save_and_sync_model(self):
        """Save model checkpoint and synchronize across distributed processes."""
        self.save_model(self.args.output_dir, _internal_call=True)

        if self.args.world_size > 1:
            # In distributed settings, ensure all processes have the same model state
            dist.barrier()

    def _evaluate_on_additional_tasks(
            self, metrics: Dict[str, float], metric_key_prefix: str, is_best_run: bool
    ):
        """Evaluate on additional custom tasks and update metrics."""
        additional_metrics = {}

        inference_model_dir = os.path.join(self.args.output_dir, "lm_model")

        # Additional behaviour - evaluate on BLIMP
        if self.eval_blimp:
            logging.info("Evaluating on BLIMP...")
            zeroshot_evaluator = ZeroShotEvaluator(
                self.args.output_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                is_best_run=is_best_run,
                use_dummy_eval_data=self.skip_execution_of_eval_scripts_for_debugging,
                experiment_name=self.experiment_name,
                global_steps=self.state.global_step,
                evaluator_name="BLIMP"
            )
            # Get average of blimp metrics
            blimp_metrics = zeroshot_evaluator()
            additional_metrics.update(blimp_metrics)  # type: ignore

        if self.eval_glue:
            logging.info("Evaluating on SUPERGLUE...")
            super_glue_evaluator = SuperGlueEvaluator(
                self.args.output_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                is_best_run=is_best_run,
                use_dummy_eval_data=self.skip_execution_of_eval_scripts_for_debugging,
                experiment_name=self.experiment_name,
                global_steps=self.state.global_step,
                evaluator_name="SUPER_GLUE",
                task_prefix_to_add="superglue",
            )
            # Get average of blimp metrics
            super_glue_metrics = super_glue_evaluator()
            additional_metrics.update(super_glue_metrics)

        # Ensure that every metric begins with 'metric_key_prefix'
        for key in list(additional_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                additional_metrics[f"{metric_key_prefix}_{key}"] = (
                    additional_metrics.pop(key)
                )
        metrics.update(additional_metrics)
        return metrics

    def _compute_speed_metrics(
            self, metrics: Dict[str, float], metric_key_prefix: str, start_time: float
    ):
        """Compute speed-related metrics and update."""
        # Adjust start_time if JIT compilation time was recorded
        if f"{metric_key_prefix}_jit_compilation_time" in metrics:
            start_time += metrics[f"{metric_key_prefix}_jit_compilation_time"]
        # Add speed-related metrics
        metrics.update(speed_metrics(metric_key_prefix, start_time))
        return metrics

    def _record_best_model_step(
            self, metrics: Dict[str, float], metric_key_prefix: str, is_best_run: bool
    ):
        """Record the step number of the best model checkpoint."""
        if is_best_run:
            step = int(self.state.best_model_checkpoint.split("checkpoint-")[-1])
            metrics[f"{metric_key_prefix}_model_step"] = step
        return metrics