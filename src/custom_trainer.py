import copy
import logging
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
# Data curriculum related
from .data_curriculum.datasampler import CurriculumSampler, DistributedCurriculumSampler
from .data_curriculum.difficulty_scorer import get_difficulty_scorer
from .data_curriculum.pacing_fn import get_pacing_fn
from .dataloader import CurriculumDataLoader

from .evaluator import ZeroShotEvaluator, FinetuneEvaluator
from src.helper.dataset_preprocessor import base_collate_fn
from src.helper.inference import compute_trainer_perplexity, prepare_dataset_for_ppl_inference

# Set up logging for different components of the trainer
logger = logging.getLogger(__name__)
data_cl_logger = logging.getLogger("Data Curriculum")

class GradualStackingCallback(TrainerCallback):
    def __init__(self, total_training_steps: int, k_number_of_stages: int, alpha: float, layer_per_block: int):
        # Validate arguments
        if not isinstance(total_training_steps, int) or total_training_steps <= 0:
            raise ValueError(f"total_training_steps must be a positive integer, got {total_training_steps}")
        if not isinstance(k_number_of_stages, int) or k_number_of_stages <= 1:
            raise ValueError(f"k_number_of_stages must be an integer greater than 1, got {k_number_of_stages}")
        if not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be a number, got {alpha}")
        if not isinstance(layer_per_block, int) or layer_per_block <= 0:
            raise ValueError(f"layer_per_block must be a positive integer, got {layer_per_block}")
        if total_training_steps < k_number_of_stages:
            raise ValueError(
                f"total_training_steps ({total_training_steps}) must be at least as large as "
                f"k_number_of_stages ({k_number_of_stages}) so that each stage has >= 1 step"
            )
        # Set values for prop-alpha schedule
        self.total_training_steps = total_training_steps
        self.k_number_of_stages = k_number_of_stages
        self.alpha = alpha
        self.block_size = layer_per_block
        # Prepare set for tracking the steps at which the model has been grown
        self._grown_steps = set()
        # Calculate the growing schedule
        self.steps_at_which_model_should_be_grown = self._compute_prop_alpha_growing_step_schedule()

    def _compute_prop_alpha_growing_step_schedule(self):
        """
        Computes the gradual stacking growing schedule, i.e. the steps at which the model should be grown based on the prop-alpha schedule.

        Based on Saunshi et al. (2024), *On the inductive bias of stacking towards improving reasoning*):

        "For a total training budget of T steps, the schedule Prop-α spends time Tᵢ in each stage such that:

            Tᵢ ∝ i^α   for all stages i ∈ [k]

        Thus:

        Tᵢ = (i^α / Σⱼ j^α) * T"
        """
        # Calculate unnormalized weights i^α for all stages (numerator)
        unnormalized_weights = [i ** self.alpha for i in range(1, self.k_number_of_stages + 1)]
        # Sum up all weights Σⱼ j^α (denominator)
        sum_of_weights = sum(unnormalized_weights)
        # Calculate the training budget Tᵢ for each stage i
        number_of_training_steps_per_stage = [int(unnormalized_weight / sum_of_weights * self.total_training_steps) for unnormalized_weight in unnormalized_weights]
        # Compute cumulative endpoints and exclude final stage
        gradual_stacking_growing_step_schedule = []
        cumulative_steps = 0
        for number_of_training_steps_in_stage_i in number_of_training_steps_per_stage[:-1]:
            cumulative_steps += number_of_training_steps_in_stage_i
            gradual_stacking_growing_step_schedule.append(cumulative_steps)

        return gradual_stacking_growing_step_schedule

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):

        # Check that model and optimizer is there as we need them for gradual stacking
        assert model is not None and optimizer is not None, "GradualStackingCallback was called without model/optimizer. This should not happen"

        # Only grow model at the given steps based on the prop-alpha schedule
        if state.global_step <= 0 or state.global_step not in self.steps_at_which_model_should_be_grown:
            return
        # Only grow once per step
        if state.global_step in self._grown_steps:
            return

        # As the model will be initialized with the block size and only extended by duplicating whole blocks,
        # this should always be true
        if len(model.model.layers) % self.block_size != 0:
            raise ValueError(
                f"Number of layers ({len(model.model.layers)}) is not divisible by block size ({self.block_size}). "
                "This should not happen and might be caused by a bug inside the code"
            )

        # Log model stats before growth
        logger.info(f"Growing model at step {state.global_step}")
        total_params_before_growth = sum(p.numel() for p in model.parameters())
        logger.info(f"No. of layers before growth: {len(model.model.layers)}, total params before growth: {total_params_before_growth}")

        # Track that the model has been grown at this global step
        self._grown_steps.add(state.global_step)

        # Synchronize all processes to ensure simultaneous growth
        if args.world_size > 1:
            dist.barrier()

        duplicated_middle_block = self._duplicate_middle_block(model)

        self._register_new_parameters_in_optimizer(optimizer, duplicated_middle_block)

        # Synchronize again after growth
        if args.world_size > 1:
            dist.barrier()

    def _duplicate_middle_block(self, model):
        # Divide model layers into blocks
        blocks = [model.model.layers[i:i + self.block_size] for i in range(0, len(model.model.layers), self.block_size)]

        # Determine middle block
        middle_block_idx = math.ceil(len(blocks) / 2) - 1  # Use ceiling function as in MIDAS paper and subtract 1 for indexing lists correctly

        # Duplicate the layers of the middle block
        duplicated_middle_block = [copy.deepcopy(layer) for layer in blocks[middle_block_idx]]
        # Determine layer indices for logging
        middle_block_layer_indices = list(range(middle_block_idx * self.block_size, (middle_block_idx + 1) * self.block_size))

        # Insert the duplicated block right after the middle block
        insert_position = sum(len(block) for block in blocks[:middle_block_idx + 1])
        for i, layer in enumerate(duplicated_middle_block):
            model.model.layers.insert(insert_position + i, layer)

        # Update number of hidden layers
        model.config.num_hidden_layers += len(duplicated_middle_block)

        # Determine total no. of parameters after duplicating
        total_params_after_growth = sum(p.numel() for p in model.parameters())

        # Logging: include block and layer info
        logger.info(
            f"Duplicated block {middle_block_idx} out of {list(range(len(blocks)))} "
            f"with the initial layer indices {middle_block_layer_indices}, "
            f"new no. of layers: {len(model.model.layers)}, "
            f"total params: {total_params_after_growth}"
        )
        return duplicated_middle_block

    def _register_new_parameters_in_optimizer(self, optimizer, duplicated_middle_block):
        for layer in duplicated_middle_block:
            optimizer.param_groups[0]["params"].extend(list(layer.parameters()))
            for p in layer.parameters():
                if p.requires_grad and p not in optimizer.state:
                    optimizer.state[p] = {}


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
        self.eval_msgs = hydra_config.trainer.eval_msgs
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
                                                        layer_per_block = self.hydra_config.gradual_stacking.layer_per_block)
            self.add_callback(stacking_callback)

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
                    ) = self._compute_data_curriculum_difficulty_metrics_for_logging()
                else:
                    # Default metrics if no data curriculum is done
                    data_difficulty_percentile = 1.0
                    data_sampled_percentile = 1.0
                    num_samples = len(self.callback_handler.train_dataloader.sampler) * self.args.world_size  # type: ignore
                    max_difficulty_score = 0.0
                    min_difficulty_score = 0.0
                    median_difficulty_score = 0.0

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
            decoded_samples += (
                    f"{i + 1}: " + self.processing_class.decode(inputs["input_ids"][i]) + "\n\n"
            )
        return decoded_samples

    def _compute_data_curriculum_difficulty_metrics_for_logging(self):
        """
        Compute dynamic difficulty metrics during data curriculum learning for logging
        """
        # Get pacing function and difficulty scorer
        pacing_fn = self.callback_handler.train_dataloader.sampler.pacing_fn
        difficulty_scorer = self.callback_handler.train_dataloader.sampler.difficulty_scorer

        # Percentile of current pacing
        # Calculates the currently aimed for difficulty percentile based on the current global step
        data_difficulty_percentile = pacing_fn(self.state.global_step)

        # Filtered difficulty scores for current batch
        # TODO: What are those? Is this affecting the order of the data curriculum or is it just for logging?
        difficulty_scores = torch.tensor(
            difficulty_scorer.filtered_difficulty_scores  # type: ignore
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

        return (
            data_difficulty_percentile,
            data_sampled_percentile,
            num_samples,
            max_score,
            min_score,
            median_score,
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
            ppl_mean, ppl_std = self._simulate_perplexity_metrics()
        # Calculate perplexity on eval subset
        else:
            ppl_mean, ppl_std = self._compute_perplexity_from_dataset()

        # Add perplexity values to the metrics result dict
        metrics[f"{metric_key_prefix}_perplexity_mean"] = ppl_mean
        metrics[f"{metric_key_prefix}_perplexity_std"] = ppl_std
        return metrics

    def _simulate_perplexity_metrics(self) -> Tuple[float, float]:
        """Generate simulated perplexity metrics for debugging purposes."""
        ppl_mean = random.uniform(10.0, 50.0)
        ppl_std = random.uniform(1.0, 5.0)
        return ppl_mean, ppl_std

    def _compute_perplexity_from_dataset(self) -> Tuple[float, float]:
        """Compute mean and std of perplexity from evaluation dataset."""

        # TODO: What subset? Why only subset?
        eval_subset = self._get_eval_subset()
        logging.info("Evaluating perplexity...")
        logging.info(f" ** Number of samples: {eval_subset.num_rows}")

        perplexities = self._run_perplexity_inference(eval_subset)
        tensor_perplexities = torch.tensor(perplexities, device=self.args.device)

        ppl_mean = torch.mean(tensor_perplexities)
        ppl_std = torch.std(tensor_perplexities)

        # When the training is run in a distributed environment
        if self.is_distributed:
            self._gather_distributed_metrics(ppl_mean, ppl_std)

        # if main process
        # TODO: How is this relating to the main process? No check is done?
        # TODO: Why should it be calculated again? Has this something to do with the main process thing?
        #  But isn't this also done on non-main processes as we're not checking for that?
        ppl_mean = torch.mean(torch.tensor(perplexities)).item()
        ppl_std = torch.std(torch.tensor(perplexities)).item()

        return ppl_mean, ppl_std

    def _run_perplexity_inference(self, eval_subset) -> List[float]:
        """Run inference to compute perplexity scores for each batch."""

        # TODO: What is done during eval prep here?
        eval_subset = prepare_dataset_for_ppl_inference(self, eval_subset)

        dataloader = DataLoader(
            eval_subset,  # type: ignore
            batch_size=4,
            shuffle=False,
            collate_fn=base_collate_fn,
            pin_memory=True,
        )

        perplexities = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # TODO: What is compute_trainer_perplexity?
                batch_ppl = compute_trainer_perplexity(batch, self.processing_class, self)
                perplexities.extend(batch_ppl)
        return perplexities

    def _gather_distributed_metrics(self, mean_tensor: torch.Tensor, std_tensor: torch.Tensor) -> None:
        """Aggregate metrics across distributed processes."""
        # TODO: This method I don't get at all, nothing is changed, so how is it supposed to work at all???
        dist.barrier()

        gathered_mean = [torch.zeros_like(mean_tensor) for _ in range(self.args.world_size)]
        gathered_std = [torch.zeros_like(std_tensor) for _ in range(self.args.world_size)]

        dist.all_gather(gathered_mean, mean_tensor)
        dist.all_gather(gathered_std, std_tensor)

    def _get_eval_subset(self):
        """
        Retrieve a subset of the evaluation dataset for perplexity computation

        Returns:
            Subset of the evaluation dataset
        """
        # TODO: Check, this was just copy pasted for now
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
        # TODO: Adjust for new eval pipeline
        additional_metrics = {}

        inference_model_dir = os.path.join(self.args.output_dir, "lm_model")

        # Additional behaviour - evaluate on BLIMP
        if self.eval_blimp:
            logging.info("Evaluating on BLIMP and AOA...")
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
            )
            # Get average of blimp metrics
            blimp_metrics = zeroshot_evaluator()
            additional_metrics.update(blimp_metrics)  # type: ignore

        if self.eval_glue or self.eval_msgs:
            # TODO: add for finetuning tasks?
            a = 1

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