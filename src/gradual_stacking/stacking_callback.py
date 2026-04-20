import copy
import logging
import math
import os

import torch.distributed as dist
from transformers import TrainerCallback
from src.gradual_stacking.scheduler import PropAlphaScheduler
from src.helper.visualization import create_layer_and_block_similarity_plots

logger = logging.getLogger("Gradual Stacking")


class GradualStackingCallback(TrainerCallback):
    """
    A callback that dynamically increases the depth of a neural network mid-training
    by duplicating its middle architectural blocks according to a predefined schedule
    """

    def __init__(self, total_training_steps: int, k_number_of_stages: int, alpha: float, layer_per_block: int, align_with_staged_data_curriculum: bool, cleaning_optimizer_state: bool):
        """
        Initializes the GradualStackingCallback

        Args:
            total_training_steps: The total number of training steps scheduled
            k_number_of_stages: The total number of architectural growth stages
            alpha: The pacing parameter that dictates the prop-alpha growth schedule
            layer_per_block: The number of layers in each architectural block to be duplicated
            align_with_staged_data_curriculum: Boolean flag to synchronize model growth with data curriculum stages
            cleaning_optimizer_state: Boolean flag to clear the optimizer momentum state after growth

        Returns:
            None
        """
        if not isinstance(layer_per_block, int) or layer_per_block <= 0:
            raise ValueError(f"layer_per_block must be a positive integer, got {layer_per_block}")
        self.block_size = layer_per_block
        self.align_with_staged_data_curriculum = align_with_staged_data_curriculum
        self.total_training_steps = total_training_steps
        self.cleaning_optimizer_state = cleaning_optimizer_state
        # Prepare set for tracking the steps at which the model has been grown
        self._grown_steps = set()
        # Initialize Prop-alpha scheduler
        self.scheduler = PropAlphaScheduler(
            total_training_steps=total_training_steps,
            k_number_of_stages=k_number_of_stages,
            alpha=alpha
        )
        # Calculate the growing schedule
        self.steps_at_which_model_should_be_grown = set(self.scheduler.get_growing_steps())
        self._curriculum_alignment_is_initialized = False

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Ensures boundaries are synced with the curriculum before the first step

        Args:
            args: The training arguments
            state: The current TrainerState containing the global step
            control: The TrainerControl object
            **kwargs: Additional keyword arguments, notably 'train_dataloader'

        Returns:
            None
        """
        if self.align_with_staged_data_curriculum and not self._curriculum_alignment_is_initialized:
            logger.info("Aligning Gradual Stacking stage boundaries with Staged Data Curriculum...")
            self._initialize_stage_boundaries_from_dataloader(train_dataloader=kwargs.get("train_dataloader"))

    def _initialize_stage_boundaries_from_dataloader(self, train_dataloader):
        """
        Initializes and aligns the model growth stage boundaries based on the data curriculum dataloader

        Args:
            train_dataloader: The dataloader containing the staged data curriculum components

        Returns:
            None
        """
        self._set_stage_boundaries_in_callback(train_dataloader)
        self._curriculum_alignment_is_initialized = True

    def _set_stage_boundaries_in_callback(self, train_dataloader):
        """
        Calculates the exact step numbers where the pacing function triggers
        a dataset stage transition and updates the gradual stacking schedule to match

        Args:
            train_dataloader: The dataloader providing access to the difficulty scorer and pacing function

        Returns:
            None
        """
        # Access difficulty scorer
        staged_difficulty_scorer = train_dataloader.sampler.difficulty_scorer
        pacing_fn = train_dataloader.sampler.pacing_fn
        threshold_percentiles = staged_difficulty_scorer.transition_thresholds

        step_boundaries = []
        for specific_percentile in threshold_percentiles:
            low = 0
            high = self.total_training_steps
            boundary_step = high

            while low <= high:
                mid = (low + high) // 2
                if pacing_fn(mid) >= specific_percentile:
                    boundary_step = mid
                    high = mid - 1
                else:
                    low = mid + 1
            step_boundaries.append(boundary_step)
        logger.info(f"Old step boundaries: {self.steps_at_which_model_should_be_grown}")
        self.steps_at_which_model_should_be_grown = step_boundaries
        logger.info(f"New step boundaries after alignment: {self.steps_at_which_model_should_be_grown}")

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """
        Called at the end of each training step. Verifies if the model should be grown
        based on the predetermined schedule and executes the duplication mechanism

        Args:
            args: The training arguments
            state: The current TrainerState containing the global step
            control: The TrainerControl object
            model: The model undergoing training
            optimizer: The optimizer handling model updates
            **kwargs: Additional keyword arguments

        Returns:
            None
        """
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
        results_dir = args.output_dir.replace("checkpoints/", "results/")
        create_layer_and_block_similarity_plots(
            model,
            output_dir=os.path.join(results_dir, f"checkpoint-{state.global_step}"),
            stage_name=f"end_of_stage_{self.scheduler.get_current_stage(state.global_step)}",
            step=state.global_step,
            block_size=self.block_size
        )

        # Log model stats before growth
        logger.info(f"Growing model before step {state.global_step}")
        total_params_before_growth = sum(p.numel() for p in model.parameters())
        logger.info(f"No. of layers before growth: {len(model.model.layers)}, total params before growth: {total_params_before_growth}")

        # Track that the model has been grown at this global step
        self._grown_steps.add(state.global_step)

        # Synchronize all processes to ensure simultaneous growth
        if args.world_size > 1:
            dist.barrier()

        original_middle_block, duplicated_middle_block = self._duplicate_middle_block(model)

        self._register_new_parameters_in_optimizer(optimizer, original_middle_block, duplicated_middle_block)

        # Synchronize again after growth
        if args.world_size > 1:
            dist.barrier()

    def _duplicate_middle_block(self, model):
        """
        Identifies the middle architectural block of the network, duplicates its layers
        via deepcopy and inserts them directly following the source block

        Args:
            model: The model being manipulated

        Returns:
            A tuple containing two lists: (original_middle_block_layers, duplicated_middle_block_layers)
        """
        # Divide model layers into blocks
        blocks = [model.model.layers[i:i + self.block_size] for i in range(0, len(model.model.layers), self.block_size)]

        # Determine middle block
        middle_block_idx = math.ceil(len(blocks) / 2) - 1  # Use ceiling function as in MIDAS paper and subtract 1 for indexing lists correctly

        # Extract original middle block
        original_middle_block = blocks[middle_block_idx]

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
        return original_middle_block, duplicated_middle_block

    def _register_new_parameters_in_optimizer(self, optimizer, original_middle_block, duplicated_middle_block):
        """
        Manually injects the newly created neural network parameters into the
        optimizer state to ensure they receive gradient updates during subsequent steps

        Args:
            optimizer: The training optimizer
            original_middle_block: The list of original source layers
            duplicated_middle_block: The list of newly duplicated layers

        Returns:
            None
        """
        # Iterate through all layers of the duplicated block
        for original_layer, duplicated_layer in zip(original_middle_block, duplicated_middle_block):
            # Iterate through all params
            for (original_param_name, original_param), (duplicated_param_name, duplicated_param) in zip(original_layer.named_parameters(), duplicated_layer.named_parameters()):
                if duplicated_param.requires_grad:
                    self._add_param_to_appropriate_optimizer_group(optimizer, duplicated_param, duplicated_param_name)
                    self._deepcopy_optimizer_state_from_param(optimizer, original_param, duplicated_param)
        if self.cleaning_optimizer_state:
            optimizer.state.clear()
            logger.info("Optimizer states have been cleared for the new stage")

    def _add_param_to_appropriate_optimizer_group(self, optimizer, duplicated_param, duplicated_param_name):
        """
        Routes a given parameter to the correct optimizer parameter group based on
        weight decay rules (e.g. separating biases and LayerNorm parameters from weights)

        Args:
            optimizer: The training optimizer
            duplicated_param: The parameter tensor being registered
            duplicated_param_name: The string name of the parameter tensor

        Returns:
            None
        """
        decay_group = optimizer.param_groups[0]["params"]
        no_decay_group = optimizer.param_groups[1]["params"] if len(optimizer.param_groups) > 1 else None
        if self._is_param_in_no_decay_group(no_decay_group, duplicated_param, duplicated_param_name):
            no_decay_group.append(duplicated_param)
        else:
            decay_group.append(duplicated_param)


    def _is_param_in_no_decay_group(self, no_decay_group, param, param_name):
        """
        Evaluates whether a parameter belongs to the weight decay exemption group

        Args:
            no_decay_group: The optimizer group designated for non-decaying parameters
            param: The parameter tensor
            param_name: The string name of the parameter

        Returns:
            A boolean indicating True if the parameter should be exempt from weight decay
        """
        is_decay_group_in_optimizer = no_decay_group is not None
        is_bias_param = (param.ndim == 1 or param_name.endswith(".bias"))
        if is_decay_group_in_optimizer and is_bias_param:
            return True
        else:
            return False

    def _deepcopy_optimizer_state_from_param(self, optimizer, original_param, duplicated_param):
        """
        Copies momentum and related tracking states from an original parameter
        in the optimizer to the corresponding duplicated parameter

        Args:
            optimizer: The training optimizer
            original_param: The source parameter whose state is being copied
            duplicated_param: The newly duplicated parameter inheriting the state

        Returns:
            None
        """
        # Deepcopy optimizer state from original parameter
        if original_param in optimizer.state:
            optimizer.state[duplicated_param] = copy.deepcopy(optimizer.state[original_param])
        else:
            # Fallback
            optimizer.state[duplicated_param] = {}
