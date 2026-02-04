import copy
import logging
import math
import torch.distributed as dist
from transformers import TrainerCallback
from src.gradual_stacking.scheduler import PropAlphaScheduler

logger = logging.getLogger(__name__)


class GradualStackingCallback(TrainerCallback):
    def __init__(self, total_training_steps: int, k_number_of_stages: int, alpha: float, layer_per_block: int):
        if not isinstance(layer_per_block, int) or layer_per_block <= 0:
            raise ValueError(f"layer_per_block must be a positive integer, got {layer_per_block}")
        self.block_size = layer_per_block
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