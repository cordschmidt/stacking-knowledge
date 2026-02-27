import math
import logging
import bisect

from transformers import TrainerCallback, get_scheduler

from src.config import BabyLMConfig

logger = logging.getLogger("Continual Pretraining")


class LearningRateResetCallback(TrainerCallback):
    def __init__(self, trainer, cfg: BabyLMConfig):
        self.trainer = trainer
        self.cfg = cfg
        self.rewarm_steps_per_stage = cfg.continual_pretraining.rewarm_steps
        self.rewarm_fraction = cfg.continual_pretraining.rewarm_fraction
        self.total_training_budget = self.cfg.trainer.max_training_steps
        self.max_rewarm_lr = cfg.continual_pretraining.max_rewarm_lr
        self.base_max_lr = cfg.trainer.lr

        self._is_initialized = False
        self.stage_durations = []
        self.last_active_stage = -1


    def on_step_begin(self, args, state, control, **kwargs):

        if not self._is_initialized:
            self._initialize_stage_boundaries_from_dataloader(train_dataloader=kwargs.get("train_dataloader"))

        current_global_step = state.global_step

        # Use bisect to find which "bucket" the current step falls into
        # If boundaries are [1000, 3000]:
        #   Step 500 -> index 0
        #   Step 1500 -> index 1
        #   Step 3500 -> index 2
        current_stage = bisect.bisect_right(self.step_boundaries , current_global_step)

        # Check if we have entered a new stage
        if current_stage != self.last_active_stage:
            self.last_active_stage = current_stage
            self._reset_learning_rate_scheduler_for_new_stage(current_stage=current_stage, current_global_step=current_global_step, lr_scheduler_type=args.lr_scheduler_type)

    def _initialize_stage_boundaries_from_dataloader(self, train_dataloader):
        self._set_stage_boundaries_in_callback(train_dataloader)
        self._determine_stage_durations()
        self._is_initialized = True

    def _set_stage_boundaries_in_callback(self, train_dataloader):
        """
        Calculates the step numbers where the pacing function triggers
        a dataset stage transition.
        """

        # Access difficulty scorer
        staged_difficulty_scorer = train_dataloader.sampler.difficulty_scorer
        pacing_fn = train_dataloader.sampler.pacing_fn
        threshold_percentiles = staged_difficulty_scorer.transition_thresholds

        step_boundaries = []

        for specific_percentile in threshold_percentiles:
            # Use binary search to find the first step where the pacing function
            # reaches or exceeds the threshold percentile
            low = 0
            high = self.total_training_budget
            boundary_step = high

            while low <= high:
                mid = (low + high) // 2
                pacing_value_of_mid = pacing_fn(mid)
                if pacing_value_of_mid >= specific_percentile:
                    boundary_step = mid
                    high = mid - 1
                else:
                    low = mid + 1

            step_boundaries.append(boundary_step)

        logger.info(f"Step boundaries: {step_boundaries}")

        self.step_boundaries = step_boundaries

    def _determine_stage_durations(self):
        # A stage starts where the previous one ended, first starts at step 0
        starts = [0] + self.step_boundaries
        # A stage ends where the next one begins, final stage ends at max_steps
        ends = self.step_boundaries + [self.total_training_budget]

        # Calculate how many steps each stage actually lasts
        self.stage_durations = []
        for stage_start, stage_end in zip(starts, ends):
            duration = stage_end - stage_start
            self.stage_durations.append(duration)

    def _reset_learning_rate_scheduler_for_new_stage(self, current_stage, current_global_step, lr_scheduler_type):

        steps_in_this_stage = self.stage_durations[current_stage]
        warmup_steps = self._calculate_warmup_steps(steps_in_this_stage=steps_in_this_stage)

        logger.info(f"--- Continual Pre-Training: Stage Transition Detected ---")
        logger.info(f"Now starting Stage {current_stage + 1} at global step {current_global_step}")
        logger.info(f"Stage duration: {steps_in_this_stage} steps")
        logger.info(f"Warmup: {warmup_steps} steps")

        self._set_max_learning_rate_for_current_stage(current_stage=current_stage)

        # Re-initialize the Hugging Face scheduler for the duration of this specific stage
        self.trainer.lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=self.trainer.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=steps_in_this_stage,
            scheduler_specific_kwargs= self.cfg.trainer.lr_scheduler_kwargs
        )

    def _calculate_warmup_steps(self, steps_in_this_stage):
        # Determine warmup steps, use fraction if available, otherwise use fixed steps
        if self.rewarm_fraction is not None:
            warmup_steps = math.ceil(self.rewarm_fraction * steps_in_this_stage)
        else:
            warmup_steps = self.rewarm_steps_per_stage if self.rewarm_steps_per_stage is not None else 0
        return warmup_steps

    def _set_max_learning_rate_for_current_stage(self, current_stage: int):
        # Use max_rewarm_lr for stages > 0 if provided
        if current_stage > 0 and self.max_rewarm_lr is not None:
            total_stages = len(self.stage_durations)

            if total_stages > 1:
                # Calculate how far along the stages we are (from 0.0 to 1.0)
                progress_fraction = current_stage / (total_stages - 1)
                # Linearly interpolate between base LR and the target rewarm LR
                max_lr = self.base_max_lr + (self.max_rewarm_lr - self.base_max_lr) * progress_fraction
            else:
                # If we only have 2 stages just set the new max_lr to max_rewarm_lr
                max_lr = self.max_rewarm_lr

            logger.info(f"Max Learning Rate: {self.max_rewarm_lr} for stage {current_stage + 1}")

            # Set the initial learning rate for every parameter
            for param_group in self.trainer.optimizer.param_groups:
                param_group['initial_lr'] = max_lr