import math
import logging
import bisect
from transformers import TrainerCallback, get_scheduler

from src.config import BabyLMConfig
from src.data_curriculum.difficulty_scorer.stages import NUM_STAGES

from src.data_curriculum.pacing_fn import get_pacing_fn

logger = logging.getLogger("Continual Pretraining")


class ContinualPretrainingCallback(TrainerCallback):
    def __init__(self, trainer, cfg: BabyLMConfig):
        self.trainer = trainer

        # Get the raw transition steps (e.g., [1000, 3000])
        self.stage_transition_boundaries = self._get_staged_boundaries(cfg)

        # Define the start and end of every stage window
        total_training_budget = cfg.trainer.max_training_steps

        # A stage starts where the previous one ended (Stage 0 starts at step 0)
        starts = [0] + self.stage_transition_boundaries
        # A stage ends where the next one begins (The final stage ends at max_steps)
        ends = self.stage_transition_boundaries + [total_training_budget]

        # Calculate how many steps each stage actually lasts
        self.stage_durations = []
        for stage_start, stage_end in zip(starts, ends):
            duration = stage_end - stage_start
            self.stage_durations.append(duration)

        self.rewarm_steps_per_stage = cfg.continual_pretraining.rewarm_steps
        self.last_active_stage = -1

    def on_step_begin(self, args, state, control, **kwargs):
        current_global_step = state.global_step

        # Use bisect to find which "bucket" the current step falls into
        # If boundaries are [1000, 3000]:
        #   Step 500 -> index 0
        #   Step 1500 -> index 1
        #   Step 3500 -> index 2
        current_stage = bisect.bisect_right(self.stage_transition_boundaries, current_global_step)

        # Check if we have crossed into a new stage
        if current_stage != self.last_active_stage:
            self.last_active_stage = current_stage
            steps_in_this_stage = self.stage_durations[current_stage]

            # Re-initialize the Hugging Face scheduler for the duration of this specific stage
            self.trainer.lr_scheduler = get_scheduler(
                name=args.lr_scheduler_type,
                optimizer=self.trainer.optimizer,
                num_warmup_steps=self.rewarm_steps_per_stage,
                num_training_steps=steps_in_this_stage
            )

            logger.info(
                f"--- Continual Pre-Training: Stage Transition Detected ---"
                f"Now starting Stage {current_stage + 1} at global step {current_global_step}. "
                f"Stage duration: {steps_in_this_stage} steps."
            )

    def _get_staged_boundaries(self, cfg: BabyLMConfig):
        """
        Calculates the step numbers where the pacing function triggers
        a dataset stage transition.
        """
        total_steps = cfg.trainer.max_training_steps
        p_fn = get_pacing_fn(
            cfg.data_curriculum.pacing_fn_name,
            total_steps,
            **cfg.data_curriculum.pacing_fn_kwargs
        )

        boundaries = []
        current_stage = 1
        # Check transitions across the total training budget
        for step in range(total_steps):
            percentile = p_fn(step)
            # Match StagedDataSplitSorter logic: min(NUM_STAGES, floor(p * NUM_STAGES) + 1)
            stage_at_step = min(NUM_STAGES, math.floor(percentile * NUM_STAGES) + 1)
            if stage_at_step > current_stage:
                boundaries.append(step)
                current_stage = stage_at_step
        return boundaries