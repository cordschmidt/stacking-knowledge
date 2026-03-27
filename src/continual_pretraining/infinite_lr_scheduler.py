import logging
import math
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger("Continual Pretraining")

class InfiniteLRScheduler(LRScheduler):
    def __init__(self, optimizer, lr_max, lr_min, const_steps, total_max_steps, initial_last_stage_budget=None,
                 lr_const=None, last_epoch=-1):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_const = lr_const if lr_const is not None else (lr_max + lr_min) / 2.0
        self.const_steps = const_steps
        self.warmup_steps = max(1, int(0.01 * const_steps))
        self.total_max_steps = total_max_steps
        budget_for_decay = initial_last_stage_budget if initial_last_stage_budget is not None else total_max_steps
        self.decay_duration = max(1, int(0.15 * budget_for_decay))
        self.decay_start_step = self.total_max_steps - self.decay_duration
        self.in_dynamic_decay = False
        self.dynamic_end_step = None
        self.current_phase = None
        super().__init__(optimizer, last_epoch)

    def update_last_stage_budget(self, remaining_steps):
        if not self.in_dynamic_decay:
            self.decay_duration = max(1, int(0.15 * remaining_steps))
            self.decay_start_step = self.total_max_steps - self.decay_duration
            logger.info(
                f"LR Schedule: Dynamic budget updated. Decay phase will now start at step {self.decay_start_step} for {self.decay_duration} steps"
            )

    def force_decay_phase(self, current_step, steps_spent_in_last_stage):
        # Set decay duration in a way that it accounts for 15% of the total number of steps in the last stage
        self.decay_duration = max(1, int((3/17) * steps_spent_in_last_stage))
        self.decay_start_step = current_step
        self.in_dynamic_decay = True
        self.dynamic_end_step = current_step + self.decay_duration
        logger.info(
            f"LR Schedule: Forced exponential decay triggered at step {current_step} for {self.decay_duration} steps"
        )

    def get_lr(self):
        step = self.last_epoch

        # Determine the phase we are currently in
        if step >= self.decay_start_step:
            new_phase = "Exponential Decay"
        elif step < self.warmup_steps:
            new_phase = "Linear Warmup"
        elif step < self.const_steps:
            new_phase = "Cosine Annealing"
        else:
            new_phase = "Constant LR"

        # Log if we transitioned into a new phase
        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self._log_phase_change(new_phase)

        # Route to the appropriate lr calculation
        if new_phase == "Exponential Decay":
            return self._get_exponential_decay_lr(step)
        elif new_phase == "Linear Warmup":
            return self._get_linear_warmup_lr(step)
        elif new_phase == "Cosine Annealing":
            return self._get_cosine_annealing_lr(step)
        else:
            return self._get_constant_lr()

    def _log_phase_change(self, phase):
        """Helper method to log informative details about the new phase."""
        if phase == "Linear Warmup":
            logger.info(
                f"LR Schedule Phase Change: Entering '{phase}' (Target LR: {self.lr_max:.2e} at step {self.warmup_steps})"
            )
        elif phase == "Cosine Annealing":
            logger.info(
                f"LR Schedule Phase Change: Entering '{phase}' (Target LR: {self.lr_const:.2e} at step {self.const_steps})"
            )
        elif phase == "Constant LR":
            logger.info(
                f"LR Schedule Phase Change: Entering '{phase}' (Holding steady at {self.lr_const:.2e})"
            )
        elif phase == "Exponential Decay":
            logger.info(
                f"LR Schedule Phase Change: Entering '{phase}' (Target LR: {self.lr_min:.2e} over {self.decay_duration} steps)"
            )

    def _get_linear_warmup_lr(self, step):
        return [self.lr_max * (step / float(self.warmup_steps)) for _ in self.base_lrs]

    def _get_cosine_annealing_lr(self, step):
        progress = (step - self.warmup_steps) / float(max(1, self.const_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.lr_const + (self.lr_max - self.lr_const) * cosine_decay
        return [lr for _ in self.base_lrs]

    def _get_constant_lr(self):
        return [self.lr_const for _ in self.base_lrs]

    def _get_exponential_decay_lr(self, step):
        steps_into_decay = step - self.decay_start_step
        if steps_into_decay >= self.decay_duration:
            return [self.lr_min for _ in self.base_lrs]
        lr_start = self.lr_const if step < self.const_steps else self.lr_const
        decay_factor = (self.lr_min / lr_start) ** (steps_into_decay / float(self.decay_duration))
        return [lr_start * decay_factor for _ in self.base_lrs]