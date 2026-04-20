import logging
import math
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger("Continual Pretraining")

class InfiniteLRScheduler(LRScheduler):
    """
    A custom learning rate scheduler designed for continual pre-training environments
    with unpredictable stage lengths. It transitions through linear warmup, cosine
    annealing, a prolonged constant phase, and a dynamically triggered exponential decay.
    """
    def __init__(self, optimizer, lr_max, lr_min, const_steps, total_max_steps, initial_last_stage_budget=None,
                 lr_const=None, last_epoch=-1):
        """
        Initializes the InfiniteLRScheduler

        Args:
            optimizer: The optimizer for which to schedule the learning rate
            lr_max: The maximum peak learning rate reached after warmup
            lr_min: The minimum learning rate reached at the end of the exponential decay phase
            const_steps: The step at which the cosine annealing phase ends and the constant LR phase begins
            total_max_steps: The absolute maximum number of training steps scheduled
            initial_last_stage_budget: The estimated number of steps in the final data stage, used to calculate the initial decay duration
            lr_const: The constant learning rate to maintain. Defaults to the midpoint between lr_max and lr_min
            last_epoch: The index of the last epoch. Defaults to -1
        """
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
        """
        Updates the duration and starting step of the exponential decay phase based
        on the remaining steps in the final stage

        Args:
            remaining_steps: The dynamically calculated number of steps remaining in the training process

        Returns:
            None
        """
        if not self.in_dynamic_decay:
            self.decay_duration = max(1, int(0.15 * remaining_steps))
            self.decay_start_step = self.total_max_steps - self.decay_duration
            logger.info(
                f"LR Schedule: Dynamic budget updated. Decay phase will now start at step {self.decay_start_step} for {self.decay_duration} steps"
            )

    def force_decay_phase(self, current_step, steps_spent_in_last_stage):
        """
        Forces the learning rate scheduler to immediately enter the exponential decay phase,
        dynamically calculating its duration

        Args:
            current_step: The current global training step at which the decay is triggered
            steps_spent_in_last_stage: The number of steps already executed within the final training stage

        Returns:
            None
        """
        # Set decay duration in a way that it accounts for 15% of the total number of steps in the last stage
        self.decay_duration = max(1, int((3/17) * steps_spent_in_last_stage))
        self.decay_start_step = current_step
        self.in_dynamic_decay = True
        self.dynamic_end_step = current_step + self.decay_duration
        logger.info(
            f"LR Schedule: Forced exponential decay triggered at step {current_step} for {self.decay_duration} steps"
        )

    def get_lr(self):
        """
        Determines the current learning rate phase and calculates the appropriate learning rate

        Returns:
            A list containing the calculated learning rate for each parameter group in the optimizer
        """
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
        """
        Helper method to log details when transitioning to a new learning rate phase

        Args:
            phase: The name of the new phase being entered.

        Returns:
            None
        """
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
        """
        Calculates the learning rate during the linear warmup phase

        Args:
            step: The current training step

        Returns:
            A list containing the linear warmup learning rate for each parameter group
        """
        return [self.lr_max * (step / float(self.warmup_steps)) for _ in self.base_lrs]

    def _get_cosine_annealing_lr(self, step):
        """
        Calculates the learning rate during the cosine annealing phase, decaying from lr_max to lr_const

        Args:
            step: The current training step

        Returns:
            A list containing the annealed learning rate for each parameter group
        """
        progress = (step - self.warmup_steps) / float(max(1, self.const_steps - self.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = self.lr_const + (self.lr_max - self.lr_const) * cosine_decay
        return [lr for _ in self.base_lrs]

    def _get_constant_lr(self):
        """
        Retrieves the learning rate for the constant learning rate phase

        Returns:
            A list containing the constant learning rate for each parameter group
        """
        return [self.lr_const for _ in self.base_lrs]

    def _get_exponential_decay_lr(self, step):
        """
        Calculates the learning rate during the final exponential decay phase, decaying towards lr_min

        Args:
            step: The current training step

        Returns:
            A list containing the exponentially decayed learning rate for each parameter group
        """
        steps_into_decay = step - self.decay_start_step
        if steps_into_decay >= self.decay_duration:
            return [self.lr_min for _ in self.base_lrs]
        lr_start = self.lr_const if step < self.const_steps else self.lr_const
        decay_factor = (self.lr_min / lr_start) ** (steps_into_decay / float(self.decay_duration))
        return [lr_start * decay_factor for _ in self.base_lrs]