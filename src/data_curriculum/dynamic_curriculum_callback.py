import logging
from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger("Data Curriculum")


class DynamicCurriculumCallback(TrainerCallback):
    """
    A callback that actively monitors the dev dataset for perplexity plateauing
    and dynamically forces the data curriculum to advance to the next stage to prevent overfitting
    """

    def __init__(self, trainer, dev_dataset, eval_steps, subset_size, patience):
        """
        Initializes the DynamicCurriculumCallback

        Args:
            trainer: The Hugging Face Trainer instance
            dev_dataset: The complete dev dataset used to evaluate perplexity
            eval_steps: The frequency (in training steps) at which perplexity is evaluated
            subset_size: The maximum number of samples to evaluate from the dev dataset to save compute
            patience: The number of consecutive evaluations with non-improving perplexity before triggering a stage transition

        Returns:
            None
        """
        self.trainer = trainer
        self.eval_steps = eval_steps
        self.patience = patience

        self.best_dev_ppl = float('inf')
        self.consecutive_increases = 0
        self.current_stage_start_step = 0
        self.last_stage_triggered = False

        num_rows = len(dev_dataset)
        world_size = trainer.args.world_size
        process_index = trainer.args.process_index
        target_samples_per_process = max(1, subset_size // world_size)
        step_size = max(1, num_rows // target_samples_per_process)
        self.dev_dataset = dev_dataset.select(range(process_index, num_rows, step_size))

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each training step to check for evaluation boundaries,
        calculate perplexity, and potentially trigger curriculum progression

        Args:
            args: The training arguments
            state: The current TrainerState containing step metrics
            control: The TrainerControl object to modify training flow
            **kwargs: Additional keyword arguments, including the training dataloader

        Returns:
            None
        """
        scheduler = self.trainer.lr_scheduler
        train_loader = kwargs.get("train_dataloader")

        # Check if we are in the dynamic decay phase and should terminate
        if self._handle_dynamic_decay_termination(scheduler, state, control):
            return

        # Get info of staged data split scorer
        scorer_info = self._get_scorer_info(train_loader=train_loader)
        scorer, current_stage, num_stages = scorer_info

        # Handle transitioning into the final stage
        self.update_last_stage_scheduler_budget_when_triggered(current_stage, num_stages, state, scheduler, args.max_steps)

        # Periodically evaluate dev perplexity and check for overfitting
        if self._is_eval_step(state):
            current_ppl = self._evaluate_dev_perplexity(state)
            self._check_and_handle_overfitting(current_ppl, current_stage, num_stages, scorer, state, scheduler)

    def _handle_dynamic_decay_termination(self, scheduler, state: TrainerState, control: TrainerControl) -> bool:
        """
        Checks if the training is currently in an exponential decay phase and should be terminated early

        Args:
            scheduler: The learning rate scheduler instance
            state: The current TrainerState
            control: The TrainerControl object

        Returns:
            A boolean indicating True if training was successfully terminated, False otherwise
        """
        if hasattr(scheduler, "in_dynamic_decay") and scheduler.in_dynamic_decay:
            if state.global_step >= scheduler.dynamic_end_step:
                logger.info("Dynamic exponential decay finished. Terminating training early")
                control.should_evaluate = True
                control.should_save = True
                control.should_training_stop = True
            return True
        return False

    def _get_scorer_info(self, train_loader):
        """
        Safely extracts the difficulty scorer and curriculum stage information from the dataloader

        Args:
            train_loader: The data loader used for training

        Returns:
            A tuple containing (scorer_instance, current_stage_integer, total_num_stages_integer)
        """
        scorer = train_loader.sampler.difficulty_scorer
        return scorer, scorer.current_stage, scorer.num_stages

    def update_last_stage_scheduler_budget_when_triggered(self, current_stage: int, num_stages: int, state: TrainerState, scheduler,
                                                          max_steps: int):
        """
        Informs the infinite learning rate scheduler if the final curriculum stage has been
        triggered so it can accurately calculate its final decay budget

        Args:
            current_stage: The current active curriculum stage
            num_stages: The total number of stages in the curriculum
            state: The current TrainerState
            scheduler: The learning rate scheduler instance
            max_steps: The absolute maximum number of training steps

        Returns:
            None
        """
        if current_stage == num_stages and not self.last_stage_triggered:
            self.last_stage_triggered = True
            self.current_stage_start_step = state.global_step
            if hasattr(scheduler, "update_last_stage_budget"):
                scheduler.update_last_stage_budget(max_steps - state.global_step)

    def _is_eval_step(self, state: TrainerState) -> bool:
        """
        Determines if the current global step aligns with the predefined evaluation frequency

        Args:
            state: The current TrainerState

        Returns:
            A boolean indicating True if an evaluation should occur, False otherwise
        """
        return state.global_step > 0 and state.global_step % self.eval_steps == 0

    def _evaluate_dev_perplexity(self, state: TrainerState) -> float:
        """
        Calculates and logs the mean perplexity of the model on the pre-selected development subset

        Args:
            state: The current TrainerState

        Returns:
            A float representing the calculated mean perplexity
        """
        logger.info(
            f"Step {state.global_step}: Calculating Dev Perplexity on subset of {len(self.dev_dataset)} samples..."
        )
        ppl_metrics = self.trainer._compute_perplexity_from_dataset(self.dev_dataset)
        current_ppl = ppl_metrics.get("perplexity_mean", float('inf'))
        logger.info(f"Step {state.global_step}): Dev PPL = {current_ppl:.4f}")

        self.trainer.log({"dynamic_curriculum/dev_perplexity": current_ppl})

        return current_ppl

    def _check_and_handle_overfitting(self, current_ppl: float, current_stage: int, num_stages: int, scorer,
                                      state: TrainerState, scheduler):
        """
        Tracks consecutive perplexity increases on the dev set and triggers either a stage
        progression or early termination if the patience threshold is exceeded

        Args:
            current_ppl: The newly calculated perplexity float
            current_stage: The current active curriculum stage
            num_stages: The total number of stages
            scorer: The difficulty scorer instance managing data transitions
            state: The current TrainerState
            scheduler: The learning rate scheduler instance

        Returns:
            None
        """
        if current_ppl > self.best_dev_ppl:
            self.consecutive_increases += 1
        else:
            self.best_dev_ppl = current_ppl
            self.consecutive_increases = 0

        if self.consecutive_increases >= self.patience:
            if current_stage < num_stages:
                self._force_next_stage(scorer, state, current_stage)
            else:
                self._force_early_termination(scheduler, state)

    def _force_next_stage(self, scorer, state: TrainerState, current_stage: int):
        """
        Forces the data curriculum scorer to advance prematurely due to detected overfitting

        Args:
            scorer: The difficulty scorer instance
            state: The current TrainerState
            current_stage: The current active curriculum stage

        Returns:
            None
        """
        logger.info(f"Overfitting! Forcing stage {current_stage} -> {current_stage + 1}")
        scorer.force_next_stage()
        self.current_stage_start_step = state.global_step
        self.consecutive_increases = 0
        self.best_dev_ppl = float('inf')

    def _force_early_termination(self, scheduler, state: TrainerState):
        """
        Triggers the final exponential decay phase of the learning rate scheduler if
        overfitting occurs during the final data stage

        Args:
            scheduler: The learning rate scheduler instance
            state: The current TrainerState

        Returns:
            None
        """
        logger.info("Overfitting in final stage, forcing early termination...")
        steps_spent = state.global_step - self.current_stage_start_step
        scheduler.force_decay_phase(state.global_step, steps_spent)