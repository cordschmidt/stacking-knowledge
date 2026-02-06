from typing import List


class PropAlphaScheduler:
    """
    Computes the prop-alpha schedule logic used for both Gradual Stacking
    and synchronized Data Curriculum.
    """

    def __init__(self, total_training_steps: int, k_number_of_stages: int, alpha: float):
        self._validate_args(total_training_steps, k_number_of_stages, alpha)
        self.total_training_steps = total_training_steps
        self.k_number_of_stages = k_number_of_stages
        self.alpha = alpha

        # Compute the full list of boundaries (end of stage 1, end of stage 2, ..., end of stage k)
        self.stage_boundaries = self._compute_schedule()

    def _validate_args(self, total_training_steps: int, k_number_of_stages: int, alpha: float):
        if not isinstance(total_training_steps, int) or total_training_steps <= 0:
            raise ValueError(f"total_training_steps must be a positive integer, got {total_training_steps}")
        if not isinstance(k_number_of_stages, int) or k_number_of_stages <= 1:
            raise ValueError(f"k_number_of_stages must be an integer greater than 1, got {k_number_of_stages}")
        if not isinstance(alpha, (int, float)):
            raise ValueError(f"alpha must be a number, got {alpha}")
        if total_training_steps < k_number_of_stages:
            raise ValueError(
                f"total_training_steps ({total_training_steps}) must be at least as large as "
                f"k_number_of_stages ({k_number_of_stages})"
            )

    def _compute_schedule(self) -> List[int]:
        """
        Computes the gradual stacking growing schedule, i.e. the steps at which the model should be grown based on the prop-alpha schedule.

        Based on Saunshi et al. (2024), On the inductive bias of stacking towards improving reasoning:

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
        number_of_training_steps_per_stage = [int(unnormalized_weight / sum_of_weights * self.total_training_steps) for
                                              unnormalized_weight in unnormalized_weights]
        # Compute cumulative endpoints and exclude final stage
        gradual_stacking_growing_step_schedule = []
        cumulative_steps = 0
        for number_of_training_steps_in_stage_i in number_of_training_steps_per_stage[:-1]:
            cumulative_steps += number_of_training_steps_in_stage_i
            gradual_stacking_growing_step_schedule.append(cumulative_steps)
        return gradual_stacking_growing_step_schedule

    def get_growing_steps(self) -> List[int]:
        """
        Returns the steps at which the model should grow.
        """
        return self.stage_boundaries

    def get_current_stage(self, step: int) -> int:
        """
        Returns the 0-indexed stage number for a given global step.
        Useful for the Pacing Function to know 'where' in the curriculum we are.
        """
        for i, boundary in enumerate(self.stage_boundaries):
            if step < boundary:
                return i
        # If step >= total_steps, return the last stage index
        return self.k_number_of_stages - 1