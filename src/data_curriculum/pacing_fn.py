"""
Module for establishing a pacing function for data-driven curriculum learning.
Used by the CurriculumSampler class to determine the upper limit of the sampling difficulty.
"""

from typing import Callable

def get_pacing_fn(
    pacing_fn_name: str,
    total_steps: int,
    start_percent: float, # Start curriculum e.g. at 10% of total steps
    end_percent: float, # Reach full difficulty e.g. by 70% of training steps
    starting_difficulty: float = 0.2, # Initially sample from bottom 20% easiest examples
    max_difficulty: float = 1.0, # Eventually allow sampling from full difficulty range
) -> Callable[[int], float]:
    """
    Modified from: https://github.com/google-research/understanding-curricula/blob/main/utils/utils.py

    Args:
        * pacing_fn_name (str): The name of the pacing function to use, one of:
            'linear', 'quad', 'root', 'step', 'exp', or 'log'
        * total_steps (int): The total number of steps in the training process, defining the pacing schedule's time frame
        * start_percent (float): The percentage of steps from the total number of steps that
            have been taken before we begin increasing the data difficulty. Determines after how many steps the curriculum learning starts / increases
        * end_percent (float): The percentage of steps from the total number of steps that
            have been taken after which we stop increasing the data difficulty. Determines when the curriculum learning stops
        * starting_difficulty (float): The starting difficulty of the dataset as a percentile of
            the dataset's difficulty. A value of 0.2 means that initially, we sample from the
            bottom 20% difficult examples.
        * max_difficulty (float): The maximum difficulty of the dataset as a percentile of
            the dataset's difficulty. A value of 1.0 means that the maximum difficulty we
            can sample is the maximum difficulty in the dataset.

    Returns:
        * (callable): A function that takes in the current step and returns the number of
            data points to use.

    """

    # Ensure pacing window is logically valid
    assert (
            start_percent < end_percent
    ), f"For the Pacing Fn: start_percent ({start_percent}) must be less than end_percent ({end_percent})"

    # Convert percent-of-training to absolute step numbers
    step_start = start_percent * total_steps
    step_end = end_percent * total_steps

    # Number of steps between start and end of pacing
    num_steps = int(step_end - step_start)

    # === LINEAR pacing function ===
    if pacing_fn_name == "linear":
        # Increase linearly from starting to max difficulty
        rate = (max_difficulty - starting_difficulty) / num_steps

        def _linear_function(step: int):
            # Before pacing starts, use starting difficulty
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start
            # Linearly increase difficulty, clipped to max
            return float(min(rate * step_diff + starting_difficulty, max_difficulty))

        return _linear_function

    # === QUADRATIC pacing === (slow start, rapid increase)
    elif pacing_fn_name == "quad":
        rate = (max_difficulty - starting_difficulty) / (num_steps) ** 2

        def _quad_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start
            return float(min(rate * step_diff ** 2 + starting_difficulty, max_difficulty))

        return _quad_function

    # === ROOT pacing === (fast start, slow finish)
    elif pacing_fn_name == "root":
        rate = (max_difficulty - starting_difficulty) / (num_steps) ** 0.5

        def _root_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start
            return float(min(rate * step_diff ** 0.5 + starting_difficulty, max_difficulty))

        return _root_function

    # === STEP pacing === (no increase until end step, then jump)
    elif pacing_fn_name == "step":

        def _step_function(step):
            if step < step_end:
                return starting_difficulty
            else:
                return max_difficulty

        return _step_function

    # === EXPONENTIAL pacing === (very slow start, aggressive increase near end)
    elif pacing_fn_name == "exp":
        import numpy as np

        c = 10  # Controls the sharpness of the curve
        tilde_b = starting_difficulty
        tilde_a = num_steps
        rate = (max_difficulty - tilde_b) / (np.exp(c) - 1)
        constant = c / tilde_a

        def _exp_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start
            return float(
                min(rate * (np.exp(step_diff * constant) - 1) + tilde_b, max_difficulty)
            )

        return _exp_function

    # === LOGARITHMIC pacing === (aggressive early increase, flattens toward end)
    elif pacing_fn_name == "log":
        import numpy as np

        c = 10
        tilde_b = starting_difficulty
        tilde_a = num_steps
        ec = np.exp(-c)
        N_b = max_difficulty - tilde_b

        def _log_function(step):
            if step < step_start:
                return starting_difficulty

            step_diff = step - step_start
            return min(
                N_b * (1 + (1.0 / c) * np.log(step_diff / tilde_a + ec)) + tilde_b,
                max_difficulty,
            )

        return _log_function

    else:
        # Default fallback: use max difficulty from start (no pacing)
        return lambda step: 1.0
