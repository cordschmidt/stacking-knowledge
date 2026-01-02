"""  Implements an abstract base class for difficulty scorers.

It provides a standard interface for all difficulty scorers by defining:
1. A base class `BaseDifficultyScorer` with an abstract method `score_difficulty`.
2. A utility method to filter scores above a certain difficulty percentile.

All custom difficulty scorers should inherit from this class and implement
`score_difficulty` to provide their own difficulty calculation. """

from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset


class BaseDifficultyScorer(metaclass=ABCMeta):
    """
    Base class for all difficulty scorers.

    This class defines the interface and common functionality that all
    difficulty scorers should follow. It uses the `ABCMeta` metaclass to
    enforce that subclasses must implement certain abstract methods.
    """
    def __init__(self, uniform_sampling: bool = False):
        """
        Initializes the base scorer

        Args:
            uniform_sampling (bool):
                If True, difficulty scores will all be 1.0 instead of varying
                by sample difficulty. This is useful as a control experiment
                to simulate random/uniform sampling.
        """
        self.use_uniform_sampling = uniform_sampling

    def remove_scores_above_max_difficulty(self, difficulty_scores: Sequence[float],
                                           max_difficulty_percentile: float) -> Sequence[float]:
        """
        Filters out samples that are above a certain difficulty percentile.

        1. Computes the score threshold based on the given percentile
        2. Any sample whose difficulty exceeds this threshold is set to 0.0 (effectively excluded from training)
        3. If `uniform_sampling` is True, all scores are replaced with 1.0

        Args:
            difficulty_scores (Sequence[float]):
                A list/array of raw difficulty scores for each sample
            max_difficulty_percentile (float):
                The fraction (0.0 to 1.0) of samples to consider.
                For example, 0.9 means keep only the 90% easiest samples.

        Returns:
            Sequence[float]:
                A sequence of filtered difficulty scores. Scores above the
                computed percentile threshold are set to 0.0. Remaining scores
                are kept as-is or set to 1.0 if `uniform_sampling` is enabled.
        """

        # Computes the maximal difficulty score that that should be considered in this stage
        max_difficulty = float(
            np.percentile(difficulty_scores, max_difficulty_percentile * 100)
        )

        # Iterate over each score and:
        # 1. If the score exceeds the threshold -> set to 0.0 (filtered out)
        # 2. If uniform sampling is enabled -> set to 1.0 (ignore actual difficulty)
        # 3. Otherwise, keep the original score as a float
        _difficulty_scores = [
            0.0
            if score > max_difficulty
            else float(score)
            if not self.use_uniform_sampling
            else 1.0
            for score in difficulty_scores
        ]
        return _difficulty_scores

    @abstractmethod
    def score_difficulty(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        global_stepnum: int,
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        """
        Abstract method to score the difficulty of samples in a dataset. Each subclass must implement this to define how "difficulty" is calculated,
        which could be based on:
        - Model loss
        - Perplexity
        - Heuristic metrics
        - External signals

        Args:
            * dataset (Dataset): The dataset to score
            * indices (Sequence[int]): The indices of the dataset to score
                (in the same order as the dataset). This is used for distributed training, where
                the dataset is split across multiple processes, and each process only has a subset
                of the dataset.
            * global_stepnum (int): The global step number of the training loop
            * max_difficulty_percentile (float): The maximum difficulty percentile to use
        Returns:
            * filtered_difficulty_scores (Sequence[float]): A list of difficulty scores that correspond to the
                difficulty of each sample in the passed in dataset (in the same order as the dataset).
                The difficulty scores that are above the max_difficulty_percentile should be set
                to 0.

        """
        raise NotImplementedError
