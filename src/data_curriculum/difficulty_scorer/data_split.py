"""
DataSplitSorter difficulty scorer.

This class scores the difficulty of dataset samples based on a **predefined file-based order**
rather than a learned or statistical metric like perplexity.

Key idea:
- Each dataset sample has a filename associated with it.
- We assign a difficulty score based on which source file the sample came from.
- We can choose between two curriculum strategies:
    1. **Spoken-first**: Prioritize spoken language datasets (like transcripts and subtitles) as easier.
    2. **Grammatical-first**: Prioritize structured, grammatical text as easier.

This is useful for **curriculum learning**, where training starts with simpler data and
progressively moves to more difficult examples.

Unlike model-based scorers, this one:
- Does not require a tokenizer or trainer
- Does not perform dynamic scoring based on model performance
- Uses a **fixed difficulty order** of files for the entire training process
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

# typing imports
from datasets import Dataset

# Import the base scorer and registry for registering this scorer
from .base_difficulty_scorer import BaseDifficultyScorer
from .registry import register_difficulty_scorer

# Logger for curriculum learning progress
data_cl_logger = logging.getLogger("Data Curriculum")


# =============================
# Predefined difficulty orders
# =============================

# TODO: Have to change this in order to make it compatible with the 2025 dataset

# Difficulty order for spoken-first curriculum
# Lower numbers mean "easier" datasets to start training on
SPOKEN_FIRST_DATASET_ORDER = {
    "childes.train": 1,
    "bnc_spoken.train": 2,
    "switchboard.train": 2,
    "open_subtitles.train": 3,
    "qed.train": 3,
    "cbt.train": 4,
    "children_stories.train": 4,
    "simple_wiki.train": 5,
    "wikipedia.train": 6,
    "gutenberg.train": 6,
}

# Difficulty order for grammatical-first curriculum
GRAMMATICAL_FIRST_DATASET_ORDER = {
    "cbt.train": 1,
    "children_stories.train": 1,
    "simple_wiki.train": 2,
    "wikipedia.train": 3,
    "gutenberg.train": 3,
    "open_subtitles.train": 4,
    "bnc_spoken.train": 5,
    "switchboard.train": 5,
    "qed.train": 6,
    "childes.train": 6,
}

# ========================================================
# Difficulty scorer class that uses static dataset ordering
# ========================================================

@register_difficulty_scorer("data_split")
class DataSplitSorter(BaseDifficultyScorer):
    """
    A difficulty scorer that assigns difficulty based on a fixed file-based ordering
    """
    def __init__(self, spoken_first: bool, **kwargs: Any):
        """
        Initialize the DataSplitSorter

        Args:
            spoken_first (bool):
                If True, use the SPOKEN_FIRST_DATASET_ORDER
                If False, use the GRAMMATICAL_FIRST_DATASET_ORDER
            **kwargs (Any): Forwarded to BaseDifficultyScorer (e.g., `uniform_sampling`)
        """
        super().__init__(**kwargs)
        self.filename_map = (
            SPOKEN_FIRST_DATASET_ORDER
            if spoken_first
            else GRAMMATICAL_FIRST_DATASET_ORDER
        )

    def score_difficulty(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        global_stepnum: int,
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        """
        Scores the difficulty of the dataset according to a fixed data split order and returns a sequence of scores.

        This function assigns a numeric difficulty score to each selected sample based on
        which file it originated from. Files are ordered by either spoken-first or
        grammatical-first curriculum.

        Args:
            * dataset (Dataset): The dataset to score, must include a "filename" column
            * indices (Sequence[int]): The indices of the dataset to score
                (in the same order as the dataset). This is used for distributed training, where
                the dataset is split across multiple processes, and each process only has a subset
                of the dataset.
            * global_stepnum (int): The global step number of the training loop
            * max_difficulty_percentile (float): The maximum difficulty percentile to use
        Returns:
            * filtered_difficulty_scores: A list of difficulty scores that correspond to the
                difficulty of each sample in the passed in dataset (in the same order as the dataset).
                The difficulty scores that are above the max_difficulty_percentile should be set
                to 0.
        """

        # Compute difficulty scores only at the start of training
        # because the data order does not change dynamically
        if global_stepnum == 0:

            # Ensure the dataset has filenames for mapping difficulty
            assert (
                "filename" in dataset.column_names
            ), "Dataset must contain file names to use Data Split difficulty scorer"

            # Initialize an internal list to store computed difficulty scores
            self._difficulty_scores: Sequence[float] = []

            # indices is a list of indices that we want to score the difficulty of
            # (if we are using distributed training, not all indices will be scored - only those
            # assigned to the current process)
            curr_indices_idx = 0

            data_cl_logger.info(
                "Scoring difficulty according to fixed data split order"
            )

            # Iterate over the full dataset
            for _idx, item in enumerate(dataset):
                # Only score items corresponding to the requested subset of indices
                if _idx == indices[curr_indices_idx]:
                    # Look up the difficulty score based on the sample's filename
                    # TODO: Isn't this then an INT value? Then the difficulty-threshold might be affecting most of the samples?
                    # TODO: Check interaction here with pacing_fn.py, I think this and the config are determining the max_difficulty_percentile
                    difficulty = self.filename_map[item["filename"]]  # type: ignore
                    self._difficulty_scores.append(difficulty)

                    curr_indices_idx += 1

                    # Stop early if we have scored all requested indices
                    if curr_indices_idx == len(indices):
                        break

        # Ensure that difficulty scores exist before filtering
        assert hasattr(
            self, "_difficulty_scores"
        ), "Difficulty scores have not been computed but about to filter them."

        # Filter out scores above the max_difficulty_percentile
        self.filtered_difficulty_scores = (
            self.remove_scores_above_max_difficulty(
                self._difficulty_scores, max_difficulty_percentile
            )
        )

        return self.filtered_difficulty_scores

