"""
StagedDataSplitSorter difficulty scorer.

- Implements STRICT STAGE training (only the current stage is active, previous/future are 0).
- OPTIMIZED for efficiency: Caches the mask to avoid re-computing every step.
- Uses a CUSTOM dataset order.
"""
from __future__ import annotations

import logging
import math

import torch
import numpy as np
from typing import Any, Sequence

from datasets import Dataset

# Import base classes
from .base_difficulty_scorer import BaseDifficultyScorer
from .registry import register_difficulty_scorer

# Import dataset / stage info
from .stages import CUSTOM_STAGED_ORDER, NUM_STAGES

data_cl_logger = logging.getLogger("Data Curriculum")

@register_difficulty_scorer("staged_data_split")
class StagedDataSplitSorter(BaseDifficultyScorer):
    """
    A difficulty scorer that implements distinct stage training.
    """

    def __init__(self, account_for_dataset_proportions: bool, **kwargs: Any):
        super().__init__(**kwargs)
        # Use custom order
        self.filename_to_difficulty_map = CUSTOM_STAGED_ORDER
        self.account_for_dataset_proportions = account_for_dataset_proportions

        # Optimization caches
        self._difficulty_scores_tensor = None
        self._last_active_level = None
        self.filtered_difficulty_scores = None

        # List which holds the percentiles at which a new stage begins
        self.transition_thresholds = []


    def score_difficulty(
            self,
            dataset: Dataset,
            indices: Sequence[int],
            global_stepnum: int,
            max_difficulty_percentile: float,
    ) -> Sequence[float]:

        indices_to_score = indices

        # Initialization of difficulty scores on first step
        if global_stepnum == 0:
            self._initialize_difficulty_scores(dataset=dataset, indices_to_score=indices_to_score)
            self._calculate_transition_thresholds(dataset=dataset)

        # Ensure initialization has happened
        if not hasattr(self, "_difficulty_scores_tensor") or self._difficulty_scores_tensor is None:
            raise RuntimeError("Difficulty scores were not initialized at step 0.")

        active_difficulty_level = self._determine_current_stage(max_difficulty_percentile)

        # If the stage hasn't changed since the last step, return the cached mask immediately
        if (self._last_active_level == active_difficulty_level) and (self.filtered_difficulty_scores is not None):
            return self.filtered_difficulty_scores

        # Otherwise update the filtered samples that should be considered for sampling
        else:
            data_cl_logger.info(f"Switching Curriculum Stage: Activating Difficulty Level {active_difficulty_level} at step {global_stepnum}")
            self._update_filtered_difficulty_scores_for_new_stage(active_difficulty_level)
            return self.filtered_difficulty_scores

    def _initialize_difficulty_scores(self, indices_to_score: Sequence[int], dataset: Dataset) -> None:
        data_cl_logger.info("Initializing Staged Data Split Scorer...")
        assert "filename" in dataset.column_names, "Dataset must have 'filename' column"
        self._difficulty_scores = self._get_difficulties_based_on_filename_mapping(indices_to_score=indices_to_score,
                                                                                   dataset=dataset)
        # Convert to tensor for vectorization
        self._difficulty_scores_tensor = torch.tensor(self._difficulty_scores, dtype=torch.float32)

    def _get_corpora_sizes_on_complete_dataset(self, dataset: Dataset):
        # Get the 'filename' column for every sample in the complete dataset
        filenames_for_every_sample = dataset["filename"]

        # Map every filename-value in the global dataset to its stage
        stage_for_every_sample = [
            self.filename_to_difficulty_map.get(filename_for_specific_sample) for filename_for_specific_sample in
            filenames_for_every_sample
        ]

        # Convert to tensor to count frequencies of stages 1 through NUM_STAGES
        stage_for_every_sample_tensor = torch.tensor(stage_for_every_sample, dtype=torch.long)

        # Calculate counts for stages 1 - 6 (torch starts indexing at 0, so we get stage 0 with 0 counts as well,
        # which we have to filter out, since our stages are 1-indexed)
        counts = torch.bincount(stage_for_every_sample_tensor, minlength=NUM_STAGES + 1)
        filtered_counts = counts[1:NUM_STAGES + 1]

        return filtered_counts

    def _calculate_transition_thresholds(self, dataset: Dataset) -> None:
        """
        Calculates the percentile thresholds where the stage transitions happen
        """
        if not self.account_for_dataset_proportions:
            # Equal duration for every stage, e.g. [1/6, 2/6, 3/6, 4/6, 5/6]
            self.transition_thresholds = [i / NUM_STAGES for i in range(1, NUM_STAGES)]
            data_cl_logger.info(f"Calculated equal-interval thresholds: {self.transition_thresholds}")
        else:
            counts_for_each_corpus = self._get_corpora_sizes_on_complete_dataset(dataset=dataset)
            total_samples = counts_for_each_corpus.sum()
            # Calculate cumulative sample sizes, normalize to percentiles
            cumulative_percentiles = torch.cumsum(counts_for_each_corpus, dim=0).float() / total_samples
            # Exclude 1.0, this threshold we don't need
            self.transition_thresholds = cumulative_percentiles[:-1].tolist()
            data_cl_logger.info(f"Calculated proportion-based thresholds: {self.transition_thresholds}")

    def _determine_current_stage(self, max_difficulty_percentile: float) -> int:
        """
        Determines current stage by iterating through list of percentiles and checking when the current difficulty
        percentile is smaller than one of our thresholds
        """
        for i, threshold in enumerate(self.transition_thresholds):
            if max_difficulty_percentile < threshold:
                return i + 1
        # If we exceed every threshold we are in the last stage
        return NUM_STAGES

    def _update_filtered_difficulty_scores_for_new_stage(self, active_difficulty_level: int) -> None:
        # Strict staging, filter for all samples with the active_difficulty_level of the current stage, where
        # samples belonging to the current stage take the value 1.0 and those that are not take the value 0.0
        mask = (self._difficulty_scores_tensor == active_difficulty_level).float()
        # Convert mask tensor to list
        self.filtered_difficulty_scores = mask.tolist()
        # Set _last_active_level to the current difficulty level
        self._last_active_level = active_difficulty_level

    def _get_difficulties_based_on_filename_mapping(self, indices_to_score: Sequence[int], dataset: Dataset):
        # Get difficulties for desired subset based on the filename mapping
        temp_difficulty_scores = []

        for dataset_idx in indices_to_score:
            sample_to_score = dataset[dataset_idx]
            current_filename = sample_to_score["filename"]
            difficulty = self.filename_to_difficulty_map.get(current_filename)
            temp_difficulty_scores.append(difficulty)
        return temp_difficulty_scores
