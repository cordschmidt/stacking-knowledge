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
        # Use your custom order
        self.filename_to_difficulty_map = CUSTOM_STAGED_ORDER
        self.account_for_dataset_proportions = account_for_dataset_proportions

        # Optimization caches
        self._difficulty_scores_tensor = None
        self._last_active_level = None
        self.filtered_difficulty_scores = None


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

    def _determine_current_stage(self, max_difficulty_percentile: float) -> int:
        if self.account_for_dataset_proportions:
            return self._get_active_stage_considering_dataset_proportions(max_difficulty_percentile=max_difficulty_percentile)
        else:
            return self._get_active_stage_regardless_of_dataset_proportion(max_difficulty_percentile=max_difficulty_percentile)


    def _get_active_stage_regardless_of_dataset_proportion(self, max_difficulty_percentile: float) -> int:
        """
        This just maps the max_difficulty_percentile to one of the stages, where each stage has equal proportion regardless of dataset size
        """
        if max_difficulty_percentile <= 0.0:
            return 1
        else:
            # Map current max difficulty percentile to stages, starting from 1
            current_stage = math.floor(max_difficulty_percentile * NUM_STAGES) + 1
            # In case max_difficulty_precentile is 1.0
            return min(
                NUM_STAGES,
                current_stage
            )
    def _get_active_stage_considering_dataset_proportions(self, max_difficulty_percentile: float) -> int:
        """
        This takes into account the proportion of the dataset, so when the max_difficulty_percentile is 0.3,
        it will look what is the highest difficulty in 1/3 of the "easiest" data considering the difficulties in
        CUSTOM_STAGED_ORDER
        """
        active_difficulty_level = float(
            np.percentile(
                self._difficulty_scores_tensor.numpy(),
                max_difficulty_percentile * 100,
                interpolation='nearest'
            )
        )
        return active_difficulty_level

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

    def _initialize_difficulty_scores(self, indices_to_score : Sequence[int], dataset: Dataset) -> None:
        data_cl_logger.info("Initializing Staged Data Split Scorer...")
        assert "filename" in dataset.column_names, "Dataset must have 'filename' column"
        self._difficulty_scores = self._get_difficulties_based_on_filename_mapping(indices_to_score=indices_to_score,  dataset=dataset)
        # Convert to tensor for vectorization
        self._difficulty_scores_tensor = torch.tensor(self._difficulty_scores, dtype=torch.float32)
