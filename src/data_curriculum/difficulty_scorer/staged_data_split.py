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
from torch import Tensor

# Import base classes
from .base_difficulty_scorer import BaseDifficultyScorer
from .registry import register_difficulty_scorer

# Import dataset / stage info
from .stages import CUSTOM_STAGED_ORDER, NUM_STAGES, DATASET_TOKEN_COUNTS

data_cl_logger = logging.getLogger("Data Curriculum")
cp_logger = logging.getLogger("Continual Pretraining")

@register_difficulty_scorer("staged_data_split")
class StagedDataSplitSorter(BaseDifficultyScorer):
    """
    A difficulty scorer that implements distinct stage training.
    """

    def __init__(self, proportion_mode: str = None, data_replay_mode: str = None, data_replay_fraction: float = 0.0, data_replay_decay: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        # Use custom order
        self.filename_to_difficulty_map = CUSTOM_STAGED_ORDER
        self.proportion_mode = proportion_mode  # Expects None, sample or token

        # Parameters for data replay
        self.data_replay_mode = data_replay_mode
        self.data_replay_fraction = data_replay_fraction
        self.data_replay_decay = data_replay_decay

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


    def _calculate_transition_thresholds(self, dataset: Dataset) -> None:
        """
        Calculates the percentile thresholds where the stage transitions happen
        """
        if self.proportion_mode is None:
            # Equal duration for every stage, e.g. [1/6, 2/6, 3/6, 4/6, 5/6]
            self.transition_thresholds = [i / NUM_STAGES for i in range(1, NUM_STAGES)]
            data_cl_logger.info(f"Calculated equal-interval thresholds: {self.transition_thresholds}")
        elif self.proportion_mode == "sample":
            counts_for_each_corpus = self._get_corpora_sample_sizes_on_complete_dataset(dataset=dataset)
            total_samples = counts_for_each_corpus.sum()
            # Calculate cumulative sample sizes, normalize to percentiles
            cumulative_percentiles = torch.cumsum(counts_for_each_corpus, dim=0).float() / total_samples
            # Exclude 1.0, this threshold we don't need
            self.transition_thresholds = cumulative_percentiles[:-1].tolist()
            data_cl_logger.info(f"Calculated proportion-based thresholds: {self.transition_thresholds}")
        elif self.proportion_mode == "token":
            number_of_tokens_for_each_corpus = self._get_corpora_token_sizes()
            total_number_of_tokens = number_of_tokens_for_each_corpus.sum()
            cumulative_percentiles = torch.cumsum(number_of_tokens_for_each_corpus, dim=0).float() / total_number_of_tokens
            self.transition_thresholds = cumulative_percentiles[:-1].tolist()
            data_cl_logger.info(f"Calculated token-proportion thresholds: {self.transition_thresholds}")

    def _get_corpora_sample_sizes_on_complete_dataset(self, dataset: Dataset):
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

    def _get_corpora_token_sizes(self) -> torch.Tensor:
        """
        Aggregates token counts per stage using the static mapping in stages.py
        """
        stage_counts = torch.zeros(NUM_STAGES, dtype=torch.float32)
        for filename, stage in self.filename_to_difficulty_map.items():
            token_count = DATASET_TOKEN_COUNTS.get(filename, 0)
            # stages are 1-indexed, tensor is 0-indexed
            stage_counts[stage - 1] += token_count
        return stage_counts

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
        # Use data replay (considering only the previous stage), when enabled and we're
        # at least in stage 2, as there is otherwise no previous stage
        if self.data_replay_mode == "previous_stage_only" and active_difficulty_level > 1:

            number_of_tokens_current_stage, number_of_tokens_previous_stage = self._get_token_sizes_for_current_and_previous_stage(active_difficulty_level=active_difficulty_level)
            weight_mapping = self._get_weight_mapping_for_current_and_previous_stage(number_of_tokens_current_stage=number_of_tokens_current_stage,
                                                                                     number_of_tokens_previous_stage=number_of_tokens_previous_stage,
                                                                                     active_difficulty_level=active_difficulty_level)

            # Map the stage IDs in our dataset to the calculated weights
            mask = weight_mapping[self._difficulty_scores_tensor.long()]

            cp_logger.info(
                f"Replay Active: Stage {active_difficulty_level} (w={weight_mapping[active_difficulty_level]:.4f}), "
                f"Replaying Stage {active_difficulty_level - 1} (w={weight_mapping[active_difficulty_level - 1]:.4f})"
            )

        elif self.data_replay_mode == "all_previous_stages" and active_difficulty_level > 1:
            weight_mapping = self._get_weight_mapping_for_all_previous_weighted(active_difficulty_level)
            mask = weight_mapping[self._difficulty_scores_tensor.long()]
            cp_logger.info(f"Weighted Decay Replay Active for Stage {active_difficulty_level}")

        # Case 2: Default / Strict Staging (or Stage 1 where no replay is possible)
        else:
            # Strict staging: samples in current stage = 1.0, others = 0.0
            mask = (self._difficulty_scores_tensor == active_difficulty_level).float()
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

    def _get_token_sizes_for_current_and_previous_stage(self, active_difficulty_level: int):
        token_counts = self._get_corpora_token_sizes()

        # Indices for token_counts tensor (0-indexed)
        curr_idx = active_difficulty_level - 1
        prev_idx = active_difficulty_level - 2

        number_of_tokens_current_stage = token_counts[curr_idx]
        number_of_tokens_previous_stage = token_counts[prev_idx]

        return number_of_tokens_current_stage, number_of_tokens_previous_stage

    def _get_weight_mapping_for_current_and_previous_stage(self, number_of_tokens_current_stage: int, number_of_tokens_previous_stage: int, active_difficulty_level:int):
        # Calculate weight for current stage, considering data replay
        weight_current_stage = 1.0 - self.data_replay_fraction
        # Calculate weight for previous stage while taking into account the number of tokens,
        # which balances the sampling probability relative to the corpus size
        weight_previous_stage = self.data_replay_fraction * (number_of_tokens_current_stage / number_of_tokens_previous_stage)

        # Create a mapping for all possible stages (1 to NUM_STAGES)
        weight_mapping = torch.zeros(NUM_STAGES + 1, dtype=torch.float32)
        weight_mapping[active_difficulty_level] = weight_current_stage
        weight_mapping[active_difficulty_level - 1] = weight_previous_stage

        return weight_mapping

    def _get_weight_mapping_for_all_previous_weighted(self, active_difficulty_level):


        number_tokens_all_stages = self._get_corpora_token_sizes()
        number_tokens_current_stage = number_tokens_all_stages[active_difficulty_level - 1]


        # Get all previous stages
        all_previous_stages = list(range(1, active_difficulty_level))

        # Determine decay factors, so that earlier stages may be less likely in current stage
        normalized_decay_factors = self._calculate_decay_factors_for_all_previous_stages(active_difficulty_level=active_difficulty_level, previous_stages=all_previous_stages)

        # Initialize weight mapping tensor
        weight_mapping = torch.zeros(NUM_STAGES + 1, dtype=torch.float32)

        # Set current stage weight
        weight_mapping[active_difficulty_level] = 1.0 - self.data_replay_fraction

        # Include weights for all previous stages based on number of tokens & the weight decay factor
        weight_mapping = self._get_weight_mapping_for_current_and_all_previous_stages(prev_stages=all_previous_stages, number_tokens_all_stages=number_tokens_all_stages, decay_factors=normalized_decay_factors, weight_mapping=weight_mapping, number_tokens_current_stage=number_tokens_current_stage)

        return weight_mapping

    def _calculate_decay_factors_for_all_previous_stages(self, active_difficulty_level: int, previous_stages: list[int]):
        # Calculate unnormalized decay factors for all previous stages, where stage (n-1) gets decay^0,
        # stage (N-2) gets decay^1, ...
        decay_factors = torch.tensor([
            self.data_replay_decay ** (active_difficulty_level - 1 - i)
            for i in previous_stages
        ], dtype=torch.float32)

        # Normalize decay factors so they sum to 1.0, representing total replay budget
        normalized_decay_factors = decay_factors / torch.sum(decay_factors)
        return normalized_decay_factors

    def _get_weight_mapping_for_current_and_all_previous_stages(self, prev_stages: list, number_tokens_all_stages: Tensor, decay_factors: list, weight_mapping, number_tokens_current_stage: Tensor):
        # Calculate weights for each previous stage
        for stage_number in prev_stages:
            number_tokens_stage_i = number_tokens_all_stages[stage_number - 1]
            # Consider token ratio between current stage and the i-th previous stage in order to ensure correct token-wise representation of the corpora
            token_ratio = number_tokens_current_stage / number_tokens_stage_i
            # Calculate the weight for i-th stage
            weight_mapping[stage_number] = self.data_replay_fraction * token_ratio * decay_factors[stage_number - 1]
        return weight_mapping
