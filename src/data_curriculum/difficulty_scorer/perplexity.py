from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generator, List, Sequence, Tuple, Union

import torch

# typing imports
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    # avoid circular imports
    from src.custom_trainer import CustomTrainer

# NLTK Package provides functionality for n-gram models
from nltk.lm import MLE
from nltk.util import everygrams
from torch.utils.data import DataLoader
from tqdm import tqdm

# Data processing utils
from src.helper.dataset_preprocessor import SequentialSubsetSampler, base_collate_fn

# Perplexity Computation
from src.helper.inference import (
    compute_trainer_perplexity,
    prepare_dataset_for_ppl_inference,
)

from .base_difficulty_scorer import BaseDifficultyScorer
from .registry import register_difficulty_scorer

# Logger for this module
data_cl_logger = logging.getLogger("Data Curriculum")


class PerplexityBaseClass(BaseDifficultyScorer):
    """A base class encapsulating shared logic between SelfPerplexityScorer and NGramPerplexityScorer"""

    @property
    def tokenizer(self) -> Union[PreTrainedTokenizerFast, None]:
        # Getter for the tokenizer used to convert text <-> token IDs
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerFast):
        # Setter for tokenizer; must be set before model training
        self._tokenizer = tokenizer


@register_difficulty_scorer("ngram_perplexity")
class NGramPerplexityScorer(PerplexityBaseClass):
    """Difficulty scorer using a classical n-gram language model to compute sample perplexity."""
    def __init__(self, n_gram: int, train_subsample_factor: int = 1, **kwargs):
        """
        Initializes the n-gram perplexity scorer.

        Args:
            * n_gram (int): Maximum n-gram size (e.g., 3 = trigram model)
            * train_subsample_factor (int): The factor by which to subsample the dataset for
                training the n-gram model. For example, if dataset_subsample_factor = 2, then
                the n-gram model will be trained on half the dataset.
        """

        super().__init__(**kwargs)

        data_cl_logger.info("Initializing n-gram perplexity scorer")

        self.n_gram = n_gram

        # Tokenizer must be set externally before training
        self._tokenizer = None

        self.train_subsample_factor = train_subsample_factor

    def _train_model(
        self,
        dataset: Dataset,
    ):
        """
        Trains a n-gram model on the dataset.

        Args:
            * dataset (Dataset): The dataset to train the model on
        Returns:
            * None
        """

        assert self.tokenizer is not None and hasattr(
            self.tokenizer, "pad_token_id"
        ), "The tokenizer must be set before training the n-gram model"

        data_cl_logger.info("Training n-gram model")

        # TODO: Refactor, this method might be defined outside to foster readability
        def remove_padding_tokens(example_batch):
            """Remove pad tokens from tokenized inputs for accurate probability estimation."""
            batch = {"unpadded_input_ids": []}
            for example in example_batch["input_ids"]:
                batch["unpadded_input_ids"].append(
                    [
                        str(_id)
                        for _id in example
                        if _id != self.tokenizer.pad_token_id  # type: ignore
                    ]
                )

            return batch

        # Remove padding tokens from the dataset in parallel using 64 processes
        dataset = dataset.map(
            remove_padding_tokens,
            batched=True,
            num_proc=64,
        )

        # Save tokenized and unpadded data for later scoring
        self.tokenized_text = dataset["unpadded_input_ids"]

        # We split the tokenized dataset into n-gram that we can feed into the n-gram model
        train_data_n_grams = (
            everygrams(sent, max_len=self.n_gram)
            for sent in self.tokenized_text[
                0 : dataset.num_rows : self.train_subsample_factor
            ]
        )

        # Build vocabulary as string tokens for NLTK model
        train_vocab = [str(val) for val in self.tokenizer.vocab.values()]

        # Train a Maximum Likelihood Estimation n-gram model
        self.lm = MLE(self.n_gram)
        self.lm.fit(train_data_n_grams, train_vocab)

    def _compute_ngram_perplexity(
        self, example: Generator[Tuple[str], Any, None]
    ):
        """Compute perplexity for a single sequence of n-grams using the trained LM"""
        return self.lm.perplexity(example)

    def score_difficulty(
        self,
        dataset: Dataset,
        indices: List[int],
        global_stepnum: int,
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        """
        Scores the difficulty of the dataset, and returns a list of filtered difficulty scores. It:
        - Trains an n-gram model on the dataset
        - Computes perplexity for the subset of samples indicated by `indices`
        - Filters scores above the given percentile

        Args:
            * dataset (Dataset): The dataset to score
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

        # Only train and compute at step 0; assumes static scoring thereafter
        if global_stepnum == 0:

            self._train_model(dataset)

            assert hasattr(self, "lm") and hasattr(
                self, "tokenized_text"
            ), "n-gram model not trained"

            self._difficulty_scores: Sequence[float] = []

            # Convert tokenized sentences into n-grams for scoring
            data_n_grams = (
                everygrams(sent, max_len=self.n_gram)
                for sent in self.tokenized_text
            )

            # indices is a list of indices that we want to score the difficulty of
            # (if we are using distributed training, not all indices will be scored - only those
            # assigned to the current process)
            curr_indices_idx = 0

            data_cl_logger.info("Evaluating perplexity [NGram Model]")

            for _idx, n_gram in enumerate(data_n_grams):
                # For distributed training, only compute scores for our assigned `indices
                if _idx == indices[curr_indices_idx]:
                    # Compute perplexity for the current sample
                    perplexity = self._compute_ngram_perplexity(n_gram)
                    self._difficulty_scores.append(perplexity)

                    curr_indices_idx += 1

                    if curr_indices_idx == len(indices):
                        break
            else:
                raise RuntimeError("Not all indices were scored")

        assert hasattr(
            self, "_difficulty_scores"
        ), "Difficulty scores have not been computed but about to filter them."

        # Filter out scores above the max difficulty percentile
        self.filtered_difficulty_scores = (
            self.remove_scores_above_max_difficulty(
                self._difficulty_scores, max_difficulty_percentile
            )
        )

        return self.filtered_difficulty_scores


@register_difficulty_scorer("self_perplexity")
class SelfPerplexityScorer(PerplexityBaseClass):
    """
    Difficulty scorer that:
        - Optionally initializes with an n-gram model
        - Periodically updates difficulty using the model’s own perplexity
        - Can adapt to the model as it trains (active learning / curriculum learning)
    """
    def __init__(self, n_gram: int, update: int, **kwargs):
        """
        Initializes the self-perplexity scorer

        Args:
            * n_gram (int): The n-gram to use for the initial n-gram model; setting to 0 to
                randomly sample from the dataset
            * update (int): The number of steps to wait before updating the n-gram model
        """

        super().__init__(**kwargs)

        data_cl_logger.info(
            "Initializing active learning perplexity difficulty scorer"
        )

        if n_gram <= 0:
            self.ngram_model = None
        else:
            self.ngram_model = NGramPerplexityScorer(n_gram)

        self.update = update
        self._trainer = None
        self._tokenizer = None

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["_trainer"]
        return state

    @property
    def trainer(self) -> Union[CustomTrainer, None]:
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: CustomTrainer):
        self._trainer = trainer

    def score_difficulty(
        self,
        dataset: Dataset,
        indices: List[int],
        global_stepnum: int,
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        """
        Scores the difficulty of the dataset, and returns a list of filtered difficulty scores. It either:
        @Step 0:
            - Use n-gram model if available; otherwise uniform scoring
        @Step > 0 and divisible by `update`:
            - Use the trainer model to compute batch-wise perplexity

        Args:
            * dataset (Dataset): The dataset to score
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

        assert self.tokenizer is not None and self.trainer is not None

        # if this is the initial step, use the ngram model class's method
        if global_stepnum == 0:

            if self.ngram_model is None:
                data_cl_logger.info(
                    "No NGram Model specified; sampling uniformly on global step 0"
                )
                self._difficulty_scores = [1.0 for _ in indices]
            else:
                # Delegate to the n-gram model
                self.ngram_model.tokenizer = self.tokenizer

                # NOTE: the n_gram model will return a list of perplexity scores that have already
                # been filtered by the max_difficulty_percentile; however, it will contain
                # the _difficulty_scores attribute
                _ = self.ngram_model.score_difficulty(
                    dataset, indices, global_stepnum, max_difficulty_percentile
                )

                self._difficulty_scores = self.ngram_model._difficulty_scores

        else:
            # Recompute difficulty periodically based on the model's own predictions
            if global_stepnum % self.update == 0:

                # NOTE: remove keys from dataset that are not in the signature of the model,
                # since we pass the data through the model

                # Prepare dataset for inference by stripping unused keys
                dataset = prepare_dataset_for_ppl_inference(
                    self.trainer, dataset
                )

                data_cl_logger.info(
                    f"Recalculating sample weights using model at step {global_stepnum}"
                )
                self._difficulty_scores: Sequence[float] = []

                with torch.no_grad():
                    data_cl_logger.info(
                        "Evaluating perplexity [Trainer Model]"
                    )

                    # NOTE If we are using distributed training, not all indices will be scored
                    # only those assigned to the current process
                    # Use a sampler to only iterate over assigned subset of indices
                    sampler = SequentialSubsetSampler(indices)

                    inference_dataloader = DataLoader(
                        dataset,  # type: ignore
                        batch_size=4,
                        shuffle=False,
                        collate_fn=base_collate_fn,
                        sampler=sampler,
                        pin_memory=True,
                    )

                    # Compute perplexity batch-wise and accumulate
                    for batch in tqdm(inference_dataloader):
                        batch_perplexity = compute_trainer_perplexity(
                            batch, self.tokenizer, self.trainer
                        )

                        self._difficulty_scores.extend(batch_perplexity)

        # Validate that scores exist
        assert hasattr(
            self, "_difficulty_scores"
        ), "Difficulty scores have not been computed but about to filter them."

        # Apply percentile filtering to discard overly difficult samples
        self.filtered_difficulty_scores = (
            self.remove_scores_above_max_difficulty(
                self._difficulty_scores, max_difficulty_percentile
            )
        )

        return self.filtered_difficulty_scores
