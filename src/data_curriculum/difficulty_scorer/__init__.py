""" This module uses the difficulty scorer registry to get a difficulty scorer

The purpose of this module is to provide a unified function `get_difficulty_scorer`
that, given the name of a difficulty scorer, will return an instance of the corresponding
difficulty scorer class.

It also defines `Protocol` classes to indicate that certain scorers may require
access to a tokenizer or a trainer

"""

from transformers import PreTrainedTokenizerFast, Trainer
from typing_extensions import Protocol, runtime_checkable

# typing imports
from src.config import DifficultyScorerKwargsType

# Base class for all difficulty scorers
from .base_difficulty_scorer import BaseDifficultyScorer

# Data Split Scorers
from .data_split import DataSplitSorter
from .staged_data_split import StagedDataSplitSorter


# importing for registry to register difficulty scorers
from .perplexity import NGramPerplexityScorer
from .registry import DIFFICULTY_SCORER_REGISTRY


@runtime_checkable
class UsesTokenizer(Protocol):
    """
    Protocol (interface) for scorers that require a tokenizer.

    A class that implements this protocol must have a `tokenizer` attribute
    of type `PreTrainedTokenizerFast`.

    The @runtime_checkable decorator allows us to use `isinstance(obj, UsesTokenizer)`
    at runtime to check if an object follows this protocol
    """
    tokenizer: PreTrainedTokenizerFast


@runtime_checkable
class UsesTrainer(Protocol):
    """
    Protocol (interface) for scorers that require access to a HuggingFace Trainer.

    A class implementing this must have a `trainer` attribute of type `Trainer`.
    This is used by active-learning scorers that evaluate model performance
    during training to compute difficulty.
    """
    trainer: Trainer


def get_difficulty_scorer(
    difficulty_scorer_name: str,
    difficulty_scorer_kwargs: DifficultyScorerKwargsType,
    trainer: Trainer,
) -> BaseDifficultyScorer:
    """
    Returns an initialized difficulty scorer based on the provided name. This function:
    1. Looks up the scorer class from the global registry `DIFFICULTY_SCORER_REGISTRY`.
    2. Instantiates it with the provided kwargs.
    3. Optionally injects the trainer and/or tokenizer if the scorer needs them.

    Args:
        * difficulty_scorer_name (str): The name of the difficulty scorer
        * difficulty_scorer_kwargs (DifficultyScorerKwargsType): The kwargs for the difficulty
            scorer
        * trainer (Trainer): The trainer object, some of the difficulty scorers need access to
            certain attributes of the trainer or even the entire trainer object iself if we are
            using an active-learning difficulty scorer.
    Returns:
        * BaseDifficultyScorer: An initialized difficulty scorer ready to compute difficulty scores
    """

    # Check if the scorer name exists in the global registry of difficulty scorers
    # This registry is populated by the `@register_difficulty_scorer` decorator
    if difficulty_scorer_name in DIFFICULTY_SCORER_REGISTRY:
        difficulty_scorer = DIFFICULTY_SCORER_REGISTRY[difficulty_scorer_name](
            **difficulty_scorer_kwargs,  # type: ignore
        )

        # If the difficulty scorer needs access to the trainer or the tokenizer, we pass it in
        # NOTE: The trainer is needed if the difficulty scorer uses the trainer itself to score
        # the difficulty of the dataset

        if isinstance(difficulty_scorer, UsesTrainer):
            difficulty_scorer.trainer = trainer

        if isinstance(difficulty_scorer, UsesTokenizer):
            # NOTE: This assert statement should never fail, since we run a similar check on the
            # tokenizer before initializing the trainer. It is needed, however, to narrow the type
            # to pass type checking
            assert isinstance(trainer.tokenizer, PreTrainedTokenizerFast)
            difficulty_scorer.tokenizer = trainer.tokenizer

        # Return the fully-initialized scorer
        return difficulty_scorer

    else:
        # If the scorer name is not registered, raise an error
        raise ValueError(
            f"Difficulty Scorer {difficulty_scorer_name} not supported."
        )
