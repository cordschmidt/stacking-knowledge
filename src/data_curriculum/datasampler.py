import torch
import copy

from typing import Union, Sequence, Callable, Iterator
from typing_extensions import Protocol
from torch import Generator
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

# Local imports
from .difficulty_scorer import BaseDifficultyScorer


class CurriculumIterTypeProtocol(Protocol):
    """Defines the attributes that a curriculum-based sampler must implement."""

    @property
    def generator(self) -> Union[Generator, None]:
        ...

    @property
    def global_stepnum(self) -> int:
        ...

    @property
    def batch_size(self) -> int:
        ...

    @property
    def indices(self) -> Sequence[int]:
        ...

    @property
    def pacing_fn(self) -> Callable[..., float]:
        ...

    @property
    def difficulty_scorer(self) -> BaseDifficultyScorer:
        ...

    @property
    def dataset(self) -> Dataset:
        ...

class CurriculumIterMixin:
    """
    Mixin class for functionality that is shared between the CurriculumSampler and the
    DistributedCurriculumSampler. Shared logic for sampling batches based on curriculum learning.
    """

    def _curriculum_iter(self: CurriculumIterTypeProtocol):
        """
        Returns an iterator for data-driven curriculum learning that continuously generates
        samples of indices. Each batch of indices is aware of the current global_stepnum and
        will re-compute the current upper-limit index that can be sampled from.
        """
        while True:
            # Use pacing function to determine how much of the dataset to expose (via percentile)
            max_difficulty_percentile: float = self.pacing_fn(self.global_stepnum)

            # Score the difficulty of the dataset items up to the given indices
            difficulty_scores = self.difficulty_scorer.score_difficulty(
                self.dataset,
                self.indices,  # The index list for the current replica or whole set
                self.global_stepnum,
                max_difficulty_percentile,
            )

            # Convert difficulty scores to tensor for multinomial sampling
            difficulty_scores_tensor = torch.tensor(difficulty_scores)

            # Sample a batch of indices biased by difficulty (via multinomial)
            for i in torch.multinomial(
                    difficulty_scores_tensor, self.batch_size, replacement=False
            ):
                yield self.indices[i]

class CurriculumSampler(CurriculumIterMixin, Sampler):
    """
    A custom sampler for curriculum learning on a single GPU/device.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scorer: BaseDifficultyScorer,
        pacing_fn: Callable[[int], float],
        batch_size: int,
        generator: Union[Generator, None] = None,
        global_stepnum: int = 0,
    ) -> None:
        """
        Args:
            * dataset: the dataset to sample from
            * difficulty_scorer: the difficulty scorer to use for curriculum learning; scores
                the difficulty of the dataset and returns a list of scores
            * pacing_fn: a function that takes in the global stepnum and returns the upper limit
                of the index that we can sample to from the dataset
            * batch_size: the batch size
            * generator: a torch.Generator object
            * global_stepnum: the global stepnum of the training loop
        """

        # Deep copy the dataset to avoid mutation side effects
        self.dataset = copy.deepcopy(dataset)

        # All indices in the dataset
        self.indices: Sequence[int] = list(range(len(dataset)))  # type: ignore[arg-type]

        # Set curriculum-related components
        self.difficulty_scorer = difficulty_scorer
        self.pacing_fn = pacing_fn
        self.batch_size = batch_size
        self.generator = generator
        self.global_stepnum = global_stepnum

    def __iter__(self) -> Iterator[int]:
        # Delegate sampling logic to shared curriculum iterator
        yield from self._curriculum_iter()

    def __len__(self):
        # NOTE: CurriculumSampler does not have a concept of epoch 'length'
        return None


class DistributedCurriculumSampler(CurriculumIterMixin, DistributedSampler):
    """
    A DDP-compatible sampler for curriculum learning that partitions data by GPU rank.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scorer: BaseDifficultyScorer,
        pacing_fn: Callable[[int], float],
        batch_size: int,
        generator: Union[Generator, None] = None,
        global_stepnum: int = 0,
        **kwargs,
    ) -> None:
        """
        Args:
            * dataset: the dataset to sample from
            * difficulty_scorer: the difficulty scorer to use for curriculum learning; scores
                the difficulty of the dataset and returns a list of scores
            * pacing_fn: a function that takes in the global stepnum and returns the upper limit
                of the index that we can sample to from the dataset
            * batch_size: the batch size
            * generator: a torch.Generator object
            * global_stepnum: the global stepnum of the training loop
            * kwargs: kwargs for DistributedSampler (num_replicas, rank, drop_last)
        """
        # NOTE: Shuffle needs to be False otherwise there's no point to applying a curriculum
        kwargs["drop_last"] = True
        kwargs["shuffle"] = False
        # Call DistributedSampler constructor with modified kwargs
        super().__init__(copy.deepcopy(dataset), **kwargs)

        # Set curriculum-specific state
        self.difficulty_scorer = difficulty_scorer
        self.pacing_fn = pacing_fn
        self.batch_size = batch_size
        self.generator = generator
        self.global_stepnum = global_stepnum

        # List of all indices in the full dataset, not just the dataset available on a single GPU / device
        _indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # Truncate to total size
        # Total number of examples after rounding up to make the data evenly divisible across all GPUs
        _indices = _indices[: self.total_size]
        assert len(_indices) == self.total_size

        # Slice the index list to only include the shard for this process, each
        # GPU gets its own distinct slice of the data
        self.indices = _indices[
            self.rank : self.total_size : self.num_replicas
        ]
        assert len(self.indices) == self.num_samples

    def __iter__(self) -> Iterator[int]:
        # Use curriculum iterator as in single-device sampler
        yield from self._curriculum_iter()

    def __len__(self):
        # NOTE: CurriculumSampler does not have a concept of epoch 'length'
        return None
