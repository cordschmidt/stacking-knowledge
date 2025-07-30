""" Custom Dataloading comptaible with Curriculum Learning """
# TODO: Check if this whole thing can be removed, as only now the ignore columns thing is custom logic

import logging

# typing imports
from typing import Dict, List, Optional

# PyTorch imports for tensor operations and dataloading
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory as _torch_pin_memory
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe

# HuggingFace tokenizer for text preprocessing
from transformers import PreTrainedTokenizerFast

# Local imports
from src.helper.data import base_collate_fn

# Standard logging setup for debugging
logger = logging.getLogger(__name__)

class CurriculumDataLoader(DataLoader):
    def __init__(
        self,
        global_stepnum: int,
        ignore_columns: Optional[List[str]] = None,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """
        Custom DataLoader. The data driven aspect is encapsulated in the sampler, which is passed to the DataLoader.

        Args:
            * global_stepnum (int): The current training step number. Used to determine
                the active curriculum
            * ignore_columns (Optional[List[str]], optional): A list of columns to ignore.
                Defaults to None.
            * num_workers (int, optional): The number of workers to use. Defaults to 0.
        """
        self.global_stepnum = global_stepnum
        self.ignore_columns = ignore_columns

        # TODO: I don't get that. Why is this done? What is meant by multi-process? Why is it not possible?
        #  Does this mean no multi-GPU support is available?
        if num_workers != 0:
            # NOTE: Multi-process dataloading is not implemented for this custom iterator.
            # The default Trainer typically uses 0 workers and is already performant.
            logger.warning(
                "Multi-process dataloading is not supported yet - using 0 workers."
            )

        # Initialize the parent DataLoader with single-process setup
        super().__init__(num_workers=0, **kwargs)

    def __iter__(self):
        """
        Override the default iterator to return our custom single-process iterator
        that supports curriculum-aware collators and vocabulary mapping.
        """
        return _CustomSingleProcessDataLoaderIter(self)

class _CustomSingleProcessDataLoaderIter(_BaseDataLoaderIter):
    """
    Custom iterator for the CurriculumDataLoader

    Responsibilities:
    - Fetch data samples sequentially (single-process only)
    - Remove ignored columns if specified
    """
    def __init__(self, loader: CurriculumDataLoader):
        super().__init__(loader)
        # Ensure single-process mode
        assert self._timeout == 0
        assert self._num_workers == 0

        self.loader = loader

        # Torch DataPipes are currently not supported by this implementation
        # TODO: What are those?
        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            raise NotImplementedError(
                "IterDataPipe and MapDataPipe are not supported yet"
            )

        # Use the static collator function in the dataset fetcher
        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            base_collate_fn,
            self._drop_last,
        )

    def _next_index(self):
        """
        Retrieve the next sample index from the sampler iterator
        """
        idx = next(self._sampler_iter)
        return idx

    def _next_data(self):
        """
        Retrieve the next batch of data
        """

        # Get the next data index (drives sampling)
        index = self._next_index()  # may raise StopIteration
        # Fetch the actual data batch
        data = self._dataset_fetcher.fetch(index)

        # Optionally pin memory for faster GPU transfers
        if self._pin_memory:
            data = _torch_pin_memory(data, self._pin_memory_device)  # type: ignore[arg-type]

        # Drop any ignored columns from the batch
        if self.loader.ignore_columns is not None:
            for ignore_column in self.loader.ignore_columns:
                data.pop(ignore_column, None)

        return data
