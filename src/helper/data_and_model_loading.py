import logging
import os
import torch
import random
import numpy as np

from datasets import Dataset, DatasetDict, load_dataset
from collections import Counter

from src.config import BabyLMConfig
from src.tokenizer import load_tokenizer
from src.models import load_base_model
from src.helper.dataset_preprocessor import DatasetPreprocessor

DRY_RUN_SUBSAMPLE_FACTOR = 1000 // (10 if torch.cuda.device_count() > 1 else 1)

# Logger for this file
logger = logging.getLogger(__name__)

def load_dataset_model_and_tokenizer(cfg: BabyLMConfig):
    # Loading dataset from huggingface hub
    logger.info("Loading dataset")
    dataset: DatasetDict = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        token=os.environ["HF_READ_TOKEN"],
    )
    # Check that the format is correct
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    # Loads tokenizer from huggingface
    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg)

    # Loads initialized model from huggingface
    logger.info("Initializing model")
    model = load_base_model(cfg)

    # Check tokenizer & vocab size validity
    assert (
            tokenizer.vocab_size == model.config.vocab_size
    ), "Tokenizer and model vocab size mismatch"

    train_dataset, eval_dataset = preprocess_data(cfg=cfg, tokenizer=tokenizer, dataset=dataset)

    return model, tokenizer, train_dataset, eval_dataset

def preprocess_data(cfg: BabyLMConfig, tokenizer, dataset):
    # Preprocess the data
    logger.info("Preprocessing data")
    data_preprocessor = DatasetPreprocessor(cfg, tokenizer)

    # Preprocess every sample of the training data
    train_dataset = dataset["train"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["train"].column_names,
    )

    # Preprocess every sample of the evaluation data
    eval_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
    )

    # Subsample dataset in dry run by desired factor & log dataset size & corpus distributions
    if cfg.experiment.dry_run:
        logger.info(
            f"Running in dry run mode -- stratified subsampling train and eval datasets by {DRY_RUN_SUBSAMPLE_FACTOR}x"
        )
        train_dataset = stratified_subsample_by_corpus(train_dataset, subsample_factor=DRY_RUN_SUBSAMPLE_FACTOR, corpora_column_name="filename")
        eval_dataset = stratified_subsample_by_corpus(eval_dataset, subsample_factor=DRY_RUN_SUBSAMPLE_FACTOR, corpora_column_name="filename")

        log_corpus_distribution(train_dataset, name="train_dataset (after subsampling)")
        log_corpus_distribution(eval_dataset, name="eval_dataset (after subsampling)")

    # If not in dry run, just log dataset size & corpus distributions
    else:
        log_corpus_distribution(train_dataset, name="train_dataset")
        log_corpus_distribution(eval_dataset, name="eval_dataset")

    return train_dataset, eval_dataset


def stratified_subsample_by_corpus(dataset: Dataset, subsample_factor: int, corpora_column_name: str = "filename") -> Dataset:
    """
    Subsample a HuggingFace Dataset while preserving corpus distribution.

    :param dataset: HuggingFace Dataset to subsample
    :param subsample_factor: Integer, keep ~1/subsample_factor of each corpus (minimum 1 sample per corpus)
    :param corpora_column_name: Column name holding the corpus identifier (e.g., "filename")
    :return: A new Dataset containing the subsampled rows
    """
    corpora_column = np.array(dataset[corpora_column_name])
    unique_corpora = np.unique(corpora_column)

    # Keep track of indices to remain in subsampled dataset
    indices_to_keep = []

    # Loop over each corpus and determine indices to keep
    for corpus_name in unique_corpora:
        # Find row indices for this corpus
        corpus_indices = np.nonzero(corpora_column == corpus_name)[0]

        # Determine number of samples to keep for a given corpus, at least one
        number_of_samples_to_keep_for_corpus = max(1, len(corpus_indices) // subsample_factor)

        # Randomly sample indices to keep for given corpus
        sampled_indices = random.sample(list(corpus_indices), number_of_samples_to_keep_for_corpus)

        # Keep track of these indices
        indices_to_keep.extend(sampled_indices)

    # Create a new dataset containing only the sampled rows based on tracked indices
    subsampled_dataset = dataset.select(sorted(indices_to_keep))

    return subsampled_dataset

def log_corpus_distribution(dataset, name="dataset", corpus_col_name="filename") -> None:
    """
    Log the distribution of samples across different corpora in a HuggingFace Dataset

    :param dataset: HuggingFace Dataset whose corpus distribution should be logged
    :param name: Label for the dataset used in logging output, e.g. "train_dataset"
    :param corpus_col_name: Column name containing the corpus identifier
    """
    # Count how many samples belong to each corpus
    counts_per_corpus = Counter(dataset[corpus_col_name])
    # Total number of samples in the dataset
    total = sum(counts_per_corpus.values())
    # Log the total sample count
    logger.info(f"{name} total samples: {total}")
    # Log the sample count and relative proportion for each corpus
    for corpus, count in counts_per_corpus.items():
        logger.info(f"  {corpus}: {count} samples ({count/total:.2%})")