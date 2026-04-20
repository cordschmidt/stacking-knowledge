import json
import logging
import os
import torch
import random
import numpy as np

from datasets import Dataset, DatasetDict, load_dataset
from collections import Counter
from transformers.modeling_utils import unwrap_model
from safetensors.torch import load_file

from src.config import BabyLMConfig
from src.tokenizer import load_tokenizer
from src.models import load_base_model
from src.helper.dataset_preprocessor import DatasetPreprocessor

DRY_RUN_SUBSAMPLE_FACTOR = 1000 // (10 if torch.cuda.device_count() > 1 else 1)

# Logger for this file
logger = logging.getLogger(__name__)

def load_dataset_model_and_tokenizer(cfg: BabyLMConfig):
    """
    Loads the HuggingFace dataset, tokenizer, and initializes the base model
    according to the provided configuration

    Args:
        cfg: The BabyLM configuration object containing dataset and model settings

    Returns:
        A tuple containing (model, tokenizer, dataset)
    """
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


    print_model_stats(model=model)

    # Check tokenizer & vocab size validity
    assert (
            tokenizer.vocab_size == model.config.vocab_size
    ), "Tokenizer and model vocab size mismatch"

    return model, tokenizer, dataset

def preprocess_data(cfg: BabyLMConfig, tokenizer, dataset):
    """
    Preprocesses the raw dataset splits using the configured tokenizer and
    handles dry-run subsampling if enabled in the configuration

    Args:
        cfg: The BabyLM configuration object
        tokenizer: The initialized HuggingFace tokenizer
        dataset: The raw HuggingFace DatasetDict containing train, validation and test splits

    Returns:
        A tuple of preprocessed datasets: (train_dataset, eval_dataset, dev_dataset)
    """
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

    dev_dataset = dataset["test"].map(
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
        train_dataset = stratified_subsample_by_corpus(train_dataset, subsample_factor=DRY_RUN_SUBSAMPLE_FACTOR, corpora_column_name="filename", seed=cfg.experiment.seed)
        eval_dataset = stratified_subsample_by_corpus(eval_dataset, subsample_factor=DRY_RUN_SUBSAMPLE_FACTOR, corpora_column_name="filename", seed=cfg.experiment.seed)
        dev_dataset = stratified_subsample_by_corpus(dev_dataset, subsample_factor=DRY_RUN_SUBSAMPLE_FACTOR,
                                                      corpora_column_name="filename", seed=cfg.experiment.seed)

        log_corpus_distribution(train_dataset, name="train_dataset (after subsampling)")
        log_corpus_distribution(eval_dataset, name="eval_dataset (after subsampling)")
        log_corpus_distribution(dev_dataset, name="dev_dataset (after subsampling)")

    # If not in dry run, just log dataset size & corpus distributions
    else:
        log_corpus_distribution(train_dataset, name="train_dataset")
        log_corpus_distribution(eval_dataset, name="eval_dataset")
        log_corpus_distribution(dev_dataset, name="eval_dataset")

    return train_dataset, eval_dataset, dev_dataset


def stratified_subsample_by_corpus(dataset: Dataset, subsample_factor: int, corpora_column_name: str = "filename", seed: int = 42) -> Dataset:
    """
    Subsamples a HuggingFace Dataset while preserving the proportional
    distribution of samples across different corpora

    Args:
        dataset: The HuggingFace Dataset to subsample
        subsample_factor: The integer factor by which to reduce the dataset (e.g., 10 means keeping 1/10th)
        corpora_column_name: The column name identifying the corpus for each sample
        seed: The random seed for reproducible sampling

    Returns:
        A new, subsampled HuggingFace Dataset
    """

    # Use a local RNG instance to avoid global state interference
    rng = random.Random(seed)
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
        sampled_indices = rng.sample(list(corpus_indices), number_of_samples_to_keep_for_corpus)

        # Keep track of these indices
        indices_to_keep.extend(sampled_indices)

    # Create a new dataset containing only the sampled rows based on tracked indices
    subsampled_dataset = dataset.select(sorted(indices_to_keep))

    return subsampled_dataset

def log_corpus_distribution(dataset, name="dataset", corpus_col_name="filename") -> None:
    """
    Calculates and logs the distribution of samples across different corpora
    within a given dataset split

    Args:
        dataset: The HuggingFace Dataset to analyze
        name: A string label for the dataset used in the log output (e.g., "train_dataset")
        corpus_col_name: The column name identifying the corpus for each sample

    Returns:
        None
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


def print_model_stats(model, name="Model"):
    """
    Calculates and logs detailed parameter statistics for the provided model,
    separating embedding, non-embedding, and vocabulary parameters

    Args:
        model: The initialized neural network model
        name: A string label for the model used in the log output

    Returns:
        None
    """
    num_layers = len(model.model.layers)
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate embedding parameters based on the Llama model structure
    num_embed_params = sum(p.numel() for p in model.model.embed_tokens.parameters())
    num_non_embed_params = total_params - num_embed_params
    num_head_params = sum(p.numel() for p in model.lm_head.parameters())
    num_non_vocab_params = total_params - num_embed_params - num_head_params

    logger.info(f"\n{name} size:")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Non-embedding parameters: {num_non_embed_params:,}\n")
    logger.info(f"  Non-vocabulary parameters: {num_non_vocab_params:,}\n")


def truncate_model_if_best_checkpoint_size_differs(trainer):
    """
    Checks if the current model size differs from the best checkpoint's model size
    (due to gradual stacking) and truncates the current model to match it to prevent evaluation errors

    Args:
        trainer: The HuggingFace Trainer instance containing the state and model

    Returns:
        None
    """
    if trainer.state.best_model_checkpoint is not None:
        model = unwrap_model(trainer.model)
        number_of_layers_current_model, number_of_layers_best_model = determine_number_of_layers_for_current_and_best_model(trainer=trainer, model=model)

        # Truncate model if size differs
        if number_of_layers_current_model > number_of_layers_best_model:
            truncate_model(model, number_of_layers_current_model, number_of_layers_best_model)

        elif number_of_layers_current_model == number_of_layers_best_model:
            pass
        else:
            logger.warning(
                f"Unexpected model size: Current model has fewer layers ({number_of_layers_current_model}) than best checkpoint ({number_of_layers_best_model})")

def truncate_model(model, number_of_layers_current_model, number_of_layers_best_model):
    """
    Truncates the model architecture by removing the most recently added layers
    to match a target number of layers

    Args:
        model: The neural network model to be truncated
        number_of_layers_current_model: The current integer count of hidden layers
        number_of_layers_best_model: The target integer count of hidden layers to retain

    Returns:
        None
    """
    logger.info(
        f"Truncate model from {number_of_layers_current_model} down to {number_of_layers_best_model} layers")
    model.model.layers = model.model.layers[:number_of_layers_best_model]

    # Update the config so internal Hugging Face checks don't get confused
    model.config.num_hidden_layers = number_of_layers_best_model

def determine_number_of_layers_for_current_and_best_model(trainer, model):
    """
    Extracts the layer counts for both the active model and the best saved checkpoint

    Args:
        trainer: The HuggingFace Trainer instance
        model: The active model

    Returns:
        A tuple of integers: (number_of_layers_current_model, number_of_layers_best_model)
    """
    checkpoint_dir = trainer.state.best_model_checkpoint

    # Check how many layers best checkpoint model has
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path, "r") as f:
        number_of_layers_best_model = json.load(f)["num_hidden_layers"]

    # Check how many layers current model has
    number_of_layers_current_model = len(model.model.layers)

    return number_of_layers_current_model, number_of_layers_best_model