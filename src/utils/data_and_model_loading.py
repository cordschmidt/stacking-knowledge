import logging
import os
import torch

from datasets import DatasetDict, load_dataset

from src.config import BabyLMConfig
from src.tokenizer import load_tokenizer
from src.models import load_base_model
from src.utils.dataset_preprocessor import DatasetPreprocessor

DRY_RUN_SUBSAMPLE_FACTOR = 1000 // (10 if torch.cuda.device_count() > 1 else 1)

# Logger for this file
logger = logging.getLogger(__name__)

def load_dataset_model_and_tokenizer(cfg: BabyLMConfig):
    # Loading dataset from huggingface hub
    logger.info("Loading dataset")
    dataset: DatasetDict = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        use_auth_token=os.environ["HF_READ_TOKEN"],
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

    # Subsample the training dataset in dry mode by selecting every Nth example
    if cfg.experiment.dry_run:
        logger.info(
            f"Running in dry run mode -- subsampling dataset by {DRY_RUN_SUBSAMPLE_FACTOR}x"
        )
        train_dataset = train_dataset.select(
            range(0, train_dataset.num_rows, DRY_RUN_SUBSAMPLE_FACTOR)
        )

    # Preprocess every sample of the evaluation data
    eval_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
    )
    return train_dataset, eval_dataset