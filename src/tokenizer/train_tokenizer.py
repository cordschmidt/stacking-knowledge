import os
import hydra
import logging

from tokenizers import Tokenizer, trainers, pre_tokenizers, models, decoders
from datasets import DatasetDict, load_dataset
from transformers import LlamaTokenizerFast
from hydra.core.config_store import ConfigStore
from huggingface_hub import HfApi

from src.config import BabyLMConfig
from src.helper.setup_environment import setup_environment

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

logger = logging.getLogger(__name__)


def load_dataset_from_huggingface(cfg: BabyLMConfig):
    """
    Loads the designated dataset from the Hugging Face Hub based on the provided configuration

    Args:
        cfg: The BabyLM configuration object containing dataset settings

    Returns:
        A DatasetDict containing the loaded Hugging Face dataset splits
    """
    dataset: DatasetDict = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        token=os.environ["HF_READ_TOKEN"],
    )
    return dataset

def preprocess_dataset_for_tokenizer(dataset):
    """
    Prepares the dataset for tokenizer training by removing all non-text columns

    Args:
        dataset: The raw DatasetDict loaded from Hugging Face

    Returns:
        A DatasetDict stripped of all columns except for the 'text' column
    """
    # Remove non text columns
    non_text_columns = [col for col in dataset.column_names["train"] if col != "text"]
    dataset = dataset.remove_columns(non_text_columns)
    return dataset

def train_bpe_tokenizer(dataset, vocab_size, bpe_tokenizer_json_path):
    """
    Initializes, trains and saves a Byte-Pair Encoding (BPE) tokenizer using the provided dataset

    Args:
        dataset: The preprocessed DatasetDict containing the text data
        vocab_size: The integer representing the target vocabulary size
        bpe_tokenizer_json_path: The string file path where the raw trained tokenizer JSON will be saved

    Returns:
        None
    """
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer (byte-level like GPT/LLaMA)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    tokenizer.decoder = decoders.ByteLevel()

    # Define trainer with special tokens
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Create an iterator over all training examples
    def train_iterator():
        for text in dataset["train"]["text"]:
            yield text

    # Train tokenizer from iterator
    tokenizer.train_from_iterator(train_iterator(), trainer=trainer)

    # Save the tokenizer
    tokenizer.save(bpe_tokenizer_json_path, pretty=True)

def save_tokenizer_as_llama_tokenizer(bpe_tokenizer_json_path, llama_tokenizer_save_path):
    """
    Wraps the raw trained BPE tokenizer JSON into a LlamaTokenizerFast class and saves it

    Args:
        bpe_tokenizer_json_path: The string path to the raw trained BPE tokenizer JSON file
        llama_tokenizer_save_path: The string directory path where the Hugging Face compatible LLaMA tokenizer will be saved

    Returns:
        None
    """
    # Create and save the new tokenizer as a LlamaTokenizerFast
    llama_tokenizer = LlamaTokenizerFast(
        tokenizer_file=bpe_tokenizer_json_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    llama_tokenizer.save_pretrained(llama_tokenizer_save_path)
    print(f"LLaMA tokenizer saved to {llama_tokenizer_save_path}")

def upload_tokenizer_to_hf(llama_tokenizer_save_path):
    """
    Creates a repository on the Hugging Face Hub and uploads the saved LLaMA tokenizer folder

    Args:
        llama_tokenizer_save_path: The string directory path containing the LLaMA tokenizer files to upload

    Returns:
        None
    """
    repo_id = f"stacking-babylm-2025/{llama_tokenizer_save_path}"
    api = HfApi(token=os.getenv("HF_WRITE_TOKEN"))
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True  # If it exists, it just continues
    )
    api.upload_folder(
        folder_path=llama_tokenizer_save_path,
        repo_id=repo_id,
        repo_type="model",
    )

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: BabyLMConfig):
    """
    The main execution pipeline for setting up the environment, loading/preprocessing data,
    training the tokenizer, and uploading it to Hugging Face

    Args:
        cfg: The BabyLM configuration object instantiated by Hydra

    Returns:
        None
    """
    setup_environment(cfg)
    dataset = load_dataset_from_huggingface(cfg)
    dataset = preprocess_dataset_for_tokenizer(dataset)
    vocab_size = cfg.model.model_kwargs.vocab_size
    bpe_tokenizer_json_path = f"tokenizer_{vocab_size}.json"
    train_bpe_tokenizer(dataset, vocab_size, bpe_tokenizer_json_path)
    llama_tokenizer_save_path = f"llama_tokenizer_small_{vocab_size}"
    save_tokenizer_as_llama_tokenizer(bpe_tokenizer_json_path, llama_tokenizer_save_path)
    upload_tokenizer_to_hf(llama_tokenizer_save_path)


if __name__ == "__main__":
    main()
