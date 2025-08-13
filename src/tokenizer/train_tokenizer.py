import os
import hydra
import logging

from tokenizers import Tokenizer, trainers, pre_tokenizers, models
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
    dataset: DatasetDict = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        token=os.environ["HF_READ_TOKEN"],
    )
    return dataset

def preprocess_dataset_for_tokenizer(dataset):
    # Remove non text columns
    non_text_columns = [col for col in dataset.column_names["train"] if col != "text"]
    dataset = dataset.remove_columns(non_text_columns)
    return dataset

def train_bpe_tokenizer(dataset, vocab_size, bpe_tokenizer_json_path):
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())

    # Set pre-tokenizer (byte-level like GPT/LLaMA)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

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
    api = HfApi(token=os.getenv("HF_WRITE_TOKEN"))
    api.upload_folder(
        folder_path=llama_tokenizer_save_path,
        repo_id=f"stacking-babylm-2025/{llama_tokenizer_save_path}",
        repo_type="model",
    )

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: BabyLMConfig):
    setup_environment(cfg)
    dataset = load_dataset_from_huggingface(cfg)
    dataset = preprocess_dataset_for_tokenizer(dataset)
    vocab_size = cfg.model.model_kwargs.vocab_size
    bpe_tokenizer_json_path = f"tokenizer_{vocab_size}.json"
    train_bpe_tokenizer(dataset, vocab_size, bpe_tokenizer_json_path)
    llama_tokenizer_save_path = f"llama_tokenizer_{vocab_size}"
    save_tokenizer_as_llama_tokenizer(bpe_tokenizer_json_path, llama_tokenizer_save_path)
    upload_tokenizer_to_hf(llama_tokenizer_save_path)

if __name__ == "__main__":
    main()
