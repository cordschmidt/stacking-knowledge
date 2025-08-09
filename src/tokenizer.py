import logging
import os

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from src.config import BabyLMConfig

# Set up logging for this module
logger = logging.getLogger(__name__)

def load_tokenizer(cfg: BabyLMConfig) -> PreTrainedTokenizerFast:
    """
    Loads a tokenizer based on the config settings
    """
    # Define keys to remove from tokenizer_kwargs
    remove_keys = ["name"]
    # Get tokenizer_kwargs and remove params that are not needed
    tokenizer_kwargs = {
        str(key): val
        for key, val in cfg.tokenizer.items()
        if key not in remove_keys and val is not None
    }

    # Load the tokenizer from Hugging Face hub or local path
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        cfg.tokenizer.name,                         # Tokenizer name
        token=os.environ["HF_READ_TOKEN"],          # HF Token needed to access
        **tokenizer_kwargs                          # Additional tokenizer args
    )

    # Ensure the tokenizer is the fast type
    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer must be a PreTrainedTokenizerFast"

    return tokenizer

