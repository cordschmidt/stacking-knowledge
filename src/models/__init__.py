import logging

from transformers import PreTrainedModel
from ..config import BabyLMConfig
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY

from .llama import *

# Set up a logger for this module
logger = logging.getLogger(__name__)

def validate_model_kwargs(model_kwargs: dict):
    # Check that the minimal required params "hidden_size" and "vocab_size" are present
    assert (
            "hidden_size" in model_kwargs and "vocab_size" in model_kwargs
    ), "`hidden_size` and `vocab_size` must be defined in model_kwargs"

def load_or_initialize_model(hf_model_config, model_name):
    # Pretrained weights will be loaded when path is given (local or huggingface hub)
    if getattr(hf_model_config, "name_or_path", None):
        # Load pretrained model
        model = MODEL_REGISTRY[model_name].from_pretrained(
            hf_model_config.name_or_path,
            config=hf_model_config
        )
        logger.info(f"Loaded pretrained model from '{hf_model_config.name_or_path}'")
    # When no path is given, initialize the model parameters
    else:
        # Initialize model from scratch using the config
        model = MODEL_REGISTRY[model_name](hf_model_config)
        logger.info(f"Initialized model '{model_name}' from scratch")
    return model

def load_base_model(cfg: BabyLMConfig) -> PreTrainedModel:
    """
    Loads a base language model based on the configuration provided, supports loading pretrained weights or initializing from scratch
    """

    # Extract and validate model params from the config
    model_kwargs = cfg.model.model_kwargs
    validate_model_kwargs(model_kwargs)

    # Look up the model and config class based on the model name
    model_name = cfg.model.name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' is not registered")

    # Instantiate the configuration class for the model
    hf_model_config_cls = CONFIG_REGISTRY[model_name]
    hf_model_config = hf_model_config_cls(**model_kwargs)

    model = load_or_initialize_model(hf_model_config, model_name)

    # Log detailed info about model parameters and training status
    logger.debug("Model parameter list:")
    for i, (name, param) in enumerate(model.named_parameters()):
        logger.debug(f"{i}: {name} - requires_grad: {param.requires_grad}")

    return model
