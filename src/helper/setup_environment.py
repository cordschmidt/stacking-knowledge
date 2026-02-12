import logging
import os
import random
import numpy as np
import torch

from dotenv import load_dotenv

from src.config import BabyLMConfig

# Logger for this file
logger = logging.getLogger(__name__)

TORCH_RUN_ENV_KEYS = [
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_PORT",
    "MASTER_ADDR",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "PYTHON_EXEC",
]

def setup_environment(cfg: BabyLMConfig):
    # Load environment variables to be able to work locally
    load_dotenv()

    # Check & log pytorch / CUDA versions
    log_cuda_info()

    # Check for huggingface tokens in .env
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    # Set seed
    set_seed(cfg.experiment.seed)

def log_cuda_info():
    logger.info("=== PyTorch / CUDA diagnostics ===")
    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.version.cuda: {torch.version.cuda}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available — running on CPU")

    logger.info("=================================")

def set_seed(seed: int) -> None:
    """Sets seed for reproducibility"""
    if seed < 0:
        logger.warning("Skipping seed setting for reproducibility")
        logger.warning(
            "If you would like to set a seed, set seed to a positive value in config"
        )
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)