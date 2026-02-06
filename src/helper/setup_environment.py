import logging
import os
import random
import numpy as np
import torch

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.config import BabyLMConfig
from src.data_curriculum.difficulty_scorer.stages import NUM_STAGES

DRY_RUN_TRAIN_STEPS = 100
DRY_RUN_WARMUP_STEPS = 10
DIFFICULTY_SCORER_UPDATE = 75

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

    # Additional validations that cannot be done in hydra directly easily
    do_additional_config_validations(cfg=cfg)
    adjust_parameters_in_config_for_special_setups(cfg=cfg)

    # Check for huggingface tokens in .env
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    # Log configuration
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.experiment.seed)


def do_additional_config_validations(cfg: BabyLMConfig):
    validate_prop_alpha_aligned_data_curriculum_pacing(cfg=cfg)


def validate_prop_alpha_aligned_data_curriculum_pacing(cfg: BabyLMConfig):
    if cfg.data_curriculum and \
       cfg.data_curriculum.difficulty_scorer_name == "staged_data_split" and \
       cfg.data_curriculum.pacing_fn_name == "prop_alpha":

        validate_that_number_of_prop_alpha_stages_equal_number_of_datasets(cfg=cfg)

def validate_that_number_of_prop_alpha_stages_equal_number_of_datasets(cfg: BabyLMConfig):
    # Check stage count alignment
    if cfg.gradual_stacking.k_number_of_stages != NUM_STAGES:
        raise ValueError(
            f"Alignment Error: 'prop_alpha' + 'staged_data_split' requires "
            f"gradual_stacking.k_number_of_stages ({cfg.gradual_stacking.k_number_of_stages}) "
            f"to equal the data's NUM_STAGES ({NUM_STAGES})."
        )


def adjust_parameters_in_config_for_special_setups(cfg: BabyLMConfig):
    # Adjust training parameters in dry run for faster testing & debugging
    if cfg.experiment.dry_run:
        adjust_params_for_dry_run(cfg=cfg)

    if cfg.data_curriculum.pacing_fn_name == "prop_alpha":
        insert_gradual_stacking_parameters_into_pacing_fn(cfg=cfg)

    if cfg.data_curriculum and \
            cfg.data_curriculum.difficulty_scorer_name == "staged_data_split" and \
            cfg.data_curriculum.pacing_fn_name == "prop_alpha":
        force_ignoring_dataset_sizes_in_staged_data_curriculum(cfg=cfg)

def adjust_params_for_dry_run(cfg: BabyLMConfig):
    logger.info(
        "Running in dry run mode -- overriding config with values: "
    )
    logger.info(f"\t max_training_steps: {DRY_RUN_TRAIN_STEPS}")
    logger.info(f"\t num_warmup_steps: {DRY_RUN_WARMUP_STEPS}")
    cfg.trainer.max_training_steps = DRY_RUN_TRAIN_STEPS
    cfg.trainer.num_warmup_steps = DRY_RUN_WARMUP_STEPS

    if (
            cfg.data_curriculum is not None
            and cfg.data_curriculum.difficulty_scorer_kwargs is not None
    ):

        if (
                cfg.data_curriculum.difficulty_scorer_kwargs.get("update")
                is not None
        ):
            cfg.data_curriculum.difficulty_scorer_kwargs["update"] = (
                DIFFICULTY_SCORER_UPDATE
            )
            logger.info(
                f"\t data curriculum difficulty scorer update: {DIFFICULTY_SCORER_UPDATE}"
            )

def insert_gradual_stacking_parameters_into_pacing_fn(cfg: BabyLMConfig):
    k_stages_from_gradual_stacking_config = cfg.gradual_stacking.k_number_of_stages
    alpha_from_gradual_stacking_config = cfg.gradual_stacking.alpha
    cfg.data_curriculum.pacing_fn_kwargs["k_number_of_stages"] = k_stages_from_gradual_stacking_config
    cfg.data_curriculum.pacing_fn_kwargs["alpha"] = alpha_from_gradual_stacking_config

def force_ignoring_dataset_sizes_in_staged_data_curriculum(cfg: BabyLMConfig):
    # Check proportion flag
    scorer_kwargs = cfg.data_curriculum.difficulty_scorer_kwargs or {}
    if scorer_kwargs.get("account_for_dataset_proportions") is not False:
        logger.info(f"In cfg.data_curriculum.difficulty_scorer_kwargs the attribute 'account_for_dataset_proportions' was set to True")
        logger.info(f"In order to align the data curriculum with the prop-alpha stages, 'account_for_dataset_proportions' has to be set to False")
        cfg.data_curriculum.difficulty_scorer_kwargs["account_for_dataset_proportions"] = False
        logger.info(f"'account_for_dataset_proportions' was set to {cfg.data_curriculum.difficulty_scorer_kwargs["account_for_dataset_proportions"]}")

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