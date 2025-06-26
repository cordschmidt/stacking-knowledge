import logging
import os

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.config import BabyLMConfig
from src.utils.setup import set_seed

DRY_RUN_TRAIN_STEPS = 100
DRY_RUN_WARMUP_STEPS = 10
DIFFICULTY_SCORER_UPDATE = 75

# Logger for this file
logger = logging.getLogger(__name__)

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

def setup_environment(cfg: BabyLMConfig):
    # Load environment variables to be able to work locally
    load_dotenv()

    # Check for huggingface tokens in .env
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    # Log configuration
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.experiment.seed)

    # Adjust training parameters in dry run for faster testing & debugging
    if cfg.experiment.dry_run:
        adjust_params_for_dry_run(cfg=cfg)