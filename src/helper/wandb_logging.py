import logging
import os
import wandb

from wandb.errors import CommError as WandbCommError
from omegaconf import OmegaConf

from src.config import BabyLMConfig

# Logger for this file
logger = logging.getLogger(__name__)


def disable_wandb():
    # Disable wandb by setting environment variables so no logs are sent remotely
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"

def setup_wandb_environment(cfg: BabyLMConfig):
    # Log locally first (e.g., cluster with no internet) and sync later
    if cfg.experiment.wandb_log_locally:
        os.environ["WANDB_MODE"] = "offline"

    # Retrieve wandb user from environment
    wandb_user = os.environ["WANDB_USER"]

    # Set wandb environment variables picked up by Hugging Face Trainer for logging
    os.environ["WANDB_PROJECT"] = cfg.experiment.group
    os.environ["WANDB_ENTITY"] = wandb_user

    # Convert the OmegaConf config to a plain Python dict for wandb to track config params
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # If resuming from a checkpoint, setup wandb to resume the same run
    if cfg.experiment.resume_checkpoint_path:
        resume_run_id = cfg.experiment.resume_run_id
        if resume_run_id is None:
            raise RuntimeError(
                "resume_run_id must be set if resume_checkpoint_path is set"
            )
        # Set environment variables to tell wandb which run to resume
        os.environ["WANDB_RUN_ID"] = resume_run_id
        os.environ["WANDB_RESUME"] = "allow"

    return wandb_user


def init_wandb_and_curriculum_table(cfg: BabyLMConfig, wandb_user: str):
    # Initialize wandb only on main process (rank 0) to avoid duplicate logs
    if int(os.environ.get("RANK", "0")) != 0:
        return None

    wandb.init(
        entity=wandb_user,
        project=cfg.experiment.group,
        name=cfg.experiment.name,
        config=wandb.config,
        id=cfg.experiment.resume_run_id,
        resume="allow",
    )

    # Define the columns of the curriculum learning table that tracks training data sampling and difficulty
    table_columns = [
        "global_step",
        "data_difficulty_percentile",
        "data_sampled_percentile",
        "num_samples",
        "max_difficulty_score",
        "min_difficulty_score",
        "median_difficulty_score",
        "data_samples",
    ]

    if cfg.experiment.resume_run_id:
        # Attempt to load previously saved curriculum learning table artifact from wandb if resuming
        try:
            artifact_path = (
                f"{wandb_user}/{cfg.experiment.group}/run-"
                f"{cfg.experiment.resume_run_id}-traincurriculum_learning_table:latest"
            )
            return wandb.run.use_artifact(artifact_path).get("train/curriculum_learning_table")
        except WandbCommError:
            # If not found or error, log warning and create a new empty table later on
            logger.warning("Could not find existing curriculum table. Creating new one.")

    # If not resuming or no artifact could be found, create a new empty curriculum learning table
    return wandb.Table(columns=table_columns)

def enable_wandb_logging(cfg: BabyLMConfig):
    # If running in offline mode (e.g., no internet or want to disable wandb)
    if cfg.experiment.offline_run:
        # disable wandb by setting environment variables so no logs are sent remotely
        disable_wandb()
        curriculum_learning_table = None  # No curriculum learning table since wandb is disabled
    else:
        # Set up wandb environment variables and config
        wandb_user = setup_wandb_environment(cfg)

        # Initialize wandb and curriculum learning table on main process
        curriculum_learning_table = init_wandb_and_curriculum_table(cfg, wandb_user)

    return curriculum_learning_table