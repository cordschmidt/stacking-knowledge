import logging
import os
import hydra

from hydra.core.config_store import ConfigStore

# Local imports
from src.config import BabyLMConfig
from src.custom_trainer import CustomTrainer
from src.helper.data_and_model_loading import load_dataset_model_and_tokenizer
from src.helper.setup_environment import setup_environment
from src.helper.trainer_init import create_trainer
from src.helper.wandb_logging import enable_wandb_logging

# Logger for this file
logger = logging.getLogger(__name__)

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    setup_environment(cfg)
    model, tokenizer, dataset = load_dataset_model_and_tokenizer(cfg)

if __name__ == "__main__":
    main()