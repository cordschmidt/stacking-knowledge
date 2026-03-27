import logging
import os
import hydra

from hydra.core.config_store import ConfigStore

# Local imports
from src.config import BabyLMConfig
from src.custom_trainer import CustomTrainer
from src.helper.cleanup import cleanup_output_dir
from src.helper.data_and_model_loading import load_dataset_model_and_tokenizer, preprocess_data, \
    truncate_model_if_best_checkpoint_size_differs
from src.helper.setup_environment import setup_environment
from src.helper.trainer_init import create_trainer
from src.helper.validate_config import validate_and_adjust_config, \
    consider_step_adjustment_for_compute_equivalent_model_training
from src.helper.wandb_logging import enable_wandb_logging

# Logger for this file
logger = logging.getLogger(__name__)

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

def train_and_evaluate(cfg: BabyLMConfig, trainer: CustomTrainer, training_args):
    # If no checkpoint path is provided to resume from, run initial evaluation before training
    if not cfg.experiment.resume_checkpoint_path:
        # Evaluate the initial model to get baseline metrics
        trainer.evaluate()

    # Start or resume training
    trainer.train(resume_from_checkpoint=cfg.experiment.resume_checkpoint_path)

    # Manually load the model to restore its exact architecture from the best step,
    # overriding the fully-grown model in case of gradual stacking
    truncate_model_if_best_checkpoint_size_differs(trainer=trainer)

    # After training completes, set flags to enable evaluation on all the benchmark tasks / metrics
    trainer.eval_glue = True
    trainer.eval_blimp = True
    trainer.eval_perplexity = True

    # Evaluate the best model found during training
    # Since 'load_best_model_at_end=True' is set in TrainingArguments, the best checkpoint is already loaded before this evaluation.
    trainer.evaluate(metric_key_prefix="eval_best")

    cleanup_output_dir(output_dir=training_args.output_dir)

    # Save the best model checkpoint explicitly to a "best_model" subdirectory
    # trainer.save_model(output_dir=os.path.join(training_args.output_dir, "best_model"))


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    setup_environment(cfg)
    validate_and_adjust_config(cfg=cfg)
    model, tokenizer, dataset = load_dataset_model_and_tokenizer(cfg)
    consider_step_adjustment_for_compute_equivalent_model_training(cfg=cfg, model=model)
    train_dataset, eval_dataset, dev_dataset = preprocess_data(cfg=cfg, tokenizer=tokenizer, dataset=dataset)
    curriculum_learning_table = enable_wandb_logging(cfg)
    trainer, training_args = create_trainer(cfg=cfg, model=model, tokenizer=tokenizer, train_dataset=train_dataset, eval_dataset=eval_dataset, dev_dataset=dev_dataset, curriculum_learning_table=curriculum_learning_table)
    train_and_evaluate(cfg, trainer, training_args)

if __name__ == "__main__":
    main()