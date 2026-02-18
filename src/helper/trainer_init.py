import os

from transformers.training_args import TrainingArguments

from src.config import BabyLMConfig
from src.custom_trainer import CustomTrainer

def create_trainer(cfg: BabyLMConfig, model, tokenizer, train_dataset, eval_dataset, curriculum_learning_table):
    # Create a TrainingArguments object which configures the Hugging Face Trainer behavior
    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",  # Directory to save checkpoints and outputs
        overwrite_output_dir=False,  # Don't overwrite existing output directory
        do_train=True,  # Enable training
        do_eval=True,  # Enable evaluation during training
        do_predict=False,  # Disable prediction on test set (can be enabled separately)
        full_determinism=cfg.experiment.full_determinism,

        # Batch size per device (GPU/CPU)
        per_device_train_batch_size=cfg.trainer.batch_size,

        learning_rate=cfg.trainer.lr,  # Learning rate for optimizer
        lr_scheduler_type=cfg.trainer.lr_scheduler_type, # Defines type of learning rate schedule
        lr_scheduler_kwargs=cfg.trainer.lr_scheduler_kwargs, # Additional parameters for some lr schedulers
        max_steps=cfg.trainer.max_training_steps,  # Max number of training steps
        warmup_steps=cfg.trainer.num_warmup_steps,  # Number of warmup steps for learning rate scheduler

        seed=cfg.experiment.seed,  # Random seed for reproducibility
        data_seed=cfg.experiment.seed,

        # Evaluation strategy to evaluate every few steps
        eval_strategy="steps",

        # Evaluate every N steps (1/4 of total training unless dry run, which uses 1/2)
        eval_steps=cfg.trainer.max_training_steps // (2 if cfg.experiment.dry_run else 16),

        # Save checkpoint every N steps (same as eval steps)
        save_steps=cfg.trainer.max_training_steps // (2 if cfg.experiment.dry_run else 16),

        # Logging frequency, more frequent for dry run
        logging_steps=cfg.trainer.max_training_steps // (100 if cfg.experiment.dry_run else 1000),

        run_name=cfg.experiment.name,  # Name for the training run (used in logs/wandb)

        # Report logs to wandb unless running offline
        report_to=(None if cfg.experiment.offline_run else ["wandb"]),

        save_strategy="steps",  # Save checkpoints based on steps rather than epochs

        # TODO: Adjust & enable huggingface hub model upload
        hub_strategy="every_save",  # Strategy for pushing to Hugging Face Hub
        push_to_hub=False,  # Disable pushing model to Hugging Face Hub currently

        # Model ID for pushing to Hugging Face Hub (only set if not offline)
        hub_model_id=(
            None
            if cfg.experiment.offline_run
            else f"cambridge-climb/{cfg.experiment.group}-{cfg.model.name}-model"
        ),

        # Token to authenticate pushing to Hub, read from environment (only if not offline)
        hub_token=(
            None
            if cfg.experiment.offline_run
            else os.environ["HF_WRITE_TOKEN"]
        ),

        # Drop the last incomplete batch during training if curriculum is used (to avoid breaking curriculum logic)
        dataloader_drop_last=cfg.data_curriculum is not None,

        remove_unused_columns=False,  # Keep all columns in datasets

        load_best_model_at_end=True,  # After training finishes, load the checkpoint with best eval metric

        metric_for_best_model="eval_perplexity_mean",  # Metric to select best model checkpoint

        greater_is_better=False,  # Smaller perplexity is better

        ddp_find_unused_parameters=False,  # Optimization for distributed training to reduce overhead

        ddp_timeout=28800,  # Timeout for distributed training jobs (8 hours, default is 30 minutes)
    )
    trainer = CustomTrainer(
        hydra_config=cfg,
        dry_run=cfg.experiment.dry_run,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        curriculum_learning_table=curriculum_learning_table,
    )
    return trainer, training_args