# Configuration Guide

Since this repository uses Hydra, the settings are broken down into logical groups. This guide explains the core parameters available for setting up different experiments:

---

## Core Experiment Setup

These parameters control the logging, reproducibility, and physical execution of your training run.

| Parameter                                                     | Description                                                                                                                                       |
|:--------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------|
| **`experiment.seed`**                                         | Random seed to ensure run reproducibility                                                                                                         |
| **`experiment.name`**                                         | Unique identifier for the experiment (must be set at runtime)                                                                                     |
| **`experiment.group`**                                        | Logical grouping for the experiment, analogous to a "project" in Weights & Biases (W&B)                                                           |
| **`experiment.full_determinism`**                             | Passed to the HF Trainer to strictly enforce PyTorch/Hugging Face determinism                                                                     |
| **`experiment.dry_run`**                                      | Executes a minimal version of the experiment (fewer samples/steps), ideal for local debugging                                                     |
| **`experiment.skip_execution_of_eval_scripts_for_debugging`** | Skips the evaluation pipeline and uses predefined dummy metrics, saves local compute during dry runs                                              |
| **`experiment.offline_run`**                                  | Forces the experiment to run entirely offline                                                                                                     |
| **`experiment.wandb_log_locally`**                            | Stores W&B logs locally (useful for cluster nodes without internet), logs can be synced later                                                     |
| **`experiment.resume_checkpoint_path`**                       | Filepath which can be provided to continue training from a certain checkpoint, e.g. to restore training state after a hardware failure or timeout |
| **`experiment.resume_run_id`**                                | The W&B run ID required to resume logging when using a checkpoint                                                                                 |
| **`experiment.push_to_hub`**                                  | Set to `True` to automatically upload the model to the Hugging Face Hub                                                                           |

---

## Data & Preprocessing

This section defines what data the model sees and how the text is tokenized before training

### Dataset & Tokenizer
| Parameter               | Description                                                                                                                                         |
|:------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| **`dataset.name`**      | Hugging Face dataset identifier, keep this unchanged to maintain compatibility                                                                      |
| **`dataset.subconfig`** | Target specific dataset splits (e.g., `original_strict_small_gold` for 10M or `original_strict_gold` for the 100M BabyLM 2025 dataset)              |
| **`tokenizer.name`**    | Identifier for the tokenizer (e.g., `llama_bpe_3072` for a pre-trained LLaMA BPE using a vocabulary-size of 3072), see `conf/tokenizer` for options |

### Preprocessing
| Parameter                                    | Description                                                                      |
|:---------------------------------------------|:---------------------------------------------------------------------------------|
| **`data_preprocessing.include_punctuation`** | Whether to retain punctuation marks in the training sequences                    |
| **`data_preprocessing.join_sentences`**      | If `True`, multiple sentences are concatenated to fill the exact sequence length |
| **`data_preprocessing.max_input_length`**    | The maximum number of tokens allowed in a single sample                          |

---

## Model & Trainer Dynamics

Standard hyperparameter controls for the model architecture and the Hugging Face Trainer

> **Note:** For model choice, you might want to use the predefined model configurations in `conf/model`

| Parameter                                                  | Description                                                                                             |
|:-----------------------------------------------------------|:--------------------------------------------------------------------------------------------------------|
| **`model.name`**                                           | Identifier for the architecture config (e.g., `llama_15M`), refer to `conf/model`                       |
| **`model.model_kwargs`**                                   | Dictionary for model configuration parameters (e.g. hidden_size or layer counts)                        |
| **`trainer.batch_size`**                                   | Number of samples processed per batch                                                                   |
| **`trainer.lr`**                                           | Peak learning rate (reached after warmup)                                                               |
| **`trainer.num_warmup_steps`**                             | Number of initial steps dedicated to learning rate warmup                                               |
| **`trainer.max_training_steps`**                           | Absolute cutoff point for the training run                                                              |
| **`trainer.max_flops`**                                    | Optional compute limit. Training terminates and evaluates immediately if this FLOP threshold is crossed |
| **`trainer.eval_blimp` / `eval_glue` / `eval_perplexity`** | Booleans toggling regular evaluations on BLiMP, SuperGLUE, or perplexity during checkpoints             |
| **`trainer.lr_scheduler_type`**                            | Determines the decay schedule (e.g., `linear`, `cosine_with_min_lr`)                                    |
| **`trainer.lr_scheduler_kwargs`**                          | Specific scheduler arguments (e.g., `{"min_lr": 1e-4}`)                                                 |

---

## Data Curriculum
> **Note:** For Data Curricula, it is highly recommended to stick to the predefined strategies in `conf/data_curriculum` unless building a custom pacing function.

| Parameter                                           | Description                                                                                                |
|:----------------------------------------------------|:-----------------------------------------------------------------------------------------------------------|
| **`data_curriculum.difficulty_scorer_name`**        | The metric used to sort data difficulty (e.g., n-gram perplexity, sentence length)                         |
| **`data_curriculum.pacing_fn_name`**                | The mathematical curve defining progression speed (`linear`, `step`, `exp`, etc.) of the sample difficulty |
| **`data_curriculum.pacing_fn_kwargs`**              | Further settings for specifying the pacing function, see also Pacing Function Parameters below             |
| **`data_curriculum.start_percent` / `end_percent`** | Percentage of total training steps at which the curriculum pacing begins and ends                          |
| **`data_curriculum.starting_difficulty`**           | The initial data percentile (e.g., `0.25` restricts the model to the easiest 25% of data at step 0)        |

### Pacing Function Parameters

Additional settings for changing the behaviour of the pacing function. Can be provided in `pacing_fn_kwargs` within the data curriculum parameters.

| Parameter                                                  | Description                                                                                                              |
|:-----------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------|
| **`data_curriculum.pacing_fn_kwargs.start_percent`**       | Percentage of total training steps at which the curriculum pacing begins                                                 |
| **`data_curriculum.pacing_fn_kwargs.end_percent`**         | Percentage of total training steps at which the curriculum pacing ends                                                   |
| **`data_curriculum.pacing_fn_kwargs.starting_difficulty`** | The initial data percentile (e.g. `0.25` restricts the model to the easiest 25% of data at step 0)                       |
| **`data_curriculum.pacing_fn_kwargs.max_difficulty`**      | Maximum difficulty percentile to reach at the end of the curriculum (defaults to `1.0`, meaning all data is included)    |
| **`data_curriculum.pacing_fn_kwargs.k_number_of_stages`**  | Optional parameter required for `prop-alpha` aligned pacing to set the total number of stages, will be set automatically |
| **`data_curriculum.pacing_fn_kwargs.alpha`**               | Optional parameter required for `prop-alpha` aligned pacing to determine stage spacing, will be set automatically        |

## Gradual Stacking (MIDAS)
Gradually increases model capacity during training by duplicating middle layers

| Parameter                                                                   | Description                                                                                                                                                                                                                                            |
|:----------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`gradual_stacking.enabled`**                                              | Toggles the gradual stacking feature on or off                                                                                                                                                                                                         |
| **`gradual_stacking.k_number_of_stages`**                                   | Number of growth stages, the final layer count will be `k * layer_per_block`.                                                                                                                                                                          |
| **`gradual_stacking.layer_per_block`**                                      | How many layers constitute a "block": if e.g. set to 2, two layers are duplicated at every growth stage                                                                                                                                                |
| **`gradual_stacking.number_non_embedding_params_compute_equivalent_model`** | Number of parameters of a (constant-sized) model, that the gradual stacking model should be compared to. Based on the size differences, steps will be adjusted in order to app. match compute equivalence (number of steps in config must be the same) |
| **`gradual_stacking.alpha`**                                                | Stage progression factor, a higher alpha leads to faster growth, spending more of the training budget on larger and later model stages                                                                                                                 |
| **`gradual_stacking.align_with_staged_data_curriculum`**                    | Synchronizes model growth directly with data curriculum stage transitions                                                                                                                                                                              |
| **`gradual_stacking.cleaning_optimizer_state`**                             | If `True`, resets optimizer states for newly duplicated parameters rather than copying historical momentum                                                                                                                                             |

## Continual Pre-Training (CPT)
> **Requirement:** CPT interventions strictly require a *Distinct Stage Data Curriculum*

| Parameter                                        | Description                                                                                                                                         |
|:-------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| **`continual_pretraining.enable_lr_reset`**      | Toggles learning rate resets at data stage transitions                                                                                              |
| **`continual_pretraining.max_rewarm_lr`**        | The peak learning rate targeted during the re-warming phase of the last stage (previous stages are linearly increased towards this `max_rewarm_lr`) |
| **`continual_pretraining.rewarm_steps`**         | Fixed number of steps *within a stage* dedicated to linear LR warmup (either rewarm_steps or rewarm_fraction can be used but not both)              |
| **`continual_pretraining.rewarm_fraction`**      | The percentage of steps *within a stage* dedicated to linear LR warmup, preferred over `rewarm_steps` as stage durations might not equal            |
| **`continual_pretraining.data_replay_mode`**     | Controls the inclusion of samples from previous stages, either `None`, `previous_stage_only` or `all_previous_stages`                               |
| **`continual_pretraining.data_replay_fraction`** | The percentage of samples from older stages that should be included in the new stage                                                                |
| **`continual_pretraining.data_replay_decay`**    | Controls how rapidly older stages fade from the replay buffer, approaching `0.0` leads to resembling using the previous stage only                  |

## Infinite Learning Rate Scheduler
> **Requirement:** Infinite Learning Rate Scheduler strictly requires a *Distinct Stage Data Curriculum*

| Parameter                                  | Description                                                                    |
|:-------------------------------------------|:-------------------------------------------------------------------------------|
| **`infinite_lr_scheduler.enabled`**        | Toggles the infinite constant LR schedule                                      |
| **`infinite_lr_scheduler.lr_const`**       | The sustained, constant learning rate held across all training stages          |
| **`infinite_lr_scheduler.lr_const_steps`** | The warmup/transition step count required before settling into the constant LR |
| **`infinite_lr_scheduler.lr_min`**         | The final learning rate used during the post-curriculum annealing phase        |