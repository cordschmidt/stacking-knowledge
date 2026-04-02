# Stacking Knowledge: Enhancing Curriculum Learning with Gradual Stacking and Continual Pre-Training 🧠⚡

This repository contains the codebase for my Master's Thesis "Stacking Knowledge: Enhancing Curriculum Learning with Gradual Stacking and Continual Pre-Training" in the realm of investigating advanced training interventions in Small Language Models (SLMs), specifically in the context of the **BabyLM Challenge** (10M/100M word regimes). 

> **Acknowledgements:** This project builds upon and expands the foundational **CLIMB** (Curriculum Learning for Infant-inspired Model Building) architecture originally developed by [codebyzeb/CLIMB](https://github.com/codebyzeb/CLIMB) and detailed in their [CLIMB paper](https://aclanthology.org/2023.conll-babylm.10/). While CLIMB introduced strict data curricula, this repository extends that framework to address catastrophic forgetting and computational efficiency through dynamic pacing, Continual Pre-Training (CPT), and progressive architectural growth.

The project explores how models learn structurally and temporally, intersecting three core paradigms:
1. **Data Curricula:** Structuring training data progressively from simple to complex.
2. **Continual Pre-Training (CPT):** Mitigating catastrophic forgetting during sequential data stages using Data Replay and Learning Rate Rewarming.
3. **MIDAS Gradual Stacking:** Progressively growing the model's depth (layer count) during training to accelerate convergence and reduce total FLOPs.

---

## 📁 Repository Structure & Logic Map

The project is heavily modularized using **Hydra** for configuration. Here is where the core logic lives:

```text
stacking-knowledge_clean/
├── conf/                     # Hydra configs (models, tokenizers, curricula, experiments)
├── slurm_scripts/            # Pre-configured batch scripts for HPC cluster submission
├── eval_pipeline/            # Submodule containing the official BabyLM evaluation tools
├── src/                      # Core Python source code
│   ├── custom_trainer.py     # Subclasses HF Trainer to inject curriculum sampling logic
│   ├── data_curriculum/      # Curriculum pacing, dynamic triggers, and difficulty scorers
│   ├── continual_pretraining/# LR rewarming callbacks and Infinite LR schedulers
│   ├── gradual_stacking/     # MIDAS model growth architecture and global step schedulers
│   ├── models/               # Model architecture definitions (e.g., LLaMA)
│   ├── tokenizer/            # Custom BPE/Unigram tokenizer training scripts
│   └── helper/               # Data loading, WandB logging, and visualization utilities
├── train.py                  # Main entry point for training
└── setup_environment.sh      # Setup script for SLURM/HPC clusters
```

## 🏗️ Core Architecture Deep-Dive

1. Data Curricula (src/data_curriculum/)
Handles what data the model sees and when.

datasampler.py: Replaces standard random shuffling. It queries pacing functions to restrict the model's data exposure to the currently active curriculum stage.

difficulty_scorer/: Enforces strict sequential stages. If the model is in Stage 2, it filters out all other data. It also handles Data Replay, dynamically assigning sampling weights to historical tokens (e.g., pulling 5% of tokens from past stages) to prevent forgetting.

dynamic_curriculum_callback.py: Monitors validation perplexity. If the model stops improving and begins to overfit on a specific stage, this callback intercepts the static schedule and forces an early transition to the next dataset.

2. Continual Pre-Training (CPT) (src/continual_pretraining/)
Handles the stabilization of the learning process across jarring dataset transitions.

learning_rate_reset_callback.py: Listens for stage transitions. When a new dataset starts, it automatically re-warms the optimizer's learning rate (e.g., back to 10% of the maximum LR) to restore network plasticity.

infinite_lr_scheduler.py: A custom learning rate schedule designed to pair with Dynamic Curricula, sustaining a constant learning rate when stage lengths are unpredictable.

3. Gradual Stacking (src/gradual_stacking/)
Implements the progressive depth-growth algorithm (MIDAS).

stacking_callback.py: Pauses the training loop at predefined computational boundaries, duplicates the middle block of Transformer layers, dynamically updates the optimizer state to account for the new parameters, and resumes training.

scheduler.py: Calculates the exact global steps where architectural growth should occur, allowing it to align perfectly with data curriculum stage transitions.

## 🚀 Environment Setup

Option A: Local Workstation
If you are running on a local machine (e.g., a local GPU rig or MacBook), ensure you have Conda/Miniconda installed:

```Bash
git clone --recurse-submodules <YOUR_REPO_URL>
cd stacking-knowledge_clean
```

### Creates a py313 env, installs PyTorch, dependencies, and eval pipeline requirements

```Bash
bash setup_environment.sh
conda activate py313
```

Option B: HPC Cluster (SLURM)
If deploying on a cluster, you must load environment modules (like CUDA) before installation.

```Bash
git clone --recurse-submodules <YOUR_REPO_URL>
cd stacking-knowledge_clean
```

### Loads cluster modules, builds the Conda environment

```Bash
bash setup_environment.sh
```

### Recommended: Pre-download Hugging Face datasets to the cluster's local scratch space

```Bash
bash load_resources_on_cluster.sh
```

Authentication (Required)
The project logs to Weights & Biases and pulls from Hugging Face. Do not hardcode your tokens. Export them in your terminal or add them to your ~/.bashrc:

```Bash
export HF_TOKEN="<YOUR_HF_TOKEN_HERE>"
export WANDB_API_KEY="<YOUR_WANDB_API_KEY_HERE>"
```

```
huggingface-cli login --token $HF_TOKEN
wandb login $WANDB_API_KEY
```

## 🧪 Running Experiments
All training runs are orchestrated through Hydra, meaning you can swap model sizes, tokenizers, and curriculum strategies directly from the command line without modifying Python code.

### Running Locally
To execute a standard random baseline (no curriculum interventions):

```Bash
python train.py \
    model=llama_15M \
    tokenizer=llama_bpe_3072 \
    data_curriculum=null \
    experiment.wandb_project="my_thesis" \
    experiment.name="random_baseline"
```
To execute a run utilizing Continual Pre-Training (Data Replay + LR Rewarming):

```Bash
python train.py \
    model=llama_15M \
    data_curriculum=linear_staged_data_split \
    data_curriculum.scorer.data_replay=0.05 \
    data_curriculum.scorer.replay_mode="all_previous" \
    experiment.learning_rate_reset_target=0.1
```

Submitting to a Cluster (SLURM)
For heavy, multi-seed experiments, utilize the pre-configured scripts in slurm_scripts/.

Open the script you want to run (e.g., slurm_scripts/run_baselines_gradual_stacking.sh).

Verify the #SBATCH headers (partition, account, mail-user) match your cluster's requirements.

### Submit the job:

```Bash
sbatch slurm_scripts/run_baselines_gradual_stacking.sh
```

## 📊 Evaluation Pipeline

We utilize a submodule of the official BabyLM 2025 Evaluation Pipeline to benchmark linguistic capabilities (BLiMP, SuperGLUE, Age of Acquisition).

Once a model finishes training, its checkpoints are saved locally. To evaluate a checkpoint zero-shot:

```Bash
cd eval_pipeline
```

### Pass the path to your final checkpoint folder

```Bash
bash eval_zero_shot.sh ../checkpoints/your_model_run_name/checkpoint-final
```
The evaluation script will automatically detect the LLaMA architecture and tokenizer from your checkpoint directory and output JSON results for visualization.


