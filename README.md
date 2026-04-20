# Stacking Knowledge: Enhancing Curriculum Learning with Gradual Stacking and Continual Pre-Training

This repository contains the codebase for my Master's Thesis "Stacking Knowledge: Enhancing Curriculum Learning with Gradual Stacking and Continual Pre-Training" in the realm of investigating advanced training interventions in Small Language Models (SLMs), specifically in the context of the [**BabyLM 2025 Challenge**](https://babylm.github.io/) (10M/100M word regimes). 

> **Acknowledgements:** This project builds upon and expands the foundational Curriculum Learning Framework originally developed by Martinez et al. (2023) (see [codebyzeb/CLIMB](https://github.com/codebyzeb/CLIMB)) and detailed in [CLIMB – Curriculum Learning for Infant-inspired Model Building](https://aclanthology.org/2023.conll-babylm.10/)

While several valuable training methods (including vocabulary, objective, and data curricula) were introduced by the CLIMB framework, its data curriculum was unable to reach the performance of the random-order baseline. To shed light on why this occurred, the work of Martinez et al. (2023) is expanded by:
 
**1. Distinct Stage Data Curriculum:** 

In the original framework, an expanding corpus approach was used, where batches were sampled from an ever-growing pool of data. This caused the first corpus to be heavily over-indexed, as it was included in every subsequent stage. Instead, a distinct stage curriculum is introduced, where training is conducted sequentially on each individual corpus to ensure balanced exposure.

**2. Continual Pre-Training (CPT) Integration:** 

In the original setup, learning rates might be decayed too much for later corpora to be learned effectively. To solve this, CPT techniques - which are typically used for updating LLMs without retraining from scratch - were adapted for the data curriculum approach. By implementing learning-rate re-warming on corpus transitions and incorporating data replay (where a small amount of data from previous stages is retained) in accordance with [Gupta et al. (2023)](https://arxiv.org/abs/2308.04014), the reduction of catastrophic forgetting and the improvement of adaptation to new data were targeted.

**3. Dynamic Data Curriculum:** 

During experiments, it was revealed that overfitting during the training stages was caused by the staged data curriculum approach. To counteract this, a dynamic curriculum was implemented, where progression to the next data stage is triggered automatically as soon as overfitting on a dev-dataset is detected. Since the number of steps for a learning rate reset is then unknown, infinite learning rate schedules were implemented in accordance with [Singh et al. (2025)](https://arxiv.org/abs/2503.02844).

**4. Gradual Stacking:** 

Additionally, MIDAS (MIDdle grAdual Stacking) by [Saunshi et al. (2024)](https://arxiv.org/abs/2409.19044) was implemented in this repository. It was investigated whether training convergence can be accelerated by gradually increasing model size (achieved by duplicating a middle layer or block of layers) on small language models (such as 15M parameter LLaMA-style models) trained on restricted datasets like the BabyLM corpus. Furthermore, this stacking was aligned with the staged data curriculum so that potential synergistic effects could be evaluated.

> **Note:** The framework has been further adapted to support **LLaMA-style architectures** and is adapted to the **BabyLM 2025 Challenge**, which includes the integration of the official [BabyLM 2025 Evaluation Pipeline](https://github.com/babylm/evaluation-pipeline-2025) for benchmarking.

---

## 📁 Codebase Overview

The codebase is organized modularly to separate experiment configuration and core training mechanics. The custom logic driving the framework is delineated below:

```text
stacking-knowledge/
├── conf/                           # Hydra configs (models, tokenizers, curricula, experiments)
├── debug_results/                  # Results used during debugging instead of executing eval_pipeline
├── docs/                           # Further documentation
├── eval_pipeline/                  # Submodule containing the BabyLM 2025 evaluation pipeline
├── slurm_scripts/                  # Pre-configured scripts for HPC cluster submission used during the experiments
├── src/                            # Core Python source code
│   ├── continual_pretraining/      # LR rewarming callbacks and Infinite LR scheduler
│   ├── data_curriculum/            # Curriculum pacing, dynamic triggers and difficulty scorers
│   ├── gradual_stacking/           # MIDAS model growth logic and callback
│   ├── helper/                     # Data loading, WandB logging and visualization utilities
│   ├── models/                     # Model architecture definition
│   ├── tokenizer/                  # Custom BPE tokenizer training scripts
│   ├── config.py                   # Defines the hierarchical configuration schema
│   ├── custom_trainer.py           # Custom HF Trainer, subclasses HF Trainer to inject curriculum sampling logic
│   ├── dataloader.py               # Implements a PyTorch dataloader to support curriculum learning
│   ├── evaluator.py                # Orchestrates the execution of benchmarking scripts
│   └── tokenizer.py                # Handles the initialization and loading of the tokenizer
├── load_resources_on_cluster.py    # Python script for pre-loading the dataset & model on HPC
├── load_resources_on_cluster.sh    # Shell script for pre-loading the dataset & model on HPC
├── requirements.txt                # Packages / dependencies required for this project
├── run_train.sh                    # Example script to schedule a slurm job for running an experiment
├── setup_environment.py            # Setup script
└── train.py                        # Main entry point for training
```

Further methodological information is provided in the [Methodology](docs/Methodology.md) documentation.

## 🚀 How to Use

**1. Environment Setup:** 

Set up the repository and required dependencies on your local machine or HPC cluster. Detailed instructions are provided in the [Environment Setup Guide](docs/Setup_Environment.md).

**2. Experiment Configuration:** 

Define your desired experimental setup by modifying [`config.yaml`](conf/config.yaml). For a comprehensive breakdown of the available parameters, consult the [Configuration Documentation](docs/Configuration.md).

**3. Model Training:** 

Execute the training pipeline locally by running [`train.py`](train.py) or submit a scheduled job to an HPC cluster. Further details on cluster access and job scheduling are available in the [Cluster Connection Guide](docs/Connect_to_Cluster.md).


