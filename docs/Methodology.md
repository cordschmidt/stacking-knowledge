# Methodology

## Data Curriculum

> This architecture builds heavily upon the work of Martinez et al. (2023) in [CLIMB – Curriculum Learning for Infant-inspired Model Building](https://aclanthology.org/2023.conll-babylm.10/). 
> 
> While the CLIMB codebase served as the underlying foundation for this repository, the strict discrete-stage data isolation and the dynamic (overfitting-triggered) data curriculum strategies were custom-built and added directly on top of their original framework.

The approach mainly used in this repository utilizes a strict, distinct-stage data curriculum to sequentially present isolated corpora (e.g., transitioning from child-directed speech to complex grammatical text). Rather than relying entirely on static step limits, we implement a dynamic curriculum that actively triggers data transitions when the model plateaus. 

This is implemented across three core components:

**Staged Data Split Scorer:** 

Located in `src/data_curriculum/difficulty_scorer/staged_data_split.py`. The `StagedDataSplitSorter` enforces strict dataset isolation by filtering out all samples that fall outside the currently active curriculum stage.

**Dynamic Pacing Trigger:** 

Implemented in `src/data_curriculum/dynamic_curriculum_callback.py`. The `DynamicCurriculumCallback` periodically evaluates perplexity on the dev-set. If the model fails to improve over a set number of evaluation steps, the callback forces an early data transition via `scorer.force_next_stage()` to prevent overfitting.

**Synchronized Pacing Function:** 

Implemented in `src/data_curriculum/pacing_fn.py`. To ensure the data stages align with the model's architectural growth, a `prop_alpha` pacing function calculates a continuous polynomial curve (via Piecewise Cubic Hermite Interpolation) that synchronizes dataset difficulty with specific training steps.

## MIDAS - Gradual Stacking

> The model growth mechanics are based on the MIDAS algorithm introduced by Saunshi et al. (2024) in [On the Inductive Bias of Stacking Towards Improving Reasoning](https://arxiv.org/abs/2409.19044). 
> 
> As no public code was released alongside the paper, the MIDAS algorithm and its corresponding prop-alpha scheduling were implemented independently from scratch within this repository.

With the aim to optimize training efficiency and accelerate convergence, the MIDAS gradual stacking algorithm was implemented. This framework dynamically increases model depth mid-training by duplicating the middle block of the network at predetermined step boundaries.

**Growth Scheduling:** 

Found in `src/gradual_stacking/scheduler.py`. The `PropAlphaScheduler` pre-calculates exact global step boundaries for model growth using a $T_i \propto i^\alpha$ schedule, ensuring deterministic synchronization with the data curriculum.

**Layer Duplication & Optimizer Reset:** 

Handled by the `GradualStackingCallback` in `src/gradual_stacking/stacking_callback.py`. At scheduled boundaries, it pauses the training loop, duplicates the $\lceil n/2 \rceil$-th block of layers, and inserts them back into the model. Crucially, it explicitly registers these new parameters into the optimizer with a zeroed momentum state.

## Continual Pre-training

> The continual pre-training interventions implemented in this repository synthesize findings from several recent papers:
> 
> **LR Rewarming & Data Replay:** Independently implemented from scratch based on the theoretical frameworks provided by Gupta et al. (2023), [Continual Pre-Training of Large Language Models: How to (re)warm your model?](https://arxiv.org/abs/2308.04014) and Ibrahim et al. (2024), [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763).
> 
> **Infinite LR Schedule:** The infinite learning rate scheduler was designed in accordance to Singh et al. (2025), [Beyond Cosine Decay: On the effectiveness of Infinite Learning Rate Schedule for Continual Pre-training](https://arxiv.org/abs/2503.02844).

Data stage transitions and architectural shifts introduce the risk of catastrophic forgetting and gradient instability. To counteract this, we deploy targeted Continual Pre-Training (CPT) interventions tailored for dynamic training environments.

**Learning Rate Rewarming:** 

Implemented in `src/continual_pretraining/learning_rate_reset_callback.py`. The `LearningRateResetCallback` intercepts the optimizer at every stage transition to inject a fresh, fractional warmup phase. It dynamically scales the peak learning rate of successive stages to smoothly approach a target `max_rewarm_lr`.

**Infinite LR Scheduler:** 

Located in `src/continual_pretraining/infinite_lr_scheduler.py`. Because our dynamic curriculum creates unpredictable stage lengths, the `InfiniteLRScheduler` avoids rigid epoch decays. It employs a 4-phase schedule that settles into a constant learning rate, only forcing a final exponential decay when notified that the final data stage has been triggered.

**Token-Aware Data Replay:** 

Integrated directly into the difficulty scorer in `src/data_curriculum/difficulty_scorer/staged_data_split.py`. When historical replay is enabled, it dynamically recalculates batch sampling weights. It balances sampling probabilities using absolute corpus token sizes and applies an exponential decay factor (`data_replay_decay`) to penalize older stages in favor of more recent historical data.