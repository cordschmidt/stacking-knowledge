"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from omegaconf import MISSING, DictConfig


@dataclass
class ExperimentParams(DictConfig):
    seed: int

    # Name of the experiment - needs to be set at runtime
    name: str = MISSING

    # Name of the group that the current experiment belongs to, analogous to 'project' in wandb
    group: str = MISSING

    # whether to run a minimal version of the experiment
    dry_run: bool = False

    # whether to skip running the evaluation scripts and instead use some
    # dummy metrics for debugging
    skip_execution_of_eval_scripts_for_debugging: bool = False

    # whether to run the experiment only offline
    offline_run: bool = False

    # whether to store wandb logs locally (e.g. when GPU nodes don't have internet connection)
    wandb_log_locally: bool = False

    # Optional checkpoint path to resume training from
    resume_checkpoint_path: Optional[str] = None

    # If resume_checkpoint_path is not None and we are logging to wandb,
    # we need to specify the run_id of the run we are resuming from
    resume_run_id: Optional[str] = None


@dataclass
class DatasetParams(DictConfig):
    # name of the dataset on huggingface
    name: str
    # subconfig i.e. strict-small
    subconfig: str


@dataclass
class TokenizerParams(DictConfig):
    # data processing parameters
    name: str

    # additional optional kwargs
    add_prefix_space: Optional[bool] = None


@dataclass
class DataPreprocessingParams(DictConfig):
    # params for preprocessing the dataset (i.e. tokenization)
    include_punctuation: bool
    join_sentences: bool
    max_input_length: int
    callback_functions: Optional[List[str]] = None


@dataclass
class ModelParams(DictConfig):
    # model parameters
    name: str

    # NOTE: At least 'hidden_size' needs to be specified
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerParams(DictConfig):
    batch_size: int
    lr: float
    num_warmup_steps: int
    max_training_steps: int
    eval_blimp: bool
    eval_glue: bool
    eval_msgs: bool
    eval_perplexity: bool
    max_flops: Optional[float] = None
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None


### Curriculum learning parameter: can be either objective or data-driven ###


## Data-driven curriculum learning parameters ##
@dataclass
class PacingFunctionParams(Mapping[str, Any]):
    # Num of steps to take (in percent) before beginning the curriculum
    start_percent: float
    # Num of steps to take (in percent) before ending the curriculum
    end_percent: float
    # Difficulty (percentile of the data) to start at
    starting_difficulty: float
    # Max difficulty (percentile of the data) to end at; 1.0 means include all data at the
    # end of the curriculum
    max_difficulty: Optional[float] = 1.0
    # Optional parameters required prop-alpha aligned pacing
    k_number_of_stages: Optional[int] = None
    alpha: Optional[float] = None


# Difficulty Scorer Parameters

DifficultyScorerKwargsType = Optional[Dict[str, Any]]

@dataclass
class DataCurriculumParams(DictConfig):
    # data-driven curriculum learning parameters

    # the column of the data to sort by (aka n_gram perplexity, sentence length, etc.)
    difficulty_scorer_name: str

    difficulty_scorer_kwargs: DifficultyScorerKwargsType

    # one of ['linear', 'quad', 'root', 'step', 'exp', 'log'] or None, meaning no pacing
    pacing_fn_name: str

    pacing_fn_kwargs: PacingFunctionParams

# Gradual Stacking Parameters

@dataclass
class GradualStackingParams(DictConfig):
    enabled: bool = False  # Whether to apply gradual stacking
    k_number_of_stages: Optional[int] = 4 # Number of stages for model growing, the model will be grown k-1 times. The final number of layers in the model will be k * layer_per_block (the final model consists of k blocks)
    alpha: Optional[float] = 1.0 # Factor for determining the spacing between the growing stages, greater alpha leads to more training budget spend on later stages
    layer_per_block: Optional[int] = 1 # Number of layers that are considered as one block - per stage, the middle block of layers will be duplicated
    number_non_embedding_params_compute_equivalent_model: Optional[int] = None # Number of parameters of a (constant-sized) model, that the gradual stacking model should be compared to. Based on the size differences, steps will be adjusted in order to app. match compute equivalence (number of steps in config must be the same)

@dataclass
class ContinualPretrainingParams(DictConfig):
    enable_lr_reset: bool = False
    max_rewarm_lr: Optional[float] = None  # Maximal LR for all stages after the first
    rewarm_steps: Optional[int] = None # Number of warmup-steps per stage (not recommended)
    rewarm_fraction: Optional[float] = None # Fraction of steps in a stage that should be used for linear warmup of the learning rate in each stage (actual steps of teh stages differ based on the total number of steps in each stage)
    data_replay_mode: Optional[str] = None # None, previous_stage_only, all_previous_stages
    data_replay_fraction: Optional[float] = 0.0 # Fraction, how many % of samples from the previous stage(s) should be included when training the current stage
    data_replay_decay: Optional[float] = 1.0 # Control how fast old stages decay within the data replay

### Container for entire config ###

@dataclass
class BabyLMConfig(DictConfig):
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    trainer: TrainerParams
    gradual_stacking: GradualStackingParams
    continual_pretraining: ContinualPretrainingParams
    data_curriculum: Optional[DataCurriculumParams] = None
