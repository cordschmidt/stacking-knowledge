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
    enabled: bool = False  # whether to apply gradual stacking
    grow_every_n_steps: int = 500  # interval for layer duplication

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
    data_curriculum: Optional[DataCurriculumParams] = None
