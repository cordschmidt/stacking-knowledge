import logging

from sympy.solvers.diophantine.diophantine import equivalent
from torch.fx.experimental.migrate_gradual_types.constraint_generator import embedding_inference_rule

from src.config import BabyLMConfig
from src.data_curriculum.difficulty_scorer.stages import NUM_STAGES
from omegaconf import OmegaConf

from src.gradual_stacking.scheduler import PropAlphaScheduler

DRY_RUN_TRAIN_STEPS = 100
DRY_RUN_WARMUP_STEPS = 10
DIFFICULTY_SCORER_UPDATE = 75

# Logger for this file
logger = logging.getLogger(__name__)

def validate_and_adjust_config(cfg: BabyLMConfig):
    # Additional validations that cannot be done in hydra directly easily
    do_additional_config_validations(cfg=cfg)
    adjust_parameters_in_config_for_special_setups(cfg=cfg)

    # Log configuration
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

def do_additional_config_validations(cfg: BabyLMConfig):
    validate_prop_alpha_aligned_data_curriculum_pacing(cfg=cfg)
    validate_staged_proportion_mode(cfg=cfg)
    if cfg.continual_pretraining.enable_lr_reset:
        validate_staged_data_curriculum_is_enabled_for_continual_pretraining(cfg=cfg)
        validate_either_rewarm_steps_or_fraction_is_set(cfg=cfg)
    if cfg.continual_pretraining.data_replay_mode is not None:
        validate_staged_data_curriculum_is_enabled_for_continual_pretraining(cfg=cfg)
        validate_data_replay_mode(cfg=cfg)
        validate_data_replay_fraction_is_valid(cfg=cfg)
        validate_data_replay_decay(cfg=cfg)

def validate_prop_alpha_aligned_data_curriculum_pacing(cfg: BabyLMConfig):
    if cfg.data_curriculum and \
       cfg.data_curriculum.difficulty_scorer_name == "staged_data_split" and \
       cfg.data_curriculum.pacing_fn_name == "prop_alpha":

        validate_that_number_of_prop_alpha_stages_equal_number_of_datasets(cfg=cfg)

def validate_that_number_of_prop_alpha_stages_equal_number_of_datasets(cfg: BabyLMConfig):
    # Check stage count alignment
    if cfg.gradual_stacking.k_number_of_stages != NUM_STAGES:
        raise ValueError(
            f"Alignment Error: 'prop_alpha' + 'staged_data_split' requires "
            f"gradual_stacking.k_number_of_stages ({cfg.gradual_stacking.k_number_of_stages}) "
            f"to equal the data's NUM_STAGES ({NUM_STAGES})."
        )

def validate_staged_proportion_mode(cfg: BabyLMConfig):
    if cfg.data_curriculum and cfg.data_curriculum.difficulty_scorer_name == "staged_data_split":
        scorer_kwargs = cfg.data_curriculum.difficulty_scorer_kwargs or {}
        mode = scorer_kwargs.get("proportion_mode")
        valid_modes = [None, "sample", "token"]
        if mode not in valid_modes:
             raise ValueError(
                 f"Invalid 'proportion_mode': {mode}. "
                 f"Must be one of {valid_modes} in difficulty_scorer_kwargs."
             )

def validate_staged_data_curriculum_is_enabled_for_continual_pretraining(cfg: BabyLMConfig):
    if cfg.data_curriculum is None:
        raise ValueError(
            f"Configuration Error: 'lr_reset' for continual pretraining is set to True, but can only be used "
            f"when data_curriculum is active"
        )
    if cfg.data_curriculum.difficulty_scorer_name != "staged_data_split":
        raise ValueError(
            f"Configuration Error: 'lr_reset' for continual pretraining is set to True, but this can only be used "
            f"with the 'staged_data_split' data curriculum scorer. Current scorer: "
            f"'{cfg.data_curriculum.difficulty_scorer_name}'"
        )

def validate_either_rewarm_steps_or_fraction_is_set(cfg: BabyLMConfig):
    rewarm_steps = cfg.continual_pretraining.rewarm_steps
    rewarm_fraction = cfg.continual_pretraining.rewarm_fraction

    # Check that either rewarm steps or rewarm fraction is set
    if rewarm_steps is not None and rewarm_fraction is not None:
        raise ValueError(
            "Configuration Error: Provide either 'rewarm_steps' or 'rewarm_fraction', not both."
        )

def validate_data_replay_mode(cfg: BabyLMConfig):
    mode = cfg.continual_pretraining.data_replay_mode
    valid_modes = [None, "previous_stage_only", "all_previous_stages"]
    if mode not in valid_modes:
        raise ValueError(
            f"Invalid 'data_replay_mode': {mode}"
            f"Must be one of {valid_modes}"
        )

def validate_data_replay_fraction_is_valid(cfg):
    data_replay_fraction = cfg.continual_pretraining.data_replay_fraction
    data_replay_mode = cfg.continual_pretraining.data_replay_mode

    if data_replay_fraction is None:
        raise ValueError(
            f"Configuration Error: 'data_replay_fraction' has to be set, when 'data_replay_mode' is '{data_replay_mode}'"
        )
    if data_replay_fraction <= 0.0 or data_replay_fraction >= 1.0:
        raise ValueError(
            f"Configuration Error: 'data_replay_fraction' has to be between 0.0 and 1.0, but got {data_replay_fraction} instead"
        )

def validate_data_replay_decay(cfg):
    data_replay_fraction = cfg.continual_pretraining.data_replay_decay
    if data_replay_fraction <= 0.0 or data_replay_fraction > 1.0:
        raise ValueError(
            f"Configuration Error: 'data_replay_fraction' has to be > 0.0 and <= 1.0, but got {data_replay_fraction} instead"
        )

def adjust_parameters_in_config_for_special_setups(cfg: BabyLMConfig):
    # Adjust training parameters in dry run for faster testing & debugging
    if cfg.experiment.dry_run:
        adjust_params_for_dry_run(cfg=cfg)

    if cfg.data_curriculum.pacing_fn_name == "prop_alpha":
        insert_gradual_stacking_parameters_into_pacing_fn(cfg=cfg)

    insert_data_replay_parameters_into_staged_data_split_scorer(cfg=cfg)

    if cfg.data_curriculum and \
            cfg.data_curriculum.difficulty_scorer_name == "staged_data_split" and \
            cfg.data_curriculum.pacing_fn_name == "prop_alpha":
        force_ignoring_dataset_sizes_in_staged_data_curriculum(cfg=cfg)

def adjust_params_for_dry_run(cfg: BabyLMConfig):
    logger.info(
        "Running in dry run mode -- overriding config with values: "
    )
    logger.info(f"\t max_training_steps: {DRY_RUN_TRAIN_STEPS}")
    logger.info(f"\t num_warmup_steps: {DRY_RUN_WARMUP_STEPS}")
    cfg.trainer.max_training_steps = DRY_RUN_TRAIN_STEPS
    cfg.trainer.num_warmup_steps = DRY_RUN_WARMUP_STEPS

    if (
            cfg.data_curriculum is not None
            and cfg.data_curriculum.difficulty_scorer_kwargs is not None
    ):

        if (
                cfg.data_curriculum.difficulty_scorer_kwargs.get("update")
                is not None
        ):
            cfg.data_curriculum.difficulty_scorer_kwargs["update"] = (
                DIFFICULTY_SCORER_UPDATE
            )
            logger.info(
                f"\t data curriculum difficulty scorer update: {DIFFICULTY_SCORER_UPDATE}"
            )

def insert_gradual_stacking_parameters_into_pacing_fn(cfg: BabyLMConfig):
    k_stages_from_gradual_stacking_config = cfg.gradual_stacking.k_number_of_stages
    alpha_from_gradual_stacking_config = cfg.gradual_stacking.alpha
    cfg.data_curriculum.pacing_fn_kwargs["k_number_of_stages"] = k_stages_from_gradual_stacking_config
    cfg.data_curriculum.pacing_fn_kwargs["alpha"] = alpha_from_gradual_stacking_config

def insert_data_replay_parameters_into_staged_data_split_scorer(cfg: BabyLMConfig):
    if cfg.data_curriculum and cfg.data_curriculum.difficulty_scorer_name == "staged_data_split":
        params = dict(cfg.data_curriculum.difficulty_scorer_kwargs)
        params["data_replay_mode"] = cfg.continual_pretraining.data_replay_mode
        params["data_replay_fraction"] = cfg.continual_pretraining.data_replay_fraction
        params["data_replay_decay"] = cfg.continual_pretraining.data_replay_decay
        cfg.data_curriculum.difficulty_scorer_kwargs = params

def force_ignoring_dataset_sizes_in_staged_data_curriculum(cfg: BabyLMConfig):
    # Check proportion flag
    scorer_kwargs = cfg.data_curriculum.difficulty_scorer_kwargs or {}
    if scorer_kwargs.get("account_for_dataset_proportions") is not False:
        logger.info(f"In cfg.data_curriculum.difficulty_scorer_kwargs the attribute 'account_for_dataset_proportions' was set to True")
        logger.info(f"In order to align the data curriculum with the prop-alpha stages, 'account_for_dataset_proportions' has to be set to False")
        cfg.data_curriculum.difficulty_scorer_kwargs["account_for_dataset_proportions"] = False
        logger.info(f"'account_for_dataset_proportions' was set to {cfg.data_curriculum.difficulty_scorer_kwargs["account_for_dataset_proportions"]}")

def consider_step_adjustment_for_compute_equivalent_model_training(cfg: BabyLMConfig, model):
    if cfg.gradual_stacking.enabled and cfg.gradual_stacking.number_non_embedding_params_compute_equivalent_model is not None:
        adjust_steps_based_on_params_of_compute_equivalent_model(cfg=cfg, model=model)

def adjust_steps_based_on_params_of_compute_equivalent_model(cfg: BabyLMConfig, model):
    scheduler = PropAlphaScheduler(
        total_training_steps=cfg.trainer.max_training_steps,  # placeholder
        k_number_of_stages=cfg.gradual_stacking.k_number_of_stages,
        alpha=cfg.gradual_stacking.alpha
    )
    # Estimate these from your model config
    number_of_static_non_embedding_params, number_of_params_per_block = estimate_parameter_counts(model=model, layer_per_block=cfg.gradual_stacking.layer_per_block)

    new_max_steps = scheduler.get_compute_equivalent_steps(
        baseline_steps=cfg.trainer.max_training_steps,
        baseline_params=cfg.gradual_stacking.number_non_embedding_params_compute_equivalent_model,
        number_of_static_non_embedding_params=number_of_static_non_embedding_params,
        number_of_params_per_block=number_of_params_per_block
    )
    logger.info(f"Adjusting max_steps from {cfg.trainer.max_training_steps} to {new_max_steps} for app. compute equivalence.")
    cfg.trainer.max_training_steps = new_max_steps


def estimate_parameter_counts(model, layer_per_block):
    # Get all static params
    head_params = sum(p.numel() for p in model.lm_head.parameters())
    final_norm_params = sum(p.numel() for p in model.model.norm.parameters())
    number_of_static_non_embedding_params = head_params + final_norm_params

    # Calculate parameters for a single layer, e.g. based on first layer
    number_of_params_single_layer = sum(p.numel() for p in model.model.layers[0].parameters())

    # Calculate number of parameters for one block, which gets duplicated in every stage
    number_of_params_per_block = number_of_params_single_layer * layer_per_block

    return number_of_static_non_embedding_params, number_of_params_per_block
