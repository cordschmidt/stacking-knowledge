""" Class for calling the evaluation pipeline on a model """

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Union

# typing imports
import torch
import torch.distributed as dist

from src.helper.setup_environment import TORCH_RUN_ENV_KEYS

logger = logging.getLogger(__name__)

DUMMY_DATA_DIR = "debug_results"


def parse_best_temperature_report_file_results(report_path: Path):
    """
    Gets the accuracy values for a given best_temperature_report.txt file
    """
    results = {}
    with report_path.open() as f:
        lines = [line.strip() for line in f if line.strip()]
    for line in lines:
        if line.startswith("###") or line.startswith("TEMPERATURE"):
            continue
        elif ":" in line:
            accuracy_identifier, accuracy_value = line.split(":")
            accuracy_identifier = accuracy_identifier.strip()
            accuracy_value = float(accuracy_value.strip())
            results[accuracy_identifier] = accuracy_value
        else:
            # could be average accuracy
            try:
                results["average_accuracy"] = float(line)
            except ValueError:
                pass
    return results

def parse_correlations_file_results(corr_path: Path):
    """
    Gets the correlation values for a given correlations.txt file
    """
    results = {}
    with corr_path.open() as f:
        for line in f:
            if line.strip():
                accuracy_identifier, accuracy_name = line.split()
                results[accuracy_identifier] = float(accuracy_name)
    return results

def gather_results_from_eval_pipeline(eval_output_dir: Path):
    """
    Collects all results from the evaluation output directory, stored in best_temperature_report.txt or correlations.txt files
    """
    accuracies = {}
    for task_dir in eval_output_dir.rglob("*"):
        if not task_dir.is_dir():
            continue
        benchmark_name = task_dir.name
        report_file = task_dir / "best_temperature_report.txt"
        corr_file = task_dir / "correlations.txt"
        if report_file.exists():
            task_results = parse_best_temperature_report_file_results(report_file)
            for accuracy_identifier, accuracy_value in task_results.items():
                accuracies[f"{benchmark_name}.{accuracy_identifier}"] = accuracy_value
        elif corr_file.exists():
            task_results = parse_correlations_file_results(corr_file)
            for accuracy_identifier, accuracy_value in task_results.items():
                accuracies[f"{benchmark_name}.{accuracy_identifier}"] = accuracy_value
    return accuracies

class ZeroShotEvaluator(object):
    def __init__(
        self,
        out_dir: str,
        device: torch.device,
        process_index: int,
        world_size: int,
        dry_run: bool = False,
        is_best_run: bool = False,
        use_dummy_eval_data: bool = False,
        do_fast_eval: bool = False,
        experiment_name: str = None,
        global_steps: int = None,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
            * device (torch.device): Device to run the evaluation on
            * process_index (int): Index of the current process
            * world_size (int): Number of processes
            * dry_run (bool): If True, don't actually run the evaluation script
            * is_best_run (bool): If True, keep the predictions from BLIMP
        """

        self.out_dir = out_dir
        self.device = device
        self.process_index = process_index
        self.world_size = world_size
        self.dry_run = dry_run
        self.is_best_run = is_best_run
        self.use_dummy_eval_data = use_dummy_eval_data
        self.do_fast_eval = do_fast_eval
        self.experiment_name = experiment_name
        self.global_steps = global_steps

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the BLIMP evaluation pipeline.

        NOTE: If we are using DDP, this function will run on all the processes.
        """

        logger.info("Running BLIMP and AOA evaluation script...")
        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        # Prepare command to run evaluation script

        if self.is_best_run:
            checkpoint_name = "checkpoint_best"
        else:
            checkpoint_name = f"checkpoint_{self.global_steps}"

        # Make sure to pass an absolute path
        model_path_absolute = Path(self.out_dir).resolve()
        cmd = (
            f"cd eval_pipeline && "
            f"./eval_zero_shot_fast.sh {model_path_absolute} "
            f"{checkpoint_name} causal"
        )
        # If using dummy eval data, don't run script
        if self.use_dummy_eval_data:
            logger.info(
                f"Local debugging: Evaluation script won't run, dummy data will be used for BLIMP / AOA"
            )
        else:
            subprocess.run(cmd, shell=True)

        if self.world_size > 1:
            dist.barrier() # Synchronize DDP processes

        logger.info(
            "BLIMP and AOA Evaluation script finished. Getting accuracies..."
        )

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        accuracies = {}

        # Use dummy path or real output path depending on debug mode
        if self.use_dummy_eval_data:
            eval_output_dir = Path(DUMMY_DATA_DIR)
            logger.info(
                f"Local debugging: {DUMMY_DATA_DIR} will be used to get the dummy results for BLIMP / AOA"
            )
        else:
            # Go to result directory
            results_dir = Path("eval_pipeline/results")
            eval_output_dir = (
                    results_dir
                    / self.experiment_name
                    / checkpoint_name
                    / "zero_shot"
                    / "causal"
            )

        # Get all accuracies from the evaluation results of the script
        accuracies = gather_results_from_eval_pipeline(eval_output_dir)

        if self.world_size > 1:
            # Make sure all processes have finished before removing zeroshot directory
            dist.barrier()

        # Clean up prediction folder within eval pipeline if not in debug mode and ensure this is done for only one process
        if self.process_index == 0:
            if self.use_dummy_eval_data:
                logger.info(
                    f"Local debugging: No evaluation results files will be removed"
                )
            else:
                self.move_eval_results_to_project_root(eval_output_dir)

        return accuracies

    def move_eval_results_to_project_root(self, eval_output_dir):
        """
        Move all evaluation results from eval_output_dir into a new
        results/ folder at the project root, preserving the full hierarchy
        under experiment/checkpoint/zero_shot/causal.

        The project root is inferred from eval_output_dir as the parent of eval_pipeline.
        """
        eval_output_dir = Path(eval_output_dir).resolve()

        # Check if eval directory exists
        if not eval_output_dir.exists():
            raise FileNotFoundError(f"Evaluation results directory does not exist: {eval_output_dir}")

        # Infer project root (parent of eval_pipeline)
        try:
            # Find 'eval_pipeline' in the path
            project_root_index = eval_output_dir.parts.index("eval_pipeline")
        except ValueError:
            raise ValueError(f"'eval_pipeline' not found in path: {eval_output_dir}")

        project_root = Path(*eval_output_dir.parts[:project_root_index])

        # Destination directory at project root, preserving hierarchy after eval_pipeline/results
        # Get relative path after 'eval_pipeline/results'
        try:
            results_index = eval_output_dir.parts.index("results", project_root_index)
        except ValueError:
            raise ValueError(f"'results' not found in path after 'eval_pipeline': {eval_output_dir}")

        relative_path = Path(*eval_output_dir.parts[results_index + 1:])
        new_results_dir = project_root / "results" / relative_path
        new_results_dir.parent.mkdir(parents=True, exist_ok=True)

        # Move the directory
        shutil.move(str(eval_output_dir), str(new_results_dir))

        # Cleanup: remove empty parent dirs under eval_pipeline/results
        parent = eval_output_dir.parent
        while parent != project_root / "eval_pipeline":
            try:
                parent.rmdir()
            except OSError:
                break  # stop if directory not empty
            parent = parent.parent


class FinetuneEvaluator(object):
    # Define supported GLUE and MSGS tasks
    GLUE_TASKS = [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "boolq",
        "multirc",
        "wsc",
    ]

    MSGS_TASKS = [
        "main_verb_control",
        "control_raising_control",
        "syntactic_category_control",
        "lexical_content_the_control",
        "relative_position_control",
        "main_verb_lexical_content_the",
        "main_verb_relative_token_position",
        "syntactic_category_lexical_content_the",
        "syntactic_category_relative_position",
        "control_raising_lexical_content_the",
        "control_raising_relative_token_position",
    ]

    def __init__(
        self,
        out_dir: str,
        device: torch.device,
        process_index: int,
        world_size: int,
        dry_run: bool = False,
        run_glue: bool = True,
        run_msgs: bool = False,
        keep_predictions: bool = False,
        use_dummy_eval_data: bool = False,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
            * device (torch.device): Device to run the evaluation on
            * process_index (int): Index of the current process
            * world_size (int): Number of processes
            * dry_run (bool): If True, don't actually run the evaluation script
            * run_glue (bool): If True, finetune on all GLUE tasks
            * run_msgs (bool): If True, finetune on all MSGS tasks
            * keep_predictions (bool): If True, keep the predictions from the finetuning
        """

        # Raise error if both evaluation modes are disabled
        if not run_glue and not run_msgs:
            raise ValueError(
                "run_glue and run_msgs cannot both be False. Must run at least one of GLUE or MSGS tasks"
            )

        # Save constructor parameters
        self.out_dir = out_dir
        self.device = device
        self.process_index = process_index
        self.world_size = world_size
        self.dry_run = dry_run
        self.run_glue = run_glue
        self.run_msgs = run_msgs
        self.keep_predictions = keep_predictions
        self.use_dummy_eval_data = use_dummy_eval_data

    def run_script(self, task: str):

        logger.info(f"Running finetuning script for {task}...")

        # Handle MNLI task special cases
        if task == "mnli":
            valid_name = "validation_matched"
            out_dir = "mnli"
        elif task == "mnli-mm":
            valid_name = "validation_mismatched"
            task = "mnli"
            out_dir = "mnli-mm"
        else:
            valid_name = "validation"
            out_dir = task

        # Ensure output directory exists
        os.makedirs(
            os.path.join(self.out_dir, "finetune", out_dir), exist_ok=True
        )

        task_group = "glue" if task in self.GLUE_TASKS else "msgs"

        # Construct training command
        cmd = (
            "cd lib/evaluation-pipeline; ../../env/bin/python finetune_classification.py"
            + f" --model_name_or_path ../../{self.out_dir}"
            + f" --output_dir ../../{self.out_dir}/finetune/{out_dir}"
            + f" --train_file filter-data/{task_group}_filtered/{task}.train.json"
            + f" --validation_file filter-data/{task_group}_filtered/{task}.{valid_name}.json"
            + f" --do_train"
            + f" --do_eval"
            + f" --do_predict"
            + f" --use_fast_tokenizer True"  # Set to True to use fast tokenizer
            + f" --max_seq_length 128"
            + f" --per_device_train_batch_size 64"
            + f" --learning_rate 5e-5"
            + f" --num_train_epochs 10"
            + f" --evaluation_strategy steps"
            + f" --patience 10"
            + f" --eval_every 200"
            + f" --eval_steps 200"
            + f" --overwrite_output_dir"
            + f" --seed 12"
            # + f" --logging_steps 1" NOTE: ENABLE THIS FOR DEBUGGING
        )

        # print all the key names of the envrioment variables

        subprocess_env = os.environ.copy()
        # remove from subprocess_env all torch_run related variables
        for key in list(subprocess_env.keys()):
            if key in TORCH_RUN_ENV_KEYS:
                del subprocess_env[key]

        # If in DDP, set CUDA_VISIBLE_DEVICES to current GPU
        if self.world_size > 1:
            # Set CUDA_VISIBLE_DEVICES to the local process index (assuming 4 GPUs per node)
            subprocess_env["CUDA_VISIBLE_DEVICES"] = str(
                self.process_index % 4
            )

        # Disable W&B on subprocess
        # NOTE: COMMENT OUT FOR DEBUGGING
        subprocess_env["WANDB_DISABLED"] = "true"
        subprocess_env["WANDB_MODE"] = "disabled"


        logging.info(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, env=subprocess_env)
        logging.info(f"Finished finetuning {task}.")

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the GLUE evaluation pipeline.
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running Finetuning evaluation script...")

        # Select tasks depending on mode and dry run
        tasks = []
        if self.run_glue:
            if self.dry_run:
                tasks.extend(["cola"])
                logger.info("Running dry run. Only running on CoLA from GLUE.")
            else:
                tasks.extend(self.GLUE_TASKS)
                logger.info(
                    "Running on all GLUE tasks: " + ", ".join(self.GLUE_TASKS)
                )
        if self.run_msgs:
            if self.dry_run:
                tasks.extend(["main_verb_control"])
                logger.info(
                    "Running dry run. Only running on main_verb_control from MSGS."
                )
            else:
                tasks.extend(self.MSGS_TASKS)
                logger.info(
                    "Running on all MSGS tasks: " + ", ".join(self.MSGS_TASKS)
                )

        # Distribute task execution across processes and only execute script when local debug mode is not activated
        for task_idx, task in enumerate(tasks):
            if task_idx % self.world_size != self.process_index:
                continue
            if self.use_dummy_eval_data:
                logger.info(
                    f"Local debugging: Evaluation script for finetune-task {task} won't run"
                )
            else:
                self.run_script(task)

        # Synchronize DDP processes
        if self.world_size > 1:
            dist.barrier()

        logger.info(
            "Finetuning Evaluation script finished. Getting accuracies..."
        )
        accuracies = {}

        # Select dummy data dir for local debugging
        if self.use_dummy_eval_data:
            eval_data_dir = DUMMY_DATA_PATH
            logger.info(
                f"Local debugging: {DUMMY_DATA_PATH} will be used to get the dummy results for GLUE / MSGS"
            )
        else:
            eval_data_dir = self.out_dir


        # Search for tasks to get accuracies from
        tasks = os.listdir(os.path.join(eval_data_dir, "finetune"))
        # Ignore '.DS_Store' files on macOS for local debugging
        tasks = [task for task in tasks if task != ".DS_Store"]

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        for task in tasks:
            path = os.path.join(
                eval_data_dir, "finetune", task, "eval_results.json"
            )
            with open(path) as f:
                data = json.load(f)
                task_group = "glue" if task in self.GLUE_TASKS else "msgs"
                accuracies[f"{task_group}_" + task + "_accuracy"] = data[
                    "eval_accuracy"
                ]
                if "eval_f1" in data:
                    accuracies[f"{task_group}_" + task + "_f1"] = data[
                        "eval_f1"
                    ]

        # Make sure all processes have finished before removing the directory
        if self.world_size > 1:
            dist.barrier()

        # Delete the finetune directory within one process if not in debug mode
        if self.process_index == 0 and not self.keep_predictions:
            if self.use_dummy_eval_data:
                logger.info(
                    f"Local debugging: No evaluation results files will be removed"
                )
            else:
                shutil.rmtree(os.path.join(self.out_dir, "finetune"))

        return accuracies


def collect_results(out_dir: str):
    """Attempts to run the the collect_results.py script from the evaluation pipeline"""

    cmd = (
        "cd lib/evaluation-pipeline; ../../env/bin/python collect_results.py"
        + f" ../../{out_dir}"
    )

    output = subprocess.run(
        cmd, shell=True, capture_output=True, env=os.environ.copy()
    )
    if output.returncode != 0:
        logger.warning("Failed to run collect_results.py script. Skipping...")
    return
