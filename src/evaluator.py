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
from abc import ABCMeta, abstractmethod

logger = logging.getLogger(__name__)

DUMMY_DATA_DIR = "debug_results"

class BaseEvaluator(metaclass=ABCMeta):
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
        evaluator_name: str = None,
        task_prefix_to_add: str = None,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
            * device (torch.device): Device to run the evaluation on
            * process_index (int): Index of the current process
            * world_size (int): Number of processes
            * dry_run (bool): If True, don't actually run the evaluation script
            * is_best_run (bool): If True, keep the predictions
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
        self.evaluator_name = evaluator_name
        self.task_prefix_to_add = task_prefix_to_add

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the BLIMP evaluation pipeline.

        NOTE: If we are using DDP, this function will run on all the processes.
        """

        logger.info(f"Running {self.evaluator_name} evaluation script...")

        checkpoint_name = self._determine_checkpoint_name()

        # Prepare command to run evaluation script
        cmd = self._prepare_command(checkpoint_name=checkpoint_name)

        # Start a subprocess to run the evaluation pipeline script
        self._execute_eval_script(cmd=cmd)

        self._synchronize_distributed_processes()

        logger.info(
            f"{self.evaluator_name} Evaluation script finished. Getting accuracies..."
        )

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        accuracies = {}

        eval_output_dir = self._determine_output_dir_of_eval_results(checkpoint_name=checkpoint_name)

        # Get all accuracies from the evaluation results of the script
        accuracies = self._gather_results_from_eval_pipeline(eval_output_dir)

        self._synchronize_distributed_processes()

        self._move_eval_results_to_designated_folder_and_cleanup_predictions_in_eval_pipeline(eval_output_dir=eval_output_dir, checkpoint_name=checkpoint_name)

        return accuracies

    def _determine_checkpoint_name(self):
        if self.is_best_run:
            checkpoint_name = "checkpoint_best"
        else:
            checkpoint_name = f"checkpoint_{self.global_steps}"
        return checkpoint_name

    def _prepare_command(self, checkpoint_name):
        raise NotImplementedError("Subclass must implement _prepare_command() with the specific script that has to be executed")

    def _execute_eval_script(self, cmd):
        # If using dummy eval data, don't run script
        if self.use_dummy_eval_data:
            logger.info(
                f"Local debugging: Evaluation script won't run, dummy data will be used for {self.evaluator_name}"
            )
        else:
            subprocess.run(cmd, shell=True)

    def _synchronize_distributed_processes(self):
        if self.world_size > 1:
            dist.barrier()  # Synchronize DDP processes

    def _determine_output_dir_of_eval_results(self, checkpoint_name):
        # Use dummy path or real output path depending on debug mode
        if self.use_dummy_eval_data:
            eval_output_dir = Path(DUMMY_DATA_DIR) / self.evaluator_name.lower()
            logger.info(
                f"Local debugging: {eval_output_dir} will be used to get the dummy results for {self.evaluator_name}"
            )
        else:
            # Go to result directory
            results_dir = Path("eval_pipeline/results")
            if self.dry_run:
                eval_output_dir = (
                        results_dir
                        / self.experiment_name
                        / checkpoint_name
                        / "zero_shot"
                        / "causal"
                )
            # For full evaluation we cannot provide the checkpoint / revision name,
            # instead it will always be saved in the "main" checkpoint / revision
            else:
                eval_output_dir = (
                        results_dir
                        / self.experiment_name
                        / "main"
                        / "zero_shot"
                        / "causal"
                )
        return eval_output_dir

    def _gather_results_from_eval_pipeline(self, eval_output_dir: Path):
        """
        Collects all results from the evaluation output directory, stored in best_temperature_report.txt or correlations.txt files
        """
        accuracies = {}
        for task_dir in eval_output_dir.rglob("*"):
            if not task_dir.is_dir():
                continue
            benchmark_name = task_dir.name
            # Add task prefix ig given
            if self.task_prefix_to_add is not None:
                benchmark_name = f"{self.task_prefix_to_add}_{task_dir.name}"
            else:
                benchmark_name = task_dir.name
            report_file = task_dir / "best_temperature_report.txt"
            results_file = task_dir / "results.txt"
            corr_file = task_dir / "correlations.txt"
            if report_file.exists():
                task_results = self._parse_best_temperature_report_or_results_file_results(report_file)
                for accuracy_identifier, accuracy_value in task_results.items():
                    accuracies[f"{benchmark_name}.{accuracy_identifier}"] = accuracy_value
            elif corr_file.exists():
                task_results = self._parse_correlations_file_results(corr_file)
                for accuracy_identifier, accuracy_value in task_results.items():
                    accuracies[f"{benchmark_name}.{accuracy_identifier}"] = accuracy_value
            elif results_file.exists():
                task_results = self._parse_best_temperature_report_or_results_file_results(results_file)
                for accuracy_identifier, accuracy_value in task_results.items():
                    accuracies[f"{benchmark_name}.{accuracy_identifier}"] = accuracy_value
        return accuracies

    def _parse_best_temperature_report_or_results_file_results(self, report_path: Path):
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

    def _parse_correlations_file_results(self, corr_path: Path):
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

    def _move_eval_results_to_designated_folder_and_cleanup_predictions_in_eval_pipeline(self, eval_output_dir, checkpoint_name):
        # Clean up prediction folder within eval pipeline if not in debug mode and ensure this is done for only one process
        if self.process_index == 0:
            if self.use_dummy_eval_data:
                logger.info(
                    f"Local debugging: No evaluation results files will be removed"
                )
            else:
                self.move_eval_results_to_project_root_results_dir(eval_output_dir, checkpoint_name)

    def move_eval_results_to_project_root_results_dir(self, eval_output_dir, checkpoint_name):
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

        relative_path, project_root = self._determine_relative_path_within_results_dir_and_project_root(eval_output_dir=eval_output_dir)

        new_results_dir = self._determine_new_results_dir(project_root=project_root, relative_path=relative_path, checkpoint_name=checkpoint_name)

        # Safely merge dir
        if new_results_dir.exists():
            logger.warning(
                f"Destination path '{new_results_dir}' already exists. Merging fresh results with existing ones.")
            shutil.copytree(str(eval_output_dir), str(new_results_dir), dirs_exist_ok=True)
            # Delete the source dir
            shutil.rmtree(str(eval_output_dir))
        else:
            # Move the directory if it doesn't exist yet
            shutil.move(str(eval_output_dir), str(new_results_dir))

        self._remove_empty_results_dirs_in_eval_pipeline(eval_output_dir=eval_output_dir, project_root=project_root)

    def _determine_relative_path_within_results_dir_and_project_root(self, eval_output_dir):
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
        return relative_path, project_root

    def _determine_new_results_dir(self, project_root, relative_path, checkpoint_name):
        new_output_dir = os.path.dirname(self.out_dir).replace("checkpoints", "results")
        new_results_dir = project_root / new_output_dir / relative_path
        if not self.dry_run:
            # Replace main folder when using full evaluation (when not doing dry_runs)
            new_results_dir = Path(*(checkpoint_name if p == "main" else p for p in new_results_dir.parts))
        new_results_dir.parent.mkdir(parents=True, exist_ok=True)
        return new_results_dir

    def _remove_empty_results_dirs_in_eval_pipeline(self, eval_output_dir, project_root):
        # Cleanup: remove empty parent dirs under eval_pipeline/results
        parent = eval_output_dir.parent
        while parent != project_root / "eval_pipeline":
            try:
                parent.rmdir()
            except OSError:
                break  # stop if directory not empty
            parent = parent.parent



class ZeroShotEvaluator(BaseEvaluator):

    def _prepare_command(self, checkpoint_name):
        # Make sure to pass an absolute path
        model_path_absolute = Path(self.out_dir).resolve()
        # Run fast evaluation in dry run
        if self.dry_run:
            cmd = (
                f"cd eval_pipeline && "
                f"./eval_zero_shot_fast.sh {model_path_absolute} "
                f"{checkpoint_name} causal"
            )
        else:
            cmd = (
                f"cd eval_pipeline && "
                f"./eval_zero_shot.sh "
                f"{model_path_absolute} "
                f"causal "
            )
        return cmd


class SuperGlueEvaluator(BaseEvaluator):

    def __call__(self) -> Union[Dict[str, Any], None]:
        # Run the standard evaluation pipeline completely
        accuracies = super().__call__()

        # Synchronize processes one last time to ensure all processes have finished
        self._synchronize_distributed_processes()

        # Clean up the evaluation models directory
        self._delete_eval_models_directory()

        return accuracies

    def _prepare_command(self, checkpoint_name):
        model_path_absolute = Path(self.out_dir).resolve()
        logger.info(f"Model path: {model_path_absolute}")
        cmd = (
            f"cd eval_pipeline && "
            f"./eval_finetuning.sh "
            f"{model_path_absolute}"
            # "<learning_rate (optional, default: 3e-5)>"
            # "<batch_size (optional, default: 32)> "
            # "<BIG_BSZ> "
            # "<max_epochs (optional, default: 10)> "
            # "<WSC_EPOCHS> "
            # "<seed (optional, default: 42)>"
        )
        return cmd

    def _determine_output_dir_of_eval_results(self, checkpoint_name):
        # Use dummy path or real output path depending on debug mode
        if self.use_dummy_eval_data:
            eval_output_dir = Path(DUMMY_DATA_DIR) / self.evaluator_name.lower()
            logger.info(
                f"Local debugging: {eval_output_dir} will be used to get the dummy results for {self.evaluator_name}"
            )
        else:
            # Go to result directory
            results_dir = Path("eval_pipeline/results")
            eval_output_dir = (
                    results_dir
                    / self.experiment_name
                    / "main"
                    / "finetune"
            )
        return eval_output_dir

    def _determine_new_results_dir(self, project_root, relative_path, checkpoint_name):
        new_output_dir = os.path.dirname(self.out_dir).replace("checkpoints", "results")
        new_results_dir = project_root / new_output_dir / relative_path
        # Replace main folder with checkpoint name
        new_results_dir = Path(*(checkpoint_name if p == "main" else p for p in new_results_dir.parts))
        new_results_dir.parent.mkdir(parents=True, exist_ok=True)
        return new_results_dir

    def _delete_eval_models_directory(self):
        """Deletes the eval_pipeline/models/<experiment_name> directory after evaluation"""

        # Only delete on main process and if we're not using dummy data
        if self.process_index == 0 and not self.use_dummy_eval_data:
            models_dir = Path("eval_pipeline/models") / self.experiment_name
            if models_dir.exists() and models_dir.is_dir():
                logger.info(f"Cleaning up model directory in eval_pipeline: {models_dir}")
                try:
                    shutil.rmtree(models_dir)
                    logger.info(f"Successfully deleted {models_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete {models_dir}: {e}")


