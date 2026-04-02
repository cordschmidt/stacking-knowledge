#!/bin/bash
#SBATCH -a 0-11                                                 # array job with 12 tasks
#SBATCH --job-name=baseline_small_lr_hyperparm_search           # job name
#SBATCH -t 0-13:00:00                                           # estimated time
#SBATCH -p react                                                # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                               # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --nodes=1                                               # total number of nodes
#SBATCH --ntasks=1                                              # total number of tasks
#SBATCH --cpus-per-task=8                                       # number cores per task
#SBATCH --mail-type=all                                         # send mail when job begins and ends
#SBATCH --mail-user=None                                        # specify user mail adress for notifications
#SBATCH --output=./slurm_files/slurm-%x-%j.out                  # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err                   # where to write slurm error
#SBATCH --constraint=inet

# Enforce offline mode for datasets and transformers for compute nodes
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Load virtual environment
source ~/miniconda3/bin/activate
conda init
conda activate py313

# Create a temporary dir for wandb logging
export TMPDIR=/mnt/ceph-ssd/workspaces/ws/react_ag_beinborn_students/u12860-ws_stacking_knowledge/stacking-knowledge/tmp_storage
mkdir -p $TMPDIR
export WANDB_DIR=$TMPDIR
export WANDB_DATA_DIR=$TMPDIR

# Prevent wandb crash errors
export WANDB_SERVICE_WAIT=300
sleep $((RANDOM % 60))

# Define parameters to test
LRS=(1e-3 5e-4 3e-4 1e-4 5e-5 3e-5)
SCHEDS=("linear" "cosine_with_min_lr")

# Map the SLURM_ARRAY_TASK_ID to the specific parameters
# SLURM_ARRAY_TASK_ID goes from 0 to 11.
LR_IDX=$(( SLURM_ARRAY_TASK_ID % 6 ))
SCHED_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
CURRENT_LR=${LRS[$LR_IDX]}
CURRENT_SCHED=${SCHEDS[$SCHED_IDX]}

# Run the Hydra command for this single combination
python train.py \
  trainer.lr=$CURRENT_LR \
  trainer.lr_scheduler_type=$CURRENT_SCHED \
  experiment.name="baseline_small_${CURRENT_LR}_${CURRENT_SCHED}" \
  experiment.group="Baseline_Small_LR_Search" \
  experiment.dry_run=False \
  experiment.skip_execution_of_eval_scripts_for_debugging=False
  