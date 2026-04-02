#!/bin/bash
#SBATCH -a 0-6                                                  # array job with 7 tasks
#SBATCH --job-name=base_strict_cl_linear                           # job name
#SBATCH -t 0-10:00:00                                           # estimated time
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

# Define fixed parameters
FIXED_LR=1e-3
FIXED_SCHED="cosine_with_min_lr"

# Define seeds to test
SEEDS=(43 123 456 789 1024 1204 1578)

# Map the SLURM_ARRAY_TASK_ID to the specific seed
# SLURM_ARRAY_TASK_ID goes from 0 to 6
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Run the Hydra command for this single combination
python train.py \
  trainer.lr=$FIXED_LR \
  trainer.lr_scheduler_type=$FIXED_SCHED \
  +data_curriculum="linear_strict_data_split" \
  experiment.name="strict_data_cl_linear_${CURRENT_SEED}" \
  experiment.group="Fin_Seeds_Stacking_DCL" \
  experiment.seed=$CURRENT_SEED \
  experiment.dry_run=False \
  experiment.skip_execution_of_eval_scripts_for_debugging=False