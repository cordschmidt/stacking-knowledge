#!/bin/bash
#SBATCH -a 0-6                                                  # array job with 7 tasks
#SBATCH --job-name=base_dyn_lr                      # job name
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
export STORAGE_DIR=/mnt/ceph-ssd/workspaces/ws/react_ag_beinborn_students/u12860-ws_stacking_knowledge/tmp
export WANDB_DIR=$STORAGE_DIR
export WANDB_DATA_DIR=$STORAGE_DIR
export HF_DATASETS_CACHE=$STORAGE_DIR
export TMPDIR=/user/corderic.schmidt/u12860/tmp_sockets
mkdir -p $TMPDIR

# Prevent wandb crash errors
export WANDB_SERVICE_WAIT=300
sleep $((RANDOM % 60))

# Define fixed parameters
FIXED_LR=1e-3
FIXED_SCHED="cosine_with_min_lr"
FIXED_CURRICULUM="dynamic_linear_staged_data_split"
FIXED_CONST_LR=5e-4

# Define seeds to test
SEEDS=(43 123 456 789 1024 1204 1578)

# Map the SLURM_ARRAY_TASK_ID to the specific seed
# SLURM_ARRAY_TASK_ID goes from 0 to 6
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Run the Hydra command for this single combination
python train.py \
  trainer.lr=$FIXED_LR \
  trainer.lr_scheduler_type=$FIXED_SCHED \
  +data_curriculum=$FIXED_CURRICULUM \
  infinite_lr_scheduler.enabled=True \
  +infinite_lr_scheduler.lr_const=$FIXED_CONST_LR \
  +infinite_lr_scheduler.lr_const_steps=1400 \
  +infinite_lr_scheduler.lr_min=1e-4 \
  continual_pretraining.enable_lr_reset=False \
  experiment.name="dyn_cl_inf_lr_only_${CURRENT_SEED}" \
  experiment.group="Fin_Seeds_Continual" \
  experiment.seed=$CURRENT_SEED \
  experiment.dry_run=False \
  experiment.skip_execution_of_eval_scripts_for_debugging=False