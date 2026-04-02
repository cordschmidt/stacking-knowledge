#!/bin/bash
#SBATCH -a 0-17                                                 # array job with 18 tasks
#SBATCH --job-name=hyp_continual_lr_only                        # job name
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
FIXED_CURRICULUM="prop_alpha_strict_data_split"

# Define rewarm learning rates (200%, 150%, 100%, 50%, 25%, 10% of 1e-3)
REWARM_LRS=(2e-3 1.5e-3 1e-3 5e-4 2.5e-4 1e-4)
REWARM_PERCS=(200 150 100 50 25 10) # for experiment names
# Define alphas to test
ALPHAS=(0.0 1.0 2.0)

# Integer division gives the LR index (changes every 3 tasks)
LR_IDX=$((SLURM_ARRAY_TASK_ID / 3))
# Modulo gives the alpha index (cycles 0, 1, 2)
ALPHA_IDX=$((SLURM_ARRAY_TASK_ID % 3))

# Map idx to actual values
CURRENT_REWARM_LR=${REWARM_LRS[$LR_IDX]}
CURRENT_REWARM_PERC=${REWARM_PERCS[$LR_IDX]}
CURRENT_ALPHA=${ALPHAS[$ALPHA_IDX]}
SAFE_ALPHA_NAME=${CURRENT_ALPHA//./_}

# Run the Hydra command for this single combination
python train.py \
  trainer.lr=$FIXED_LR \
  trainer.lr_scheduler_type=$FIXED_SCHED \
  +data_curriculum=$FIXED_CURRICULUM \
  gradual_stacking.alpha=$CURRENT_ALPHA \
  gradual_stacking.k_number_of_stages=5 \
  continual_pretraining.enable_lr_reset=True \
  +continual_pretraining.max_rewarm_lr=$CURRENT_REWARM_LR \
  +continual_pretraining.rewarm_fraction=0.01 \
  experiment.name="continual_lr_reset_to_${CURRENT_REWARM_PERC}perc_alpha_${SAFE_ALPHA_NAME}" \
  experiment.group="Hyperparam_Continual_Reset" \
  experiment.seed=42 \
  experiment.dry_run=False \
  experiment.skip_execution_of_eval_scripts_for_debugging=False