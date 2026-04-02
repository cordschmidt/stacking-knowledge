#!/bin/bash
#SBATCH -a 0-19                                                 # array job with 7 tasks
#SBATCH --job-name=hyp_dyn_replay                          # job name
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
FIXED_SEED=42 # Using a single seed for all combinations
FIXED_LR=1e-3
FIXED_SCHED="cosine_with_min_lr"
FIXED_CONST_LR=5e-4
FIXED_CURRICULUM="dynamic_linear_staged_data_split"

# Define the 20 hyperparameter combinations
# Configs 0-4: previous_stage_only (decay is ignored, set to 1.0 as dummy)
# Configs 5-19: all_previous_stages with decays 1.0 (no decay), 0.5, and 0.1 (agressive decay)
MODES=(
  "previous_stage_only" "previous_stage_only" "previous_stage_only" "previous_stage_only" "previous_stage_only"
  "all_previous_stages" "all_previous_stages" "all_previous_stages"
  "all_previous_stages" "all_previous_stages" "all_previous_stages"
  "all_previous_stages" "all_previous_stages" "all_previous_stages"
  "all_previous_stages" "all_previous_stages" "all_previous_stages"
  "all_previous_stages" "all_previous_stages" "all_previous_stages"
)
FRACTIONS=(
  0.01 0.05 0.1 0.2 0.5
  0.01 0.01 0.01
  0.05 0.05 0.05
  0.1  0.1  0.1
  0.2  0.2  0.2
  0.5  0.5  0.5
)
DECAYS=(
  1.0 1.0 1.0 1.0 1.0 # (The first 5 are for 'previous_stage_only' where decay doesn't matter)
  1.0 0.5 0.1
  1.0 0.5 0.1
  1.0 0.5 0.1
  1.0 0.5 0.1
  1.0 0.5 0.1
)
# Short names: prv/all + fraction (01, 05, 10, 20) + decay (d10, d05, d01)
NAMES=(
  "prv_1_dyn" "prv_5_dyn" "prv_10_dyn" "prv_20_lin" "prv_50_dyn"
  "all_1_no_dec_dyn" "all_1_dec50_dyn" "all_1_dec10_dyn"
  "all_5_no_dec_dyn" "all_5_dec50_dyn" "all_5_dec10_dyn"
  "all_10_no_dec_dyn" "all_10_dec50_dyn" "all_10_dec10_dyn"
  "all_20_no_dec_dyn" "all_20_dec50_dyn" "all_20_dec10_dyn"
  "all_50_no_dec_dyn" "all_50_dec50_dyn" "all_50_dec10_dyn"
)

# Extract the specific values for this task directly using the array ID
CURRENT_MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
CURRENT_FRAC=${FRACTIONS[$SLURM_ARRAY_TASK_ID]}
CURRENT_DECAY=${DECAYS[$SLURM_ARRAY_TASK_ID]}
CURRENT_NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}

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
  +continual_pretraining.data_replay_mode=$CURRENT_MODE \
  continual_pretraining.data_replay_fraction=$CURRENT_FRAC \
  continual_pretraining.data_replay_decay=$CURRENT_DECAY \
  experiment.name="dyn_cl_replay_${CURRENT_NAME}_final" \
  experiment.group="Hyperparam_Continual_Replay" \
  experiment.seed=$FIXED_SEED \
  experiment.dry_run=False \
  experiment.skip_execution_of_eval_scripts_for_debugging=False