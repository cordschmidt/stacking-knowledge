#!/bin/bash
#SBATCH -a 0-17                                                  # array job with 7 tasks
#SBATCH --job-name=hyp_gradual_stacking_clean_optim                         # job name
#SBATCH -t 0-14:00:00                                           # estimated time
#SBATCH -p react                                                # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                               # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --nodes=1                                               # total number of nodes
#SBATCH --ntasks=1                                              # total number of tasks
#SBATCH --cpus-per-task=12                                       # number cores per task
#SBATCH --mail-type=all                                         # send mail when job begins and ends
#SBATCH --mail-user=None                                        # specify user mail adress for notifications
#SBATCH --output=./slurm_files/slurm-%x-%j.out                  # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err                   # where to write slurm error
#SBATCH --constraint=inet
#SBATCH --exclude=ggpu140

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
FIXED_SEED=42
FIXED_FLOPS=8.36361512681472e15
FIXED_COMPUTE_EQUIV_PARAMS=14442752
ALIGN_WITH_DATA_CURRICULUM=False

# Define combinations to test
STAGES=(4 4 4 4 4 4 6 6 6 6 6 6 8 8 8 8 8 8)
BLOCKS=(4 4 4 6 5 5 3 3 3 5 4 4 2 2 2 4 3 3)
ALPHAS=(0.0 1.0 2.0 0.0 1.0 2.0 0.0 1.0 2.0 0.0 1.0 2.0 0.0 1.0 2.0 0.0 1.0 2.0)

CURRENT_NUMBER_OF_STAGES=${STAGES[$SLURM_ARRAY_TASK_ID]}
CURRENT_BLOCK_SIZE=${BLOCKS[$SLURM_ARRAY_TASK_ID]}
CURRENT_ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}
SAFE_ALPHA_NAME=${CURRENT_ALPHA//./_}

# Run the Hydra command for this single combination
python train.py \
  trainer.lr=$FIXED_LR \
  trainer.lr_scheduler_type=$FIXED_SCHED \
  trainer.max_flops=$FIXED_FLOPS \
  trainer.num_warmup_steps=236 \
  trainer.max_training_steps=23565 \
  gradual_stacking.enabled=True \
  gradual_stacking.cleaning_optimizer_state=True \
  gradual_stacking.k_number_of_stages=$CURRENT_NUMBER_OF_STAGES \
  gradual_stacking.alpha=$CURRENT_ALPHA \
  gradual_stacking.layer_per_block=$CURRENT_BLOCK_SIZE \
  +gradual_stacking.number_non_embedding_params_compute_equivalent_model=$FIXED_COMPUTE_EQUIV_PARAMS \
  gradual_stacking.align_with_staged_data_curriculum=$ALIGN_WITH_DATA_CURRICULUM \
  experiment.name="stack_k${CURRENT_NUMBER_OF_STAGES}_b${CURRENT_BLOCK_SIZE}_a${SAFE_ALPHA_NAME}" \
  experiment.group="Hyperparam_Stacking" \
  experiment.seed=$FIXED_SEED \
  experiment.dry_run=False \
  experiment.skip_execution_of_eval_scripts_for_debugging=False