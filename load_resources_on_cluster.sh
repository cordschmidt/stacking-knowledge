#!/bin/bash
#SBATCH --job-name=load_resources_on_cluster                    # job name
#SBATCH -t 0-00:05:00                                           # estimated time
#SBATCH -p react                                                # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                               # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
#SBATCH --nodes=1                                               # total number of nodes
#SBATCH --ntasks=1                                              # total number of tasks
#SBATCH --cpus-per-task=8                                       # number cores per task
#SBATCH --mail-type=all                                         # send mail when job begins and ends
#SBATCH --mail-user=None                                        # specify user mail adress for notifications
#SBATCH --output=./slurm_files/slurm-%x-%j.out                  # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err                   # where to write slurm error

# Load virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda init
conda activate py313

# Execute python test script
~/miniconda3/envs/py313/bin/python load_resources_on_cluster.py