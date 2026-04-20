# Quick HPC Guide

The following section provides essential commands for connecting to and interacting with the GWDG High-Performance Computing (HPC) cluster. 

As this is a brief reference guide, consulting the official documentation is highly recommended for comprehensive setup instructions, advanced usage and troubleshooting:

**[GWDG Deep Learning Tutorial](https://gitlab-ce.gwdg.de/hpc-team-public/deep-learning-with-gpu-cores):** A step-by-step guide detailing how to connect to the cluster and allocate GPU resources.

**[Official GWDG HPC Documentation](https://docs.hpc.gwdg.de/start_here/index.html):** The complete, overarching technical manual for the GWDG infrastructure.

**[Official SLURM Documentation](https://slurm.schedmd.com/):** The definitive resource for the SLURM workload manager, covering job scheduling, queue management, and advanced scripting.

## Connecting to the Cluster

Once the public SSH key has been uploaded to your Academiccloud account, access can be established using the following command:

```bash
ssh -i .ssh/id_ed25519.pub u12345@glogin9.hlrn.de
```

Here, `u12345` corresponds to your username. To clone a Git repository on the cluster, an SSH key is required. After generating the key and uploading it to GitHub, the repository can be cloned directly on the cluster:

```bash
git@github.com:cordschmidt/stacking-knowledge.git
```

Subsequently, the environment must be configured. Detailed instructions can be found in the [Environment Setup Guide](../docs/Setup_Environment.md).

## Prepare a Script

Specific configuration parameters must be provided within the script to ensure the job is properly scheduled by SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=train-model                     # job name
#SBATCH -t 01:00:00                                # estimated time
#SBATCH -p grete                                   # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                  # take 1 GPU, see [https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html](https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html) for more options
#SBATCH --mem-per-gpu=8G                           # setting the right constraints for the splitted gpu partitions
#SBATCH --nodes=1                                  # total number of nodes
#SBATCH --ntasks=1                                 # total number of tasks
#SBATCH --cpus-per-task=8                          # number cores per task
#SBATCH --mail-type=all                            # send mail when job begins and ends
#SBATCH --mail-user=None                           # specify user mail adress for notifications
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error

source env/bin/activate
```

For detailed instructions on requesting different GPU configurations or slices, please refer to the [GWDG GPU Usage Documentation](https://docs.hpc.gwdg.de/how_to_use/slurm/gpu_usage/index.html).

## Scheduling a Job

To schedule a job (such as a training script) for execution, submit it using:

```bash
sbatch run_train.sh
```

Active and pending jobs can be monitored with the following command:

```bash
squeue -u u12345
```

Here, `u12345` corresponds to your specific username.

## Check Computing Resources

```bash
sbalance --assoc
```

## Connect to the HPC Using VS Code

To access the HPC and its filesystem via VS Code, the [Remote SSH extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) must be installed. 

Additionally, an SSH config file containing the login credentials for the GWDG HPC must be created or modified. On macOS, this file is typically located at `/Users/YOUR_USERNAME/.ssh/config`. It should contain the following configuration:

```text
Host gwdg-cluster-some-name
  User u12345
  LocalForward 8888 localhost:8888
  HostName glogin9.hlrn.de
  IdentityFile ~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY
```

Here, `u12345` represents your HPC username, and `~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY` is the path to the SSH key used for authentication. Once configured, the HPC will be accessible through the Remote-SSH extension in VS Code.

## Use Jupyter Notebook on the HPC

To utilize HPC resources within a Jupyter Notebook on your local machine (e.g., for examining model behavior), a SLURM job can be submitted to create an interactive session:

```bash
sbatch start_interactive_session.sh
```

The script `start_interactive_session.sh` should contain the following configuration:

```bash
#!/bin/bash -l
#SBATCH --constraint=inet
#SBATCH --job-name=interactive_session              # Name of the job
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=2                           # Number of CPU cores per task
#SBATCH --nodes=1                                   # Ensure that all cores are on one machine
#SBATCH --time=0-01:00                              # Runtime in D-HH:MM
#SBATCH --output=./slurm_files/slurm-%x-%j.out      # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err       # where to write slurm error
#SBATCH --mail-type=ALL                             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=None                            # Email to which notifications will be sent
#SBATCH -p react                                    # Partition to submit to
#SBATCH -G 2g.20gb:2                                # Number of requested GPUs
 
source ~/.bashrc 
conda init 
conda activate dnlp
sleep infinity
```

To access the Jupyter Notebook locally, an SSH connection to the allocated compute node must be established after the job has started. To verify if the job is running and identify the assigned node, use:

```bash
squeue --me
```

To connect to the specific node via SSH in VS Code, ensure any previous connections (such as to the login node) using port-forwarding are closed. Then, append the following entry to your SSH config file:

```text
Host gwdg-cluster-dnlp-ggpu101
   HostName ggpu101
   User u12345
   LocalForward 8888 localhost:8888
   IdentityFile ~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY
   Port 22
   ProxyJump gwdg-cluster-climb_master-thesis
```

Here, `ggpu101` corresponds to the allocated node's name, `u12345` is your HPC username, and `~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY` is the path to your SSH key. After connecting to the node via the Remote-SSH extension, JupyterLab can be initialized:

```bash
python -m jupyterlab
```

JupyterLab will then be accessible on your local machine via port-forwarding using the URL provided in the terminal.

## Array Jobs

For advanced job scheduling, reviewing the [Array Jobs Documentation](https://docs.hpc.gwdg.de/how_to_use/slurm/job_array/index.html) is highly recommended. Array jobs provide a powerful method for running multiple experiments simultaneously without the need to write separate scripts. 

This is achieved by adding a directive such as `#SBATCH -a 0-6` at the beginning of the script, which will in this case execute the script seven times. This approach is particularly useful for running e.g. multiple random seeds for a single experiment:

```bash
#!/bin/bash
#SBATCH -a 0-6                                                  # array job with 7 tasks
#SBATCH --job-name=base_continual_lr_only                       # job name
#SBATCH -t 0-10:00:00                                           # estimated time
#SBATCH -p react                                                # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                               # take 1 GPU, see [https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html](https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html) for more options
#SBATCH --nodes=1                                               # total number of nodes
#SBATCH --ntasks=1                                              # total number of tasks
#SBATCH --cpus-per-task=8                                       # number cores per task
#SBATCH --mail-type=all                                         # send mail when job begins and ends
#SBATCH --mail-user=None                                        # specify user mail adress for notifications
#SBATCH --output=./slurm_files/slurm-%x-%j.out                  # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err                   # where to write slurm error
#SBATCH --constraint=inet

# Define seeds to test
SEEDS=(43 123 456 789 1024 1204 1578)

# Map the SLURM_ARRAY_TASK_ID to the specific seed
# SLURM_ARRAY_TASK_ID goes from 0 to 6
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
```

## Workspaces

Utilizing workspaces is highly recommended to prevent storage depletion, which occurs rapidly when saving multiple model checkpoints. A workspace can be created using the following command:

```bash
ws_allocate -F ceph-ssd -r 2 -m myemail@example.com MyWorkspaceName 30
```

This command initializes a workspace named `MyWorkspaceName`. Workspaces are temporary and exist for 30 days. Two days prior to expiration, a reminder is sent to the specified email address (`myemail@example.com`). Workspaces can be extended up to two times using the following command:

```bash
ws_extend -F ceph-ssd MyWorkspaceName 30
```

For comprehensive instructions, please refer to the [Workspace Documentation](https://docs.hpc.gwdg.de/how_to_use/storage_systems/workspaces/index.html#allocating) provided by GWDG. Ensure your workspace is properly configured before initiating large training runs.