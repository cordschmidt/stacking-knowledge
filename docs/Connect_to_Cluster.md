# Connecting to the Cluster


A tutorial from GWDG how to connect to the cluster can be found [here](https://gitlab-ce.gwdg.de/hpc-team-public/deep-learning-with-gpu-cores). After uploading the public SSH key to your Academiccloud account, access can be established using:

```bash
ssh -i .ssh/id_ed25519.pub u12860@glogin9.hlrn.de
```

Where `u12860` is corresponding to your username. For cloning a git repository on the cluster, an SSH key is required. After generating the key and uploading it to github, the repository can be cloned at the cluster:

```bash
git@github.com:cordschmidt/climb_master_thesis.git
```

After that the environment has to be set up.

# Prepare a Script

Some information has to be provided in order to get the job scheduled by slurm:

```bash
#!/bin/bash
#SBATCH --job-name=train-model                     # job name
#SBATCH -t 01:00:00                                # estimated time
#SBATCH -p grete                                   # the partition you are training on (i.e., which nodes), for nodes see sinfo -p grete:shared --format=%N,%G
#SBATCH -G A100:1                                  # take 1 GPU, see https://docs.hpc.gwdg.de/compute_partitions/gpu_partitions/index.html for more options
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

In order to see in more detail how to request different slices / numbers of GPUs please check out [GWDG Documentation](https://docs.hpc.gwdg.de/how_to_use/slurm/gpu_usage/index.html).

# Scheduling a Job

In order to execute any script, please make sure that the required environment is installed and active:

```bash
source env/bin/activate
```

To schedule a job for later execution you can run e.g. a training script like this:

```bash
sbatch run_train.sh
```

Active and pending jobs can be monitored using:

```bash
squeue -u u12860
```

Where `u12860` is corresponding to your username.

# Check Computing Resources

```bash
sbalance --assoc
```

# Connect to the HPC Using VS Code

In order to access the HPC, including its filesystem, you have to install the [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) extension.

Additionally, you have to adjust / create the SHH config file containing the login information for the GWDG HPC. On Mac the file is usually located at `/Users/YOUR_USERNAME/.ssh`. It should contain the following infromation:

```
Host gwdg-cluster-climb_master-thesis
  User u12860
  LocalForward 8888 localhost:8888
  HostName glogin9.hlrn.de
  IdentityFile ~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY
```

Where `u12860` corresponds to your Username on the HPC and `~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY` to the path to your SSH key used for logging into the HPC.

After that the HPC should be accessible through the Remote-SSH extension.

# Use Jupyter Notebook on the HPC

In order to use the HPC resources within a Jupyter Notebook on your local machine, e.g. for examining model behaviour, you can submit a slurm job to create an interactive session at the login node:

```bash
sbatch start_interactive_session.sh
```

Where the script `start_interactive_session.sh` contains the following information:

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

In order to access the Jupyter Notebook from your local machine, you have to connect to the allocated node via SSH after the job has been started. To check, whether the job has started and which node has been assigned to it, you can use:

```bash
squeue --me
```

To connect to the node via SSH in VS Code, make sure any previous connection (e.g. to the login node) using port-forwarding is closed and adjust your SSH config by adding this entry:

```bash
Host gwdg-cluster-dnlp-ggpu101
   HostName ggpu101
   User u12860
   LocalForward 8888 localhost:8888
   IdentityFile ~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY
   Port 22
   ProxyJump gwdg-cluster-climb_master-thesis
```

Where `ggpu101` corresponds to the name of the node allocated to your job, `u12860` to your Username on the HPC and `~/.ssh/NAME_OF_YOUR_PRIVATE_SSH_KEY` to the path to your SSH key used for logging into the HPC. After accessing the node using Remote-SSH extension you can start Jupyterlab:

```bash
python -m jupyterlab
```

Now Jupyterlab should be accessible on your local machine through port-forwarding at the given link in the terminal.