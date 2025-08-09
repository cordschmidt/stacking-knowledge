# Setup the Environment

The setup files and requirements were taken from the original [CLIMB repository](https://github.com/codebyzeb/CLIMB). They were adapted to ensure compatibility with local machines as well as HPC clusters. During this process, issues with installing requirements from the evaluation pipeline occured. To address these, the setup was modified to install all necessary requirements for the evaluation pipeline directly from a single requirements file ([requirements_local.txt](../requirements_local.txt) or [requirements_cluster.txt](../requirements_cluster.txt) respectively).

## Local Setup

Before setting up your environment locally, please ensure that `git`, `git-lfs`, and `Python 3.9` are installed on your machine. Then begin by cloning this repository to your local system. 

To install the virtual environment, use the following command:
```bash
sh setup_local.sh
```

Once installed, set the Python interpreter of your IDE to the [python executable](../env/bin/python).

### Support for Mac MX Chips

**TBD**

## Cluster Setup

Please connect to the cluster first (you might want [this instruction](Connect_to_Cluster.md).) To set up the environment on the cluster, make sure that `git`, `git-lfs`, and `Python 3.9` are installed on the cluster. Clone the repository onto the cluster. If necessary, within [setup_cluster.sh](../setup_cluster.sh) these modules will be loaded. Please adapt this code, as this process may vary depending on the cluster system and installations available. 

To install the virtual environment on the cluster, use the following command:
```bash
bash setup_cluster.sh
```

## Project Configuration

### Weights & Biases

To enable [Weights & Biases](https://wandb.ai/site/) integration, create an account on the platform. Next, create a `.env` file in the project directory and include the following content:

```plaintext
WANDB_USER=your-username
```

Update the experiment parameters in the [config.yaml](../conf/config.yaml) file to reflect your Weights & Biases project (group) and run name (name), e.g.:

```yaml
experiment:
  name: 'try_dry_run'
  group: 'Debug_Cluster'
```

If your computing nodes are not supporting an active internet connection, you can enable local/offline logging by adding `wandb_log_locally` parameter to the [config.yaml](../conf/config.yaml):

```yaml
experiment:
  wandb_log_locally: True
```

You can then later sync the logs after model training & evaluation.

### Huggingface

Currently, Huggingface is used to download models, tokenizers, and datasets.

To integrate [Huggingface](https://huggingface.co/welcome), create an account on their platform. Then, add the following content to your `.env` file:

```plaintext
HF_READ_TOKEN=hf_ABCDEFGHIJKLMNOP
HF_WRITE_TOKEN=hf_ABCDEFGHIJKLMNOP
```

#### Upload of Model Checkpoints

**TBD**
