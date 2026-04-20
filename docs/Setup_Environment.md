# Environment Setup

This document outlines the necessary steps to configure the execution environment.

All dependencies required to run the codebase are specified in [`requirements.txt`](../requirements.txt). If the source code is modified and additional packages are required, ensure they are appended to this file.

## Preconditions

### Miniconda

Download and install Miniconda for your specific operating system. On Linux and macOS, this typically involves executing the provided installer script and ensuring it installs to the default `~/miniconda3` directory. If a custom installation path is utilized, the path within the setup script must be adjusted accordingly.

### Hugging Face

Hugging Face is utilized for retrieving models, tokenizers, and datasets.

To enable [Hugging Face](https://huggingface.co/welcome) integration, create an account on their platform. Then, append the following credentials to your `.env` file:

```plaintext
HF_READ_TOKEN=hf_ABCDEFGHIJKLMNOP
HF_WRITE_TOKEN=hf_ABCDEFGHIJKLMNOP
```

### Weights & Biases

To enable [Weights & Biases](https://wandb.ai/site/) integration for experiment tracking, create an account on the platform. Subsequently, create a .env file in the root project directory and include the following configuration:

```plaintext
WANDB_USER=your-username
```

Update the experiment parameters within the [config.yaml](../conf/config.yaml) file to designate your specific W&B project (group) and run identifier (name), for example:

```yaml
experiment:
  name: 'try_dry_run'
  group: 'Debug_Cluster'
```

If the compute nodes lack an active internet connection, local offline logging can be enabled by setting the `wandb_log_locally` parameter in [config.yaml](../conf/config.yaml):

```yaml
experiment:
  wandb_log_locally: True
```

These logs can be manually synchronized to the cloud after the model training and evaluation phases are complete.

## Install the environment

Prior to execution, ensure the setup script has the appropriate executable permissions:

```bash
chmod +x setup_environment.sh
```

To automatically configure the environment on either a local machine or an HPC cluster, execute the following command:

```bash
./setup_environment.sh
```