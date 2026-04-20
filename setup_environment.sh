#!/bin/bash
set -e  # Exit immediately if a command fails

# ---------- Color definitions ----------
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m'

# ---------- Message helpers ----------
INFO="${BLUE}"
SUCCESS="${GREEN}"
WARN="${YELLOW}"
ERROR="${RED}"

# ---------- 1. Install Miniconda ----------
if [ ! -d "$HOME/miniconda3" ]; then
    echo -e "${ERROR}ERROR: Miniconda is not installed!${NC}"
    echo -e "${WARN}Please make sure to install miniconda on your system.${NC}"
    echo -e "${WARN}After installing miniconda, execute this setup script again.${NC}"
else
    echo -e "${INFO}Miniconda already installed, skipping.${NC}"
fi

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh

# ---------- 2. Create or activate environment ----------
ENV_NAME="py313"
if conda env list | grep -q "$ENV_NAME"; then
    echo -e "${INFO}Conda environment $ENV_NAME already exists, activating...${NC}"
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    echo -e "${SUCCESS}Conda environment $ENV_NAME activated!${NC}"
else
    echo -e "${INFO}Creating conda environment $ENV_NAME...${NC}"
    conda create -y -n $ENV_NAME python=3.13.2
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate $ENV_NAME
    echo -e "${SUCCESS}Conda environment $ENV_NAME created & activated!${NC}"
fi

# Define python environment path
ENV_PY="$HOME/miniconda3/envs/py313/bin/python"

# ---------- 3. Verify Python ----------
echo -e "${INFO}Using Python version:${NC}"
$ENV_PY -c "import sys; print(sys.version)"

# ---------- 4. Ensure git submodules are initialized ----------
if [ -f .gitmodules ]; then
    echo -e "${INFO}Initializing git submodules...${NC}"
    git submodule update --init --recursive
    echo -e "${SUCCESS}Initialized git submodules!${NC}"
fi

# ---------- 5. Install repo & submodule requirements ----------
echo -e "${INFO}Installing repository and submodule requirements...${NC}"
if [ -f requirements.txt ]; then
    $ENV_PY -m pip install --upgrade -r requirements.txt -r eval_pipeline/requirements.txt
    echo -e "${SUCCESS}Installed repository and submodule requirements!${NC}"
else
    echo -e "${ERROR}ERROR: requirements_local.txt not found, unable to setup environment!${NC}"
    echo -e "${WARN}Please make sure that the required packages are defined in a requirements.txt file${NC}"
    exit 1
fi

# ---------- 6. Install PyTorch ----------
echo -e "${INFO}Installing PyTorch...${NC}"
$ENV_PY -m pip install --upgrade \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --extra-index-url https://download.pytorch.org/whl/torch_stable.html
echo -e "${SUCCESS}Installed PyTorch!${NC}"

# ---------- 7. Hugging Face & WandB login ----------
if [ -t 0 ]; then
    echo -e "${INFO}Interactive shell detected.${NC}"

    echo -e "${INFO}Logging into Hugging Face (interactive)...${NC}"
    huggingface-cli login

    echo -e "${SUCCESS}Hugging Face login completed!${NC}"

    echo -e "${INFO}Logging into Weights & Biases (interactive)...${NC}"
    wandb login

    echo -e "${SUCCESS}Weights & Biases login completed!${NC}"
else
    echo -e "${ERROR}ERROR: Non-interactive shell detected.${NC}"
    echo -e "${ERROR}Manual login is REQUIRED before using this project.${NC}"
    echo
    echo -e "${WARN}Please run the following commands manually in an interactive shell:${NC}"
    echo
    echo "  huggingface-cli login"
    echo "  wandb login"
    echo
    echo -e "${WARN}Make sure your Hugging Face token has READ + WRITE permissions.${NC}"
    echo -e "${WARN}Make sure your Weights & Biases API key is valid.${NC}"
    echo
    exit 1
fi

echo -e "${SUCCESS}Setup complete!${NC}"