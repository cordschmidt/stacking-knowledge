if [ ! -d "env_repo" ]; then
	python3.13 -m venv env_repo
	source env_repo/bin/activate
	python -c "import sys; print('Using Python version:', sys.version)"
	# Install pytorch
	echo -e "\e[1;32mInstalling torch...\e[0m"
	pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/torch_stable.html --upgrade --force-reinstall
	# Install repo requirements
	# echo -e "\e[1;32mInstalling repository requirements...\e[0m"
	# pip install --force-reinstall -v -r requirements_local.txt
else 
	source env_repo/bin/activate
	# echo -e "\e[1;Updating repository requirements...\e[0m"
	# pip install -v -r requirements_local.txt
fi
source .env

export PATH="$(pwd)/lib/bin:$PATH"
