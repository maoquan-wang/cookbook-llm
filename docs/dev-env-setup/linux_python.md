

## git setup

```bash
# config
git config --global user.name "your-name"
git config --global user.email "your-account@gmail.com"

# key gen
ssh-keygen

# copy your public key to github setting
cat ~/.ssh/id_rsa.pub


# install git-lfs
sudo apt-get install git-lfs
git lfs install  # set up Git LFS for your user account
git lfs pull # Fetch Git LFS changes from the remote

```

## use `uv` to manage python enviroment

```bash
# pip for python3
sudo apt-get install python3-pip

# install uv
python3 -m pip install uv

# create a new project by uv
mkdir project-name && cd project-name
uv init --name project-name --python 3.12 --package

# create a venv
uv venv --python 3.12

# active the python env
source .venv/bin/activate

# exit the python env
deactivate

# manage env dependencies
uv pip install ruff black isort
uv pip install huggingface_hub
uv pip install torch torchvision torchaudio
uv pip install huggingface
uv pip uninstall huggingface_hub

# manage dependencies in pyproject.toml
uv add huggingface_hub
uv remove huggingface_hub

# run script
uv run python example.py

# inspecte a package
uv pip show numpy
```


## ohmyzsh

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```
<details>
<summary>useful setting .bashrc or .zshrc</summary>

```bash
# HOME dir
export HOME="/jumbo/myhome"

# huggingface token and cache dirs
export HF_TOKEN="xxx"
export HF_HOME="/jumbo/.cache/hf"
export HF_DATASETS_CACHE="/jumbo/.cache/hf/datasets"

# wandb token
export WANDB_API_KEY="xxx"

```
</details>
