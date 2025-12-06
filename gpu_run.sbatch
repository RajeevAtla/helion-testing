#!/bin/bash
#SBATCH --job-name=run-helion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --constraint=ampere
#SBATCH --exclude=gpu[017,018]
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@rutgers.edu
#SBATCH --output=slurm/slurm_%j.out
#SBATCH --error=slurm/slurm_%j.err

echo "Job started at: $(date)"

mkdir -p slurm

module purge
module use /projects/community/modulefiles
module load apptainer
module load git/2.35.1-ez82

# check apptainer version
apptainer --version

# set cache dir to scratch so we don't run out of space
mkdir -p /scratch/${USER}/apptainer
export APPTAINER_CACHEDIR=/scratch/${USER}/apptainer

# get current git commit
echo "Current git commit: "
git rev-parse --short HEAD

PYTORCH_IMAGE=jax_25.10-py3.sif

# pull nvidia's jax container if not pulled
if [ ! -f "${PYTORCH_IMAGE}" ]; then
  echo "Cached apptainer image not found, pulling new one"
  apptainer pull "${PYTORCH_IMAGE}" docker://nvcr.io/nvidia/pytorch:25.11-py3
else
  echo "Using cached apptainer image"
fi

# load and execute commands on the container once it's been pulled
apptainer exec --nv "${PYTORCH_IMAGE}" bash -c '
  echo "Container started"
  set -euo pipefail
  
  # uv setup
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.local/bin/env
  echo "uv installed!"
  uv --version
  
  # sync dependencies
  uv sync
  
  # check pytorch+nvidia drivers+cuda are working correctly
  nvidia-smi
  nvcc --version
  uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

  # set correct libcuda
  export TRITON_LIBCUDA_PATH=/lib64
  
  # run training
  uv run python main.py
'

echo "Job completed at: $(date)"
