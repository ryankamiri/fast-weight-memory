#!/bin/bash
#SBATCH --job-name=fs-qwen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=fs-qwen-%j.log
#SBATCH --error=fs-qwen-%j.log

set -euo pipefail

# Submit from the repo root. Set up uv, HF/W&B login, and caches beforehand.
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Slurm owns CUDA_VISIBLE_DEVICES. Keep one process until DDP is implemented.
exec uv run --frozen --no-dev torchrun \
    --standalone --nnodes=1 --nproc-per-node=1 \
    -m training.train --config "${1:-training/configs/qwen3_0_6b.yaml}"
