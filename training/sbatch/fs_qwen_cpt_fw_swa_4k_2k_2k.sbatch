#!/bin/bash
#SBATCH --job-name=fs-qwen-cpt-fw-swa-4k-2k-2k
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fs-qwen-cpt-fw-swa-4k-2k-2k-%j.log
#SBATCH --error=logs/fs-qwen-cpt-fw-swa-4k-2k-2k-%j.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

# Submit from the repo root. Set up HF/W&B login and caches beforehand.
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Slurm owns CUDA_VISIBLE_DEVICES. Keep one process until DDP is implemented.
exec torchrun \
    --standalone --nnodes=1 --nproc-per-node=1 \
    -m training.train --config training/configs/qwen3_0_6b_cpt_fw_swa_4k_2k_2k.yaml
