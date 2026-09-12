#!/bin/bash
#SBATCH --job-name=fs-qwen-fw-2k-1k-512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fs-qwen-fw-2k-1k-512-%j.log
#SBATCH --error=logs/fs-qwen-fw-2k-1k-512-%j.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

exec torchrun \
    --standalone --nnodes=1 --nproc-per-node=1 \
    -m training.train --config training/configs/qwen3_0_6b_cpt_fw_swa_2k_1k_512.yaml
