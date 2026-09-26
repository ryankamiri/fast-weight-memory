#!/bin/bash
#SBATCH --job-name=fs-qwen-taal-overfit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --output=logs/fs-qwen-taal-overfit-%j.log
#SBATCH --error=logs/fs-qwen-taal-overfit-%j.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
# Keep the model and dataset cache location independent of the submission shell.
export HF_HOME="${HF_HOME:-/projects/llpr/amiri.ry/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
# Streaming the pinned dataset still contacts the Hub; use Explorer's proxy
# when the submission shell did not export one.
export http_proxy="${http_proxy:-http://10.99.0.130:3128}"
export https_proxy="${https_proxy:-http://10.99.0.130:3128}"
export CHECKPOINT_RUN_ID="taal-pilot1a-${SLURM_JOB_ID}"

# A single Python process receives Slurm's pre-limit TERM directly and can
# save final weights before the allocation expires.
exec python -m training.train \
    --config training/configs/taal/qwen3_0_6b_delayed_recall_overfit.yaml
