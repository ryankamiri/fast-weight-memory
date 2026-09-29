#!/bin/bash
#SBATCH --job-name=fs-qwen-taal-chunk8-short
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --output=logs/fs-qwen-taal-chunk8-short-%j.log
#SBATCH --error=logs/fs-qwen-taal-chunk8-short-%j.log

set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/projects/llpr/amiri.ry/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy="${http_proxy:-http://10.99.0.130:3128}"
export https_proxy="${https_proxy:-http://10.99.0.130:3128}"
export CHECKPOINT_RUN_ID="taal-pilot1a-chunk8-${SLURM_JOB_ID}"
export TAAL_TIMING_FIRST_BATCH=1
export TAAL_KERNEL_SAMPLE=1

# Authorized cadence experiment: aggregate 8 writes at chunk-start weights.
# Keep batch size, accumulation, loss, and optimization matched to chunk size 1.
python -m utils.explorer_preflight \
    --config training/configs/taal/qwen3_0_6b_delayed_recall_overfit_chunk8.yaml
exec python -m training.train \
    --config training/configs/taal/qwen3_0_6b_delayed_recall_overfit_chunk8.yaml
