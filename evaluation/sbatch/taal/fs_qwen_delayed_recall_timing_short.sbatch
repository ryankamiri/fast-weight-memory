#!/bin/bash
#SBATCH --job-name=fs-qwen-taal-audit-timing-short
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fs-qwen-taal-audit-timing-short-%j.log
#SBATCH --error=logs/fs-qwen-taal-audit-timing-short-%j.log

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
export TAAL_TIMING_FIRST_BATCH=1
export TAAL_KERNEL_SAMPLE=1

if (( $# < 1 )); then
    echo "Usage: sbatch $0 CHECKPOINT [OUTPUT_DIR]" >&2
    exit 2
fi

# Separate additive diagnostic: preserve all Pilot 1B examples/conditions/traces.
python -m utils.explorer_preflight \
    --config evaluation/configs/taal/delayed_recall_state_audit.yaml \
    --checkpoint "$1"
exec python -m evaluation.taal.state_audit \
    --config evaluation/configs/taal/delayed_recall_state_audit.yaml \
    --checkpoint "$1" \
    --output-dir "${2:-output/taal/pilot-1b-timing-${SLURM_JOB_ID}}"
