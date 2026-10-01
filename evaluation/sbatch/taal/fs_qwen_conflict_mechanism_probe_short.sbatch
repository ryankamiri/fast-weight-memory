#!/bin/bash
#SBATCH --job-name=taal-conflict-probe
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-conflict-probe-%j.log
#SBATCH --error=logs/taal-conflict-probe-%j.log

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

if (( $# != 2 )); then
    echo "Usage: sbatch $0 CHECKPOINT OUTPUT_JSON" >&2
    exit 2
fi
python -m data.taal_conflict_control \
    --output data/taal_conflict_control.jsonl --pairs 8 --window 128 --gap 192
python -m utils.explorer_preflight \
    --config evaluation/configs/taal/conflict_control_audit.yaml --checkpoint "$1"
exec python -m evaluation.taal.mechanism_probe \
    --checkpoint "$1" \
    --dataset data/taal_conflict_control.jsonl \
    --example-id conflict-00-0 \
    --compare-forget 0.0001 \
    --output "$2"
