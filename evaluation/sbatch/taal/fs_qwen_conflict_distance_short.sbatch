#!/bin/bash
#SBATCH --job-name=taal-conflict-distance
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-conflict-distance-%j.log
#SBATCH --error=logs/taal-conflict-distance-%j.log

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

if (( $# != 3 )); then
    echo "Usage: sbatch $0 GAP CHECKPOINT OUTPUT_DIR" >&2
    exit 2
fi
GAP="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"
python -m evaluation.taal.prepare_conflict_distance \
    --gap "${GAP}" --output-dir "${OUTPUT_DIR}"
CONFIG="${OUTPUT_DIR}/audit_config.yaml"
python -m utils.explorer_preflight --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
exec python -m evaluation.taal.state_audit \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --output-dir "${OUTPUT_DIR}"
