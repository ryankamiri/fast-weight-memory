#!/bin/bash
#SBATCH --job-name=taal-no-fresh-train
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/taal-no-fresh-train-%j.log
#SBATCH --error=logs/taal-no-fresh-train-%j.log

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

if (( $# != 1 )) || [[ "$(git rev-parse HEAD)" != "$1" ]]; then
    echo "Usage: sbatch $0 EXPECTED_COMMIT (must match checkout)" >&2
    exit 2
fi
CONFIG=training/configs/taal/qwen3_0_6b_conflict_no_fresh_query.yaml
if [[ ! -f data/taal_conflict_control.jsonl ]]; then
    python -m data.taal_conflict_control --output data/taal_conflict_control.jsonl --pairs 8 --window 128 --gap 192
fi
python -m utils.explorer_preflight --config "${CONFIG}"
export CHECKPOINT_RUN_ID="taal-conflict-no-fresh-query-${SLURM_JOB_ID}"
exec python -m training.train --config "${CONFIG}"
