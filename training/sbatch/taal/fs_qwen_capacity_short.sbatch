#!/bin/bash
#SBATCH --job-name=taal-capacity
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/taal-capacity-%j.log
#SBATCH --error=logs/taal-capacity-%j.log

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

if (( $# != 3 )) || [[ "$(git rev-parse HEAD)" != "$3" ]]; then
    echo "Usage: sbatch $0 CONFIG RUN_PREFIX EXPECTED_COMMIT (must match checkout)" >&2
    exit 2
fi
# Dataset is generated once before queueing, never concurrently on GPU nodes.
python -m utils.explorer_preflight --config "$1"
export CHECKPOINT_RUN_ID="$2-${SLURM_JOB_ID}"
export TAAL_TIMING_FIRST_BATCH=1
exec python -m training.train --config "$1"
