#!/bin/bash
#SBATCH --job-name=taal-lifecycle
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-lifecycle-%j.log
#SBATCH --error=logs/taal-lifecycle-%j.log

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

if (( $# != 4 )); then
    echo "Usage: sbatch $0 CONFIG CHECKPOINT OUTPUT_DIR EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$4" ]]; then
    echo "Checkout changed since submission; refusing a different experiment revision" >&2
    exit 1
fi
python -m utils.explorer_preflight --config "$1" --checkpoint "$2"
exec python -m evaluation.taal.state_audit \
    --config "$1" --checkpoint "$2" --output-dir "$3"
