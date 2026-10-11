#!/bin/bash
#SBATCH --job-name=taal-address
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-address-%j.log
#SBATCH --error=logs/taal-address-%j.log

set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/projects/llpr/amiri.ry/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy="${http_proxy:-http://10.99.0.130:3128}"
export https_proxy="${https_proxy:-http://10.99.0.130:3128}"
if (( $# != 4 )) || [[ "$(git rev-parse HEAD)" != "$4" ]]; then
    echo "Usage: sbatch $0 reconstruction|oracle CHECKPOINT OUTPUT EXPECTED_SHA"
    exit 2
fi
case "$1" in
    reconstruction|oracle) ;;
    *) echo "Unknown diagnostic $1"; exit 2 ;;
esac
CONFIG="evaluation/configs/taal/address_${1}.yaml"
python -m utils.explorer_preflight --config "$CONFIG" --checkpoint "$2"
exec python -m evaluation.taal.address_diagnostic --config "$CONFIG" --checkpoint "$2" --output-dir "$3"
