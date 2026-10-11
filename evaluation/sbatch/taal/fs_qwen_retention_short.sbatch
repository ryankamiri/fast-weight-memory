#!/bin/bash
#SBATCH --job-name=taal-retention
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-retention-%j.log
#SBATCH --error=logs/taal-retention-%j.log

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
if (( $# != 5 )) || [[ "$(git rev-parse HEAD)" != "$5" ]]; then
    echo "Usage: sbatch $0 readers|replay|interference CHECKPOINT SOURCE OUTPUT EXPECTED_SHA"
    exit 2
fi
case "$1" in
    readers|replay|interference) ;;
    *) echo "Unknown diagnostic $1"; exit 2 ;;
esac
CONFIG="evaluation/configs/taal/retention_${1}.yaml"
python -m utils.explorer_preflight --config "$CONFIG" --checkpoint "$2"
if [[ "$1" == readers ]]; then
    exec python -m evaluation.taal.retention_readers --config "$CONFIG" --source "$3" --output-dir "$4"
fi
exec python -m evaluation.taal.retention_replay --config "$CONFIG" --checkpoint "$2" --output-dir "$4"
