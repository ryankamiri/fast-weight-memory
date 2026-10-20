#!/bin/bash
#SBATCH --job-name=taal-d-normalization
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-d-normalization-%j.log
#SBATCH --error=logs/taal-d-normalization-%j.log

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
    echo "Usage: sbatch $0 CHECKPOINT SOURCE_RUN baseline|input|mlp|both" >&2
    exit 2
fi
case "$3" in
    baseline|input|mlp|both) ;;
    *) echo "Unknown normalization arm: $3" >&2; exit 2 ;;
esac
python -m utils.explorer_preflight --config evaluation/configs/taal/longmemeval_smoke.yaml --checkpoint "$1"
exec python -m evaluation.taal.normalization_ablation --checkpoint "$1" \
    --source-run "$2" --arm "$3" \
    --output-dir "output/longmemeval/normalization-${3}-${SLURM_JOB_ID}"
