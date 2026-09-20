#!/bin/bash
#SBATCH --job-name=fs-qwen-diagnostics
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/fs-qwen-diagnostics-%j.log
#SBATCH --error=logs/fs-qwen-diagnostics-%j.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

if (( $# != 2 )); then
    echo "Usage: sbatch $0 CONFIG OUTPUT_DIR" >&2
    exit 2
fi

exec python -m evaluation.run \
    --config "$1" \
    --output-dir "$2" \
    --diagnostics-only
