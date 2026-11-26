#!/bin/bash
#SBATCH --job-name=taal-single-fact
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/taal-single-fact-%j.log
#SBATCH --error=logs/taal-single-fact-%j.log

if (( $# < 1 || $# > 2 )); then
    echo "Expected output directory and optional phase" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" lr3e4
python -m scripts.single_fact_preflight
exec python -m scripts.single_fact --output-dir "$1" --phase "${2:-all}"
