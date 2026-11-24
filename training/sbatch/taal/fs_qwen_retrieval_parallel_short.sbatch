#!/bin/bash
#SBATCH --job-name=taal-parallel
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/taal-parallel-%j.log
#SBATCH --error=logs/taal-parallel-%j.log

if (( $# != 2 )); then
    echo "Expected branch and output directory" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" lr3e4
python -m scripts.retrieval_parallel_preflight
exec python -m scripts.retrieval_parallel --arm "$1" --output-dir "$2"
