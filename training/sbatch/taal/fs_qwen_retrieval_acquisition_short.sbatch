#!/bin/bash
#SBATCH --job-name=taal-retrieval-train
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/taal-retrieval-train-%j.log
#SBATCH --error=logs/taal-retrieval-train-%j.log

# Reuse the workshop's node exclusions for A100/H200 eligibility.
if (( $# != 1 )); then
    echo "Usage: sbatch $0 lr3e5|lr1e4|lr3e4" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" "$1"
export CHECKPOINT_RUN_ID="taal-retrieval-$1-${SLURM_JOB_ID}"
exec python -m training.train --config "$train_config"
