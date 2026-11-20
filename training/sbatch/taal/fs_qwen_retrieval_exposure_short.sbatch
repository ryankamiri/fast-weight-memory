#!/bin/bash
#SBATCH --job-name=taal-exposure-train
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/taal-exposure-train-%j.log
#SBATCH --error=logs/taal-exposure-train-%j.log

if (( $# != 2 )) || [[ "$1" != repeat && "$1" != broad ]]; then
    echo "Expected repeat|broad and parity receipt directory" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" "$1"
python -m scripts.retrieval_exposure_preflight --parity-dir "$2"
exec python -m scripts.retrieval_training_segment --config "$train_config" --run-id "taal-exposure-$1-$SLURM_JOB_ID"
