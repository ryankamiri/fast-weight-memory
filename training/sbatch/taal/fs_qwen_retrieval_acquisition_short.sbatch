#!/bin/bash
#SBATCH --job-name=taal-retrieval-train
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
#SBATCH --output=logs/taal-retrieval-train-%j.log
#SBATCH --error=logs/taal-retrieval-train-%j.log

# Optional root job preserves checkpoint identity across a replacement job.
if (( $# < 1 || $# > 2 )) || [[ ! "${2:-0}" =~ ^[0-9]+$ ]]; then
    echo "Usage: sbatch $0 ARM [ROOT_TRAIN_JOB]" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" "$1"
root_job=${2:-$SLURM_JOB_ID}
if (( $# == 2 )) && [[ ! -f "checkpoints/taal-retrieval-$1-$root_job/final/trainer_state.pt" ]]; then
    echo "Requested resume checkpoint is missing; refusing fresh training" >&2
    exit 2
fi
exec python -m scripts.retrieval_training_segment --config "$train_config" --run-id "taal-retrieval-$1-$root_job"
