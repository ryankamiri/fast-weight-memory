#!/bin/bash
#SBATCH --job-name=taal-exposure-audit
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/taal-exposure-audit-%j.log
#SBATCH --error=logs/taal-exposure-audit-%j.log

if (( $# < 3 || $# > 4 )); then
    echo "Expected PHASE ARM OUTPUT [TRAIN_JOB]" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" lr1e4
python -m scripts.retrieval_exposure_preflight
args=(--phase "$1" --output-dir "$3")
if [[ "$1" != parity ]]; then args+=(--arm "$2"); fi
if (( $# == 4 )); then args+=(--train-job "$4"); fi
exec python -m scripts.retrieval_exposure_audit "${args[@]}"
