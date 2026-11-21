#!/bin/bash
#SBATCH --job-name=taal-execution-parity
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
#SBATCH --output=logs/taal-execution-parity-%j.log
#SBATCH --error=logs/taal-execution-parity-%j.log

if (( $# != 2 )); then
    echo "Expected ARM OUTPUT" >&2
    exit 2
fi
case "$1" in
    A) recipe=lr3e5 ;;
    B) recipe=lr1e4 ;;
    C) recipe=lr3e4 ;;
    *) echo "Expected A, B or C" >&2; exit 2 ;;
esac
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" "$recipe"
python -m scripts.retrieval_exposure_preflight
exec python -m scripts.retrieval_execution_parity --arm "$1" --output-dir "$2"
