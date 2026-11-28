#!/bin/bash
#SBATCH --job-name=taal-sf-bf16-fp32
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
#SBATCH --output=logs/taal-single-fact-cross-%j.log
#SBATCH --error=logs/taal-single-fact-cross-%j.log

if (( $# != 3 )) || [[ ! "$3" =~ ^[0-9]+$ ]]; then
    echo "Expected output directory, parent output and acquisition job ID" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" lr3e4
python -m scripts.single_fact_preflight --fp32 --parent-output "$2" --parent-job "$3"
exec python -m scripts.single_fact_cross_precision --output-dir "$1" --parent-output "$2" --parent-job "$3"
