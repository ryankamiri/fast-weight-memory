#!/bin/bash
#SBATCH --job-name=taal-retrieval-val
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-retrieval-val-%j.log
#SBATCH --error=logs/taal-retrieval-val-%j.log

if (( $# != 2 )) || [[ ! "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: sbatch --dependency=afterok:TRAIN_JOB $0 ARM TRAIN_JOB" >&2
    exit 2
fi
source "${SLURM_SUBMIT_DIR}/scripts/explorer_retrieval_env.sh" "$1"
checkpoint="checkpoints/taal-retrieval-$1-$2/final"
output="output/retrieval-acquisition/$1-$2/val"
python -m scripts.retrieval_acquisition --train-config "$train_config" --checkpoint "$checkpoint" --output-dir "$output"
exec python -m evaluation.shared --config "$output/config.yaml" --output-dir "$output"
