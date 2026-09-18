#!/bin/bash
#SBATCH --job-name=fs-qwen-eval-fw-2k-1k-512-a05
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fs-qwen-eval-fw-2k-1k-512-a05-%j.log
#SBATCH --error=logs/fs-qwen-eval-fw-2k-1k-512-a05-%j.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

if (( $# < 1 )); then
    echo "Usage: sbatch $0 CHECKPOINT [OUTPUT_DIR] [extra evaluator arguments]" >&2
    exit 2
fi

exec torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m evaluation.run \
    --config evaluation/configs/longmemeval_fw_swa_2k_1k_512_half_reads.yaml \
    --checkpoint "$1" \
    --output-dir "${2:-output/longmemeval/fw_swa_2k_1k_512_half_reads-${SLURM_JOB_ID}}" "${@:3}"
