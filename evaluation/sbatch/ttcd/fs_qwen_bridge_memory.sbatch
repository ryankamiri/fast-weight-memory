#!/bin/bash
#SBATCH --job-name=fs-qwen-bridge-memory
#SBATCH --partition=gpu-short
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fs-qwen-bridge-memory-%A-%a.log
#SBATCH --error=logs/fs-qwen-bridge-memory-%A-%a.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

if (( $# < 2 )); then
    echo "Usage: sbatch $0 TTCD_CHECKPOINT SWA_CHECKPOINT [OUTPUT_ROOT]" >&2
    exit 2
fi

fw_checkpoint="$1"
swa_checkpoint="$2"
output_root="${3:-output/bridge-memory/${SLURM_ARRAY_JOB_ID}}"

case "${SLURM_ARRAY_TASK_ID}" in
    0) system="base_full"; checkpoint=() ;;
    1) system="swa"; checkpoint=(--checkpoint "${swa_checkpoint}") ;;
    2) system="fw_0"; checkpoint=(--checkpoint "${fw_checkpoint}") ;;
    3) system="fw_0_5"; checkpoint=(--checkpoint "${fw_checkpoint}") ;;
    4) system="fw_1"; checkpoint=(--checkpoint "${fw_checkpoint}") ;;
    *) echo "Unexpected array task: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

exec torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m evaluation.bridge_memory \
    --config evaluation/configs/ttcd/bridge_memory.yaml \
    --system "${system}" \
    "${checkpoint[@]}" \
    --output-dir "${output_root}/${system}"
