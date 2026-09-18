#!/bin/bash
#SBATCH --job-name=fs-qwen-eval-base_swa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fs-qwen-eval-base_swa-%j.log
#SBATCH --error=logs/fs-qwen-eval-base_swa-%j.log

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u

cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# The YAML selects the untouched base model. Optional arguments:
# OUTPUT_DIR, followed by --generate-only or --judge-only.

exec torchrun --standalone --nnodes=1 --nproc-per-node=1 \
    -m evaluation.run \
    --config evaluation/configs/longmemeval_base_swa.yaml \
    --output-dir "${1:-output/longmemeval/base_swa-${SLURM_JOB_ID}}" "${@:2}"
