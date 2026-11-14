#!/bin/bash
# Source this from the acquisition launchers; no model smoke test runs here.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME=/projects/llpr/amiri.ry/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers" HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
case "${1:-}" in
    lr3e5|lr3e4|fullcontext) train_config="training/configs/taal/retrieval_acquisition_$1.yaml" ;;
    lr1e4) train_config=training/configs/taal/retrieval_acquisition.yaml ;;
    *) echo "Expected arm: lr3e5, lr1e4, lr3e4 or fullcontext" >&2; exit 2 ;;
esac
echo "RETRIEVAL_SOURCE commit=$(git rev-parse HEAD) arm=$1 config=$train_config"
