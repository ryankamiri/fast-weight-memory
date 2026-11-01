#!/bin/bash
#SBATCH --job-name=taal-cpt
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/taal-cpt-%j.log
#SBATCH --error=logs/taal-cpt-%j.log

set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME=/projects/llpr/amiri.ry/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers" HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
if (( $# != 2 )) || [[ ! "$1" =~ ^[ABCD]$ ]] || [[ "$(git rev-parse HEAD)" != "$2" ]]; then
    echo "Usage: sbatch $0 ARM EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != titans-impl ]] || [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
arm="$1"
export CHECKPOINT_RUN_ID="taal-cpt-${arm}-${SLURM_JOB_ID}"
echo "CPT_SOURCE commit=$2 arm=$arm training=$CHECKPOINT_RUN_ID"
# Discard this model, then start training afresh with the configured seed.
python -m scripts.taal_cpt_preflight "$arm" --gpu-train
status=0
timeout --signal=TERM --kill-after=300 6000 python -m training.train --config "training/configs/taal/cpt_${arm}.yaml" || status=$?
if (( status != 0 && status != 124 )); then
    exit "$status"
fi
python -m scripts.taal_cpt_preflight "$arm" --checkpoint "checkpoints/$CHECKPOINT_RUN_ID/final"
