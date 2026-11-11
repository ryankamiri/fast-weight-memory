#!/bin/bash
#SBATCH --job-name=taal-workshop-train
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/taal-workshop-train-%j.log
#SBATCH --error=logs/taal-workshop-train-%j.log

# The excluded gpu-short nodes contain V100/T4; A100 and H200 are eligible.
set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME=/projects/llpr/amiri.ry/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers" HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
if (( $# != 1 )) || [[ ! "$1" =~ ^(narrow|diverse)$ ]]; then
    echo "Usage: sbatch $0 narrow|diverse" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
export CHECKPOINT_RUN_ID="taal-workshop-$1-${SLURM_JOB_ID}"
# Record the code actually executed; queue-time revisions must not block compatible pulls.
echo "WORKSHOP_TRAIN_SOURCE commit=$(git rev-parse HEAD) arm=$1 run=$CHECKPOINT_RUN_ID"
# This model/update is discarded. Training reloads the exact D449 parent.
python -m scripts.taal_workshop preflight --arm "$1"
exec python -m training.train --config "training/configs/taal/workshop_$1.yaml"
