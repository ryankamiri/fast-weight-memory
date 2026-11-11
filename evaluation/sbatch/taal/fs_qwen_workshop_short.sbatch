#!/bin/bash
#SBATCH --job-name=taal-workshop-eval
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=c[2204-2207],d[1002,1007,1009-1013,1015,1017,1019-1020,1022,1025,1027]
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-workshop-eval-%j.log
#SBATCH --error=logs/taal-workshop-eval-%j.log

set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME=/projects/llpr/amiri.ry/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers" HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
if (( $# < 2 )) || [[ "$(git rev-parse HEAD)" != "$2" ]]; then
    echo "Usage: sbatch $0 baseline EXPECTED_COMMIT | matched EXPECTED_COMMIT NARROW_RUN DIVERSE_RUN" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
echo "WORKSHOP_AUDIT_SOURCE commit=$2 mode=$1"
output="output/taal/workshop-$1-${SLURM_JOB_ID}"
if [[ "$1" == baseline ]] && (( $# == 2 )); then
    exec python -m scripts.taal_workshop baseline --output-dir "$output"
elif [[ "$1" == matched ]] && (( $# == 4 )); then
    # afterany dependencies allow valid saved checkpoints after graceful timeout.
    # The runner refuses missing or unequal-step snapshots; skipped updates remain logged.
    exec python -m scripts.taal_workshop matched --narrow-run "$3" --diverse-run "$4" --output-dir "$output"
else
    echo "Invalid workshop audit arguments" >&2
    exit 2
fi
