#!/bin/bash
#SBATCH --job-name=taal-cpt-eval
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-cpt-eval-%j.log
#SBATCH --error=logs/taal-cpt-eval-%j.log

set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export HF_HOME=/projects/llpr/amiri.ry/huggingface
export TRANSFORMERS_CACHE="${HF_HOME}/transformers" HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy=http://10.99.0.130:3128 https_proxy=http://10.99.0.130:3128
if (( $# != 3 )) || [[ ! "$1" =~ ^[ABCD]$ ]] || [[ ! "$2" =~ ^[0-9]+$ ]] || [[ "$(git rev-parse HEAD)" != "$3" ]]; then
    echo "Usage: sbatch $0 ARM TRAINING_JOB EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != titans-impl ]] || [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
arm="$1"
checkpoint="checkpoints/taal-cpt-${arm}-$2/final"
output_dir="output/taal/cpt-${arm}-$2-audit-${SLURM_JOB_ID}"
python -m scripts.taal_cpt_preflight "$arm" --checkpoint "$checkpoint"
mkdir -p "$output_dir"
cp "$checkpoint/training_metadata.json" "$output_dir/training_metadata.json"
echo "CPT_AUDIT_SOURCE commit=$3 arm=$arm training_job=$2"
python -m evaluation.taal.cpt_audit --config evaluation/configs/taal/cpt_retrieval.yaml --checkpoint "$checkpoint" --output-dir "$output_dir/retrieval"
python -m evaluation.taal.cpt_trace --checkpoint "$checkpoint" --dataset "data/taal_cpt/${arm}.jsonl" --output-dir "$output_dir/documents"
