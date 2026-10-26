#!/bin/bash
#SBATCH --job-name=taal-stability
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/taal-stability-%j.log
#SBATCH --error=logs/taal-stability-%j.log

set -eo pipefail
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate fast-weight-memory
set -u
cd "${SLURM_SUBMIT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/projects/llpr/amiri.ry/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export http_proxy="${http_proxy:-http://10.99.0.130:3128}"
export https_proxy="${https_proxy:-http://10.99.0.130:3128}"
if (( $# != 2 )) || [[ "$(git rev-parse HEAD)" != "$2" ]]; then
    echo "Usage: sbatch $0 GROUP EXPECTED_COMMIT" >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != titans-impl ]] || [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
suite_group="$1"
source_run=output/longmemeval/runaway-11248778
checkpoint=checkpoints/taal-capacity-d-distance-diversity-11021676/final
input_run=checkpoints/taal-norm-input-11329591
output_dir="output/taal/stability-${suite_group}-${SLURM_JOB_ID}"
python -m utils.explorer_preflight --config evaluation/configs/taal/capacity_d_distance_diversity_audit.yaml --checkpoint "$checkpoint"
case "$suite_group" in
  references|write_objective|norm_structure|gate_optimizer)
    exec python -m evaluation.taal.stability_suite --group "$suite_group" --source-run "$source_run" --output-dir "$output_dir"
    ;;
  numerics)
    mkdir -p "$output_dir"
    parity_status=0
    python -m evaluation.taal.stability_precision --source-run "$source_run" --output-dir "$output_dir/precision" || parity_status=$?
    cache_status=0
    python -m evaluation.taal.stability_cache --checkpoint "$checkpoint" --source-run "$source_run" --output-dir "$output_dir/cache" || cache_status=$?
    echo "STABILITY_NUMERICS_EXIT parity=$parity_status cache=$cache_status"
    (( parity_status == 0 && cache_status == 0 ))
    ;;
  length|transitions|input)
    exec python -m evaluation.taal.stability_dynamics --phase "$suite_group" --checkpoint "$checkpoint" --source-run "$source_run" --input-run "$input_run" --output-dir "$output_dir"
    ;;
  *) echo "Unknown group $suite_group" >&2; exit 2 ;;
esac
