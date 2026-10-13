#!/bin/bash
#SBATCH --job-name=taal-recipe-audit
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-recipe-audit-%j.log
#SBATCH --error=logs/taal-recipe-audit-%j.log

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

if (( $# != 5 )) || [[ "$(git rev-parse HEAD)" != "$4" ]]; then
    echo "Usage: sbatch --dependency=afterany:TRAIN_JOB $0 CONFIG CHECKPOINT OUTPUT_DIR EXPECTED_COMMIT EXPECTED_STEP" >&2
    exit 2
fi
# A time-budget final is useful evidence, but not automatically the planned
# matched budget. Refuse a wrong-step snapshot instead of silently comparing it.
python - "$2" "$5" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
expected_step = int(sys.argv[2])
if expected_step < 1:
    raise ValueError('Expected optimizer step must be positive')
metadata = json.loads((checkpoint / 'training_metadata.json').read_text())
if metadata['progress']['step'] != expected_step:
    raise ValueError(f"Matched audit requires step {expected_step}, got {metadata['progress']['step']}")
if metadata['reason'] == 'exception':
    raise ValueError('Refusing an exception checkpoint for a matched audit')
settings = metadata['training_config']
print('MATCHED_RECIPE_AUDIT_PARENT', json.dumps({
    'checkpoint': str(checkpoint), 'step': expected_step,
    'reason': metadata['reason'],
    'gradient_accumulation_steps': settings['training']['gradient_accumulation_steps'],
    'successful_supervised_examples': expected_step * settings['data']['batch_size'] * settings['training']['gradient_accumulation_steps'],
    'skipped_updates': metadata['progress']['skipped_updates'],
}), flush=True)
PY
python -m utils.explorer_preflight --config "$1" --checkpoint "$2"
exec python -m evaluation.taal.state_audit --config "$1" --checkpoint "$2" --output-dir "$3"
