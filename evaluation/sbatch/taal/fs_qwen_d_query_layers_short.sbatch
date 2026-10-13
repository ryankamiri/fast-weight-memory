#!/bin/bash
#SBATCH --job-name=taal-d-query-layers
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-d-query-layers-%j.log
#SBATCH --error=logs/taal-d-query-layers-%j.log

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

if (( $# != 3 )) || [[ "$(git rev-parse HEAD)" != "$3" ]]; then
    echo "Usage: sbatch $0 D_FINAL_CHECKPOINT OUTPUT_ROOT EXPECTED_COMMIT" >&2
    exit 2
fi

# Both audits reuse the existing successful D, not a new/partial recipe model.
python - "$1" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
metadata = json.loads((checkpoint / 'training_metadata.json').read_text())
config = json.loads((checkpoint / 'config.json').read_text())
if metadata['progress']['step'] != 449 or metadata['reason'] == 'exception':
    raise ValueError('D query-layer audits require the verified final step-449 checkpoint')
if (config['num_hidden_layers'], config['working_memory_size'], config['memory_chunk_size']) != (28, 32, 1):
    raise ValueError('D query-layer audits require 28 layers, KV32 and memory chunk1')
print('D_QUERY_LAYER_AUDIT_PARENT', json.dumps({
    'checkpoint': str(checkpoint.resolve()), 'step': 449,
    'reason': metadata['reason'], 'layers': 28,
}), flush=True)
PY

# Bundle the two short inference-only setups in one allocation. Attempt both,
# but return failure if either fails; never silently call a partial bundle done.
audit_status=0
for ablation in l27 early; do
    config="evaluation/configs/taal/d_${ablation}_query_reads_off.yaml"
    output="$2/${ablation}"
    if python -m utils.explorer_preflight --config "$config" --checkpoint "$1"; then
        python -m evaluation.taal.state_audit --config "$config" --checkpoint "$1" --output-dir "$output" || audit_status=$?
    else
        audit_status=$?
    fi
done
exit "$audit_status"
