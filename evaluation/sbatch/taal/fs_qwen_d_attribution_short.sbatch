#!/bin/bash
#SBATCH --job-name=taal-d-attribution
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/taal-d-attribution-%j.log
#SBATCH --error=logs/taal-d-attribution-%j.log

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

if (( $# != 4 )) || [[ "$(git rev-parse HEAD)" != "$4" ]]; then
    echo "Usage: sbatch $0 {layer_sources|read_cross} D_FINAL_CHECKPOINT OUTPUT_DIR EXPECTED_COMMIT" >&2
    exit 2
fi
case "$1" in
    layer_sources|read_cross) config="evaluation/configs/taal/d_${1}.yaml" ;;
    *) echo "Unknown D attribution experiment: $1" >&2; exit 2 ;;
esac

python - "$2" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

from evaluation.taal.prefix_store import checkpoint_fingerprint

checkpoint = Path(sys.argv[1])
metadata = json.loads((checkpoint / 'training_metadata.json').read_text())
config = json.loads((checkpoint / 'config.json').read_text())
if metadata['progress']['step'] != 449 or metadata['reason'] == 'exception':
    raise ValueError('Attribution requires the original successful D final449')
if (config['num_hidden_layers'], config['working_memory_size'], config['memory_chunk_size']) != (28, 32, 1):
    raise ValueError('Attribution requires all28 layers, KV32 and memory chunk1')
fingerprint = checkpoint_fingerprint(checkpoint)
if fingerprint != '285dab0ad8f638d95caefb283aeb67598ca959105974e57453a8e925a4359100':
    raise ValueError('Checkpoint differs from the verified D final449')
dataset = hashlib.sha256(Path('data/taal_capacity_16_gap192.jsonl').read_bytes()).hexdigest()
if dataset != '685041cd94587585da055371c0460576637f0d23f4169b6b53dcfcf4071e4d93':
    raise ValueError('Dataset differs from the verified16 paired gap192 episodes')
print('D_ATTRIBUTION_PARENT', json.dumps({
    'checkpoint': str(checkpoint.resolve()), 'step': 449,
    'checkpoint_sha256': fingerprint, 'dataset_sha256': dataset,
}), flush=True)
PY

python -m utils.explorer_preflight --config "$config" --checkpoint "$2"
python -m evaluation.taal.state_audit --config "$config" --checkpoint "$2" --output-dir "$3"
