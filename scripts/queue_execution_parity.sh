#!/bin/bash
# Three independent diagnostics; no training or training-release dependencies.
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Usage: bash $0 BATCH_NAME" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
queued=$(squeue -h -u "$USER" -o '%i' | wc -l)
if (( queued > 5 )); then
    echo "Three free submission slots are required; inspect the existing queue" >&2
    exit 2
fi
python3 -m scripts.retrieval_exposure_preflight
root="output/retrieval-execution-parity-$1"
mkdir -p logs
# Never silently duplicate a partly submitted batch.
mkdir "$root"
git rev-parse HEAD > "$root/submitted-source.txt"
for arm in A B C; do
    receipt=$(sbatch --parsable --job-name="taal-path-$arm" \
        evaluation/sbatch/taal/fs_qwen_execution_parity_short.sbatch "$arm" "$root/$arm")
    job=${receipt%%;*}
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "Unexpected Slurm receipt: $receipt; inspect before retrying" >&2
        exit 1
    fi
    printf '%s\t%s\n' "$arm" "$job" | tee -a "$root/jobs.tsv"
done
