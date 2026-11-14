#!/bin/bash
# Run from the canonical Explorer checkout only when submission is authorized.
set -euo pipefail
if (( $# != 0 )); then
    echo "Usage: bash $0" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
mkdir -p logs
for arm in lr3e5 lr1e4 lr3e4 fullcontext; do
    receipt=$(sbatch --parsable training/sbatch/taal/fs_qwen_retrieval_acquisition_short.sbatch "$arm")
    train_job=${receipt%%;*}
    if [[ ! "$train_job" =~ ^[0-9]+$ ]]; then
        echo "Unexpected submission receipt: $receipt; stop and inspect the queue" >&2
        exit 1
    fi
    echo "$arm training=$train_job"
    eval_receipt=$(sbatch --parsable --dependency="afterok:$train_job" --kill-on-invalid-dep=yes evaluation/sbatch/taal/fs_qwen_retrieval_acquisition_short.sbatch "$arm" "$train_job")
    echo "$arm validation=$eval_receipt depends_on=$train_job"
done
