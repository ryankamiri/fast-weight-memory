#!/bin/bash
# Run from the canonical Explorer checkout only when submission is authorized.
set -euo pipefail
if (( $# > 1 )) || [[ ! "${1:-0}" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 [LR3E5_ROOT_TRAIN_JOB]" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Expected canonical attached titans-impl checkout" >&2
    exit 2
fi
mkdir -p logs
for arm in lr3e5 lr1e4 lr3e4 fullcontext; do
    submit=(sbatch --parsable training/sbatch/taal/fs_qwen_retrieval_acquisition_short.sbatch "$arm")
    if [[ "$arm" == lr3e5 && $# == 1 ]]; then
        submit+=("$1")
    fi
    receipt=$("${submit[@]}")
    train_job=${receipt%%;*}
    if [[ ! "$train_job" =~ ^[0-9]+$ ]]; then
        echo "Unexpected submission receipt: $receipt; stop and inspect the queue" >&2
        exit 1
    fi
    root_job=$train_job
    if [[ "$arm" == lr3e5 && $# == 1 ]]; then
        root_job=$1
    fi
    echo "$arm training=$train_job checkpoint_root=$root_job"
    eval_receipt=$(sbatch --parsable --dependency="afterok:$train_job" --kill-on-invalid-dep=yes evaluation/sbatch/taal/fs_qwen_retrieval_acquisition_short.sbatch "$arm" "$root_job")
    echo "$arm validation=$eval_receipt depends_on=$train_job"
done
