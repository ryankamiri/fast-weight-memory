#!/bin/bash
# Two independent training jobs, each followed by its own complete audit.
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
if (( queued > 4 )); then
    echo "Four free submission slots are required; inspect existing jobs" >&2
    exit 2
fi
python3 -m scripts.retrieval_exposure_preflight
root="output/retrieval-exposure-$1"
mkdir -p logs
# An existing receipt directory is never silently reused: prevent duplicate submissions.
mkdir "$root"
git rev-parse HEAD > "$root/submitted-source.txt"
submit() {
    local label=$1 receipt
    shift
    receipt=$(sbatch --parsable "$@")
    submitted_job=${receipt%%;*}
    if [[ ! "$submitted_job" =~ ^[0-9]+$ ]]; then
        echo "Unexpected Slurm receipt: $receipt; inspect partial receipts before retrying" >&2
        exit 1
    fi
    printf '%s\t%s\n' "$label" "$submitted_job" | tee -a "$root/jobs.tsv"
}
audit=evaluation/sbatch/taal/fs_qwen_retrieval_exposure_short.sbatch
train=training/sbatch/taal/fs_qwen_retrieval_exposure_short.sbatch
# Acquisition and CUDA execution parity are separate experiments. Keep the
# diagnostics running independently; only each completed training budget gates
# its own audit. The audit records full-sequence scores and cached generation.
for arm in repeat broad; do
    submit "train-$arm" --job-name="taal-$arm" "$train" "$arm"
    train_job=$submitted_job
    submit "audit-$arm" --job-name="taal-eval-$arm" --dependency="afterok:$train_job" --kill-on-invalid-dep=yes "$audit" audit "$arm" "$root/audit-$arm" "$train_job"
done
