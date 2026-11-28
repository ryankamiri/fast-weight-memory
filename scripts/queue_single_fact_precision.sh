#!/bin/bash
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Expected one unique FP32 batch name" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Submit from the canonical attached titans-impl checkout" >&2
    exit 2
fi
count=$(squeue -h -u "$USER" -p gpu-short | wc -l)
if (( count + 2 > 8 )); then
    echo "Two free gpu-short slots needed; existing jobs left untouched" >&2
    exit 2
fi
python3 -m scripts.single_fact_preflight --fp32
root="output/single-fact-fp32-$1"
mkdir "$root"
mkdir -p logs
git rev-parse HEAD > "$root/submitted_commit.txt"
for phase in train audit; do
    options=(--parsable --job-name="taal-sf-fp32-$phase")
    if [[ "$phase" == audit ]]; then
        options+=(--dependency="afterok:$train_job" --kill-on-invalid-dep=yes)
    fi
    job=$(sbatch "${options[@]}" training/sbatch/taal/fs_qwen_single_fact_precision_short.sbatch "$root" "$phase")
    job=${job%%;*}
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "Unrecognized receipt: $job; inspect jobs.tsv before retrying" >&2
        exit 2
    fi
    printf '%s\t%s\t%s\n' "$phase" "$job" "$root" | tee -a "$root/jobs.tsv"
    if [[ "$phase" == train ]]; then
        train_job=$job
    fi
done
