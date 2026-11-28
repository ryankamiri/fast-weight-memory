#!/bin/bash
set -euo pipefail
if (( $# != 3 )) || [[ ! "$2" =~ ^[0-9]+$ || ! "$3" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Expected parent output directory, acquisition job ID and unique batch name" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Submit from the canonical attached titans-impl checkout" >&2
    exit 2
fi
count=$(squeue -h -u "$USER" -p gpu-short | wc -l)
if (( count + 1 > 8 )); then
    echo "One free gpu-short slot needed; existing jobs left untouched" >&2
    exit 2
fi
python3 -m scripts.single_fact_preflight --fp32 --parent-output "$1" --parent-job "$2"
root="output/single-fact-cross-$3"
mkdir "$root"
mkdir -p logs
git rev-parse HEAD > "$root/submitted_commit.txt"
job=$(sbatch --parsable --job-name=taal-sf-bf16-fp32 --dependency="afterok:$2" --kill-on-invalid-dep=yes \
    training/sbatch/taal/fs_qwen_single_fact_cross_precision_short.sbatch "$root" "$1" "$2")
job=${job%%;*}
if [[ ! "$job" =~ ^[0-9]+$ ]]; then
    echo "Unrecognized receipt: $job; inspect the scheduler before retrying" >&2
    exit 2
fi
printf 'audit\t%s\t%s\t%s\t%s\n' "$job" "$root" "$1" "$2" | tee "$root/jobs.tsv"
