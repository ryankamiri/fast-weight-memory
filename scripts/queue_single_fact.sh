#!/bin/bash
set -euo pipefail
if (( $# != 1 )) || [[ ! "$1" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Expected one unique batch name" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Submit from the canonical attached titans-impl checkout" >&2
    exit 2
fi
count=$(squeue -h -u "$USER" | wc -l)
if (( count >= 8 )); then
    echo "No free slot under the observed MaxSubmitPU=8; existing jobs left untouched" >&2
    exit 2
fi
python3 -m scripts.single_fact_preflight
root="output/single-fact-$1"
mkdir "$root"
mkdir -p logs
git rev-parse HEAD > "$root/submitted_commit.txt"
job=$(sbatch --parsable training/sbatch/taal/fs_qwen_single_fact_short.sbatch "$root")
job=${job%%;*}
if [[ ! "$job" =~ ^[0-9]+$ ]]; then
    echo "Unrecognized submission receipt: $job" >&2
    exit 2
fi
printf '%s\t%s\n' "$job" "$root" | tee "$root/jobs.tsv"
