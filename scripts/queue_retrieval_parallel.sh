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
if (( count > 1 )); then
    echo "Seven jobs require seven free slots under MaxSubmitPU=8" >&2
    exit 2
fi
python3 -m scripts.retrieval_parallel_preflight
root="output/retrieval-parallel-$1"
mkdir "$root"
mkdir -p logs
git rev-parse HEAD > "$root/submitted_commit.txt"
for arm in baselines diagnostic r3 b3 r3_fp32 s3 r3_eos; do
    job=$(sbatch --parsable --job-name="taal-$arm" training/sbatch/taal/fs_qwen_retrieval_parallel_short.sbatch "$arm" "$root/$arm")
    job=${job%%;*}
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "Unrecognized submission receipt: $job" >&2
        exit 2
    fi
    printf '%s\t%s\t%s\n' "$arm" "$job" "$root/$arm" | tee -a "$root/jobs.tsv"
done
