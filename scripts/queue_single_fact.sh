#!/bin/bash
set -euo pipefail
if (( $# < 1 || $# > 2 )) || [[ ! "$1" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "Expected unique batch name and optional training partition (gpu-short or gpu)" >&2
    exit 2
fi
if [[ "$PWD" != /projects/llpr/amiri.ry/dev/fast-weight-memory ]] || [[ "$(git branch --show-current)" != titans-impl ]]; then
    echo "Submit from the canonical attached titans-impl checkout" >&2
    exit 2
fi
training_partition=${2:-gpu-short}
if [[ "$training_partition" != gpu-short && "$training_partition" != gpu ]]; then
    echo "Expected training partition gpu-short or gpu" >&2
    exit 2
fi
# Both partitions have MaxSubmitPU=8; their limits are independently enforced.
short_count=$(squeue -h -u "$USER" -p gpu-short | wc -l)
short_slots=1
if [[ "$training_partition" == gpu-short ]]; then
    short_slots=4
else
    gpu_count=$(squeue -h -u "$USER" -p gpu | wc -l)
    if (( gpu_count + 3 > 8 )); then
        echo "Three training slots needed in gpu; existing jobs left untouched" >&2
        exit 2
    fi
fi
if (( short_count + short_slots > 8 )); then
    echo "$short_slots free slots needed in gpu-short; existing jobs left untouched" >&2
    exit 2
fi
python3 -m scripts.single_fact_preflight
root="output/single-fact-$1"
mkdir "$root"
mkdir -p logs
git rev-parse HEAD > "$root/submitted_commit.txt"
for phase in baselines acquisition narrow mixed; do
    partition=$training_partition
    if [[ "$phase" == baselines ]]; then
        partition=gpu-short
    fi
    options=(--parsable --partition="$partition" --job-name="taal-sf-$phase")
    if [[ "$phase" == narrow || "$phase" == mixed ]]; then
        options+=(--dependency="afterok:$acquisition_job" --kill-on-invalid-dep=yes)
    fi
    job=$(sbatch "${options[@]}" training/sbatch/taal/fs_qwen_single_fact_short.sbatch "$root" "$phase")
    job=${job%%;*}
    if [[ ! "$job" =~ ^[0-9]+$ ]]; then
        echo "Unrecognized submission receipt: $job; inspect jobs.tsv before retrying" >&2
        exit 2
    fi
    printf '%s\t%s\t%s\t%s\n' "$phase" "$job" "$partition" "$root" | tee -a "$root/jobs.tsv"
    if [[ "$phase" == acquisition ]]; then
        acquisition_job=$job
    fi
done
