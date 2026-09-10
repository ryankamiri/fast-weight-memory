# Fast-Weight Memory

Fast-weight memory for dense Qwen3 models, built with PyTorch.

Selected layers compare attention over long and short context windows, then use
the difference to update temporary MLP weights in chunks. The model keeps Qwen's
pretrained weights and resets temporary memory between training examples.

## Setup

Run from the repository root with uv:

```bash
uv sync --frozen
uv run hf auth login
uv run wandb login
```

## Train

```bash
uv run --frozen --no-dev torchrun --standalone --nproc-per-node=1 \
  -m training.train --config training/configs/qwen3_0_6b.yaml
```

## Configure

Edit [training/configs/qwen3_0_6b.yaml](training/configs/qwen3_0_6b.yaml):

- `model`: Qwen model, fast-weight layers, context windows, and chunk size.
- `data`: Hugging Face dataset, train/validation ranges, batch size, and sequence length.
- `optimizer` / `scheduler`: learning rate, warmup, and cosine decay.
- `training`: update limit, gradient accumulation, seed, and validation frequency.
- `wandb`: project, account, and run name. Slurm job IDs are appended automatically.
- `checkpoints`: output directory and switches for best-validation and final model saves.

Models are saved to `checkpoints/<wandb-run-id>/best` and `final`. Best is replaced
only when validation loss improves. Saves contain model weights, model config,
and training metadata, not optimizer state or temporary per-book memory.
Cancellation skips final validation and attempts a final save. Slurm requests a
five-minute warning before timeout; forced kills cannot guarantee saving.

## Evaluate

Set `OPENAI_API_KEY` in `.env` using `.env.example` as a template. Prepare the
evaluation dataset as described below.

Submit from the repo root on Explorer, replacing each run ID with its matching
checkpoint:

```bash
mkdir -p logs
sbatch evaluation/sbatch/fs_qwen_eval_full.sbatch checkpoints/FULL_RUN/best
sbatch evaluation/sbatch/fs_qwen_eval_swa.sbatch checkpoints/SWA_RUN/best
sbatch evaluation/sbatch/fs_qwen_eval_fw_swa.sbatch checkpoints/FW_RUN/best
```

Edit `evaluation/configs/longmemeval_{full,swa,fw_swa}.yaml` for dataset variant,
generation settings, and judge limits. Each job generates answers, then grades
them with Luna through paid API calls. Results go to `output/longmemeval/<mode>/`
and logs to `logs/`. Rerun the same command to resume with an unchanged checkpoint
and results directory.

## Prepare data

Training uses [Qwen-tokenized ProLong](https://huggingface.co/datasets/ryankamiri/prolong-qwen).
To rebuild it, decode the source tokens and re-tokenize for Qwen with:

```bash
uv run python -m scripts.prepare_prolong --repo-id ryankamiri/prolong-qwen
```

Conversion writes local Parquet shards, then uploads to Hugging Face. Rerun the
same command to resume from completed shards.

Evaluation uses LongMemEval with Qwen-tokenized prompts. Preparation preserves
all original fields and adds token IDs and lengths without truncating histories:

```bash
uv run python -m scripts.prepare_longmemeval --variant oracle --upload
```

Use `--variant s` or `--variant m` to prepare the other variants. Files are saved
under `datasets/longmemeval-qwen/<variant>/` and uploaded to
`ryankamiri/longmemeval-qwen`. Omit `--upload` to prepare locally only. Set
`HF_TOKEN` in `.env` or use `hf auth login` for uploads.
