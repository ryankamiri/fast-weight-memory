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

## Prepare data

Training uses [Qwen-tokenized ProLong](https://huggingface.co/datasets/ryankamiri/prolong-qwen).
To rebuild it, decode the source tokens and re-tokenize for Qwen with:

```bash
uv run python -m scripts.prepare_prolong --repo-id your-account/prolong-qwen
```

Conversion writes local Parquet shards, then uploads to Hugging Face. Rerun the
same command to resume from completed shards.
