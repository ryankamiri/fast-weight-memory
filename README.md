# Fast-Weight Memory

**Persistent neural memory for language models with bounded context.**

Fast-Weight Memory explores how a pretrained language model can continue learning
within a conversation without retaining an ever-growing KV cache. The model uses
a sliding context window as working memory and carries a fixed-size learned state
between blocks as long-term memory.

The project currently extends dense Qwen3 models with two memory architectures:

- **TTCD** learns temporary fast weights from the difference between a
  longer-context teacher and a shorter-context student.
- **TaaL** adds a Titans-style neural-memory module to each decoder layer and
  updates that memory online from the incoming sequence.

Both architectures are built around the same goal: information that leaves the
KV cache should still be able to affect later predictions.

## How it works

```text
incoming tokens
      │
      ▼
bounded sliding-window KV cache ─── exact recent context
      │
      ▼
Qwen3 decoder + persistent memory ── compressed state across windows
      │
      ▼
next-token prediction
```

The KV cache remains the model's precise working memory. The added memory state
is bounded independently of conversation length and is updated as the sequence
is processed.

| Architecture | Memory state | Write mechanism | Read mechanism |
| --- | --- | --- | --- |
| **TTCD** | Fast MLP weights at selected decoder layers | Teacher-student hidden-state corrections are accumulated and committed in chunks | The fast-weight output is added to the student representation |
| **TaaL** | A neural-memory MLP at each decoder layer | Reconstruction surprise drives learned updates with momentum and forgetting | The current hidden state queries memory and receives a residual correction |

## Installation

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/ryankamiri/fast-weight-memory.git
cd fast-weight-memory
uv sync --frozen
```

Authenticate with Hugging Face and Weights & Biases when training:

```bash
uv run hf auth login
uv run wandb login
```

## Training

All training recipes use the same typed entry point. The `model.architecture`
field in each YAML file selects TTCD or TaaL.

### Continual pretraining

[ProLong](https://huggingface.co/datasets/ryankamiri/prolong-qwen) provides
natural long-form sequences for ordinary next-token continual pretraining.

```bash
uv run --frozen --no-dev torchrun --standalone --nproc-per-node=1 \
  -m training.train \
  --config training/configs/ttcd/qwen3_0_6b.yaml
```

### Delayed-recall pilot

The bridge-memory dataset places a fact before the working-memory boundary and
asks for it after its original K/V vectors have been evicted. This provides a
controlled test of whether the added memory pathway can carry information across
windows.

```bash
uv run --frozen --no-dev torchrun --standalone --nproc-per-node=1 \
  -m training.train \
  --config training/configs/taal/qwen3_0_6b_delayed_recall_overfit.yaml
```

The initial TaaL recipe is intentionally a small overfitting test. It verifies
that the memory pathway can learn delayed recall before scaling to held-out facts,
paraphrases, and natural conversations.

### Configuration

Architecture-specific recipes live in:

- `training/configs/ttcd/`
- `training/configs/taal/`

Each recipe configures the base model, memory geometry, dataset, loss,
optimization, validation ablations, checkpointing, and experiment tracking.
Checkpoints are written to `checkpoints/<wandb-run-id>/{best,final}`.

## Datasets

The two training datasets answer different research questions:

| Dataset | Purpose | Objective |
| --- | --- | --- |
| [ProLong](https://huggingface.co/datasets/ryankamiri/prolong-qwen) | Train memory-augmented models on natural long-form text | Next-token prediction over the sequence |
| [Bridge memory](https://huggingface.co/datasets/ryankamiri/ttcd-bridge-memory) | Isolate recall after information leaves the KV cache | Delayed-answer prediction under visible, bridge, and no-bridge conditions |

Rebuild the Qwen-tokenized ProLong dataset with:

```bash
uv run python -m scripts.prepare_prolong \
  --repo-id ryankamiri/prolong-qwen
```

Generate a bridge-memory configuration with custom window geometry:

```bash
uv run python -m scripts.prepare_ttcd_bridge \
  --teacher-window-size 4096 \
  --student-window-size 2048 \
  --chunk-size 1024
```

This creates a Hugging Face dataset configuration named
`t4096-s2048-c1024`. The geometry controls where facts, bridge mentions, and
queries appear; the resulting records can be used by either memory architecture.

## Evaluation

The evaluation suite supports:

- LongMemEval generation and grading;
- full-attention and sliding-window Qwen baselines;
- memory-on, half-strength, and memory-ablated comparisons;
- OpenAI or [Jev](https://docs.typesafe.ai/introduction) judging; and
- resumable prediction and offline judging passes.

Set the relevant judge key in `.env` using `.env.example` as a template:

```bash
OPENAI_API_KEY=...
TYPESAFE_API_KEY=...
```

Prepare LongMemEval:

```bash
uv run python -m scripts.prepare_longmemeval --variant oracle --upload
```

Example Explorer submissions:

```bash
mkdir -p logs
sbatch evaluation/sbatch/ttcd/fs_qwen_eval_full.sbatch checkpoints/FULL_RUN/best
sbatch evaluation/sbatch/ttcd/fs_qwen_eval_swa.sbatch checkpoints/SWA_RUN/best
sbatch evaluation/sbatch/ttcd/fs_qwen_eval_fw_swa.sbatch checkpoints/TTCD_RUN/best
```

Predictions and judgments are written under `output/longmemeval/`. Saved
predictions can be graded again without rerunning generation.

## Research status

This repository is an active research prototype, not a production memory system.
TTCD established useful memory-dependent computation in some settings, but did
not reliably recover completely evicted evidence. TaaL is the next architecture
under evaluation, beginning with a controlled delayed-recall pilot before larger
continual-training runs.

The central evaluation standard is stricter than aggregate language-model loss:
a memory architecture should improve later predictions over the same checkpoint
with memory reads disabled, especially after the relevant K/V state has left the
working-memory window.

## Repository layout

```text
architectures/
  shared/       Shared bounded-cache and convolution components
  ttcd/         Teacher-student fast-weight architecture
  titans/       Online neural-memory implementation
  taal/         Titans-as-a-Layer Qwen integration
training/       Typed configs, dataloaders, trainer, and Slurm launchers
evaluation/     LongMemEval and bridge-memory evaluation tools
scripts/        Dataset preparation utilities
tests/          Architecture, state, training, and evaluation tests
```
