"""Training and validation loops; fast-weight session state stays inside each call."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Callable

import torch

from .config import TrainingConfig


@dataclass
class Progress:
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    skipped_updates: int = 0
    nan_losses: int = 0
    nan_gradients: int = 0
    infinite_losses: int = 0
    infinite_gradients: int = 0

    def metrics(self):
        return {
            "train/step": self.step,
            "train/epoch": self.epoch,
            "train/tokens_seen": self.tokens_seen,
            "train/skipped_updates": self.skipped_updates,
            "train/nan_losses": self.nan_losses,
            "train/nan_gradients": self.nan_gradients,
            "train/infinite_losses": self.infinite_losses,
            "train/infinite_gradients": self.infinite_gradients,
        }


@dataclass
class Accumulation:
    batches: int = 0
    prediction_tokens: int = 0
    input_tokens: int = 0
    loss_sum: float = 0.0


def precision_context(device: torch.device):
    # CPU is supported for small unit tests, not for the actual training command.
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def perplexity(loss: float) -> float:
    if math.isnan(loss):
        return math.nan
    return math.exp(loss) if loss < 700 else math.inf


def prediction_count(labels: torch.Tensor) -> int:
    # HF shifts the labels internally; position zero is never a prediction target.
    return int((labels[:, 1:] != -100).sum().item())


def build_scheduler(optimizer, config: TrainingConfig):
    warmup = config.scheduler.warmup_steps
    duration = config.scheduler.schedule_steps
    minimum = config.scheduler.final_lr / config.optimizer.lr

    def lr_multiplier(completed_steps):
        # LambdaLR configures the LR for the next optimizer update.
        step = completed_steps + 1
        if step <= warmup:
            return step / warmup
        fraction = min((step - warmup) / (duration - warmup), 1.0)
        return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * fraction))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


@torch.no_grad()
def validate(model, dataloader, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    tokens = 0
    records = 0
    try:
        for batch in dataloader:
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            count = prediction_count(batch["labels"])
            if count == 0:
                raise ValueError("Validation batch has no next-token targets")
            with precision_context(device):
                output = model(**batch, state=None, use_cache=False)
            loss_sum += output.loss.item() * count
            tokens += count
            records += batch["input_ids"].shape[0]
            del output, batch
    finally:
        model.train(was_training)
    if tokens == 0:
        raise ValueError("Validation range has no records matching seq_len")
    loss = loss_sum / tokens
    return {"val/loss": loss, "val/perplexity": perplexity(loss), "val/records": records}


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    config: TrainingConfig,
    device: torch.device,
    log: Callable[[dict], None],
    should_stop: Callable[[], bool],
) -> Progress:
    """Repeat epochs until max_steps successful updates or an external stop."""
    progress = Progress()
    group = Accumulation()
    model.train()
    optimizer.zero_grad()
    group_started = time.perf_counter()
    last_validation_step = None

    while progress.step < config.training.max_steps and not should_stop():
        train_loader.dataset.set_epoch(progress.epoch)
        progress.epoch += 1
        epoch_batches = 0

        for batch in train_loader:
            if should_stop():
                break
            epoch_batches += 1
            batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
            count = prediction_count(batch["labels"])
            if count == 0:
                raise ValueError("Training batch has no next-token targets")
            B, S = batch["input_ids"].shape
            input_tokens = B * S
            progress.tokens_seen += input_tokens

            with precision_context(device):
                output = model(**batch, state=None, use_cache=False)
                loss = output.loss
            # Do not retain the logits or returned fast-weight state across calls.
            del output, batch
            loss_value = loss.detach().item()
            # we have a nan in the loss
            if not math.isfinite(loss_value):
                progress.nan_losses += int(math.isnan(loss_value))
                progress.infinite_losses += int(math.isinf(loss_value))
                progress.skipped_updates += 1
                optimizer.zero_grad()
                group = Accumulation()
                del loss
                log(progress.metrics())
                group_started = time.perf_counter()
                continue

            # Match TTCD: average the microbatch gradients over this update.
            (loss / config.training.gradient_accumulation_steps).backward()
            del loss
            group.batches += 1
            group.prediction_tokens += count
            group.input_tokens += input_tokens
            group.loss_sum += loss_value * count
            if group.batches < config.training.gradient_accumulation_steps:
                # don't backprob yet
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.max_grad_norm,
            ).item()
            # we have a nan in the grad
            if not math.isfinite(grad_norm):
                progress.nan_gradients += int(math.isnan(grad_norm))
                progress.infinite_gradients += int(math.isinf(grad_norm))
                progress.skipped_updates += 1
                optimizer.zero_grad()
                group = Accumulation()
                log(progress.metrics())
                group_started = time.perf_counter()
                continue

            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.step += 1

            if progress.step % config.training.log_every_steps == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                loss_value = group.loss_sum / group.prediction_tokens
                metrics = progress.metrics() | {
                    "train/loss": loss_value,
                    "train/perplexity": perplexity(loss_value),
                    "optimizer/lr": lr,
                    "optimizer/grad_norm": grad_norm,
                    "train/tokens_per_second": group.input_tokens / (time.perf_counter() - group_started),
                }
                if device.type == "cuda":
                    metrics["gpu/peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                log(metrics)
            group = Accumulation()

            if progress.step % config.training.eval_every_steps == 0 and not should_stop():
                # Run val
                log(progress.metrics() | validate(model, val_loader, device))
                last_validation_step = progress.step
            group_started = time.perf_counter()
            if progress.step >= config.training.max_steps:
                break

        if epoch_batches == 0 and not should_stop():
            raise ValueError("Training range has no records matching seq_len")

    # A partial accumulation group is discarded, never applied on shutdown.
    optimizer.zero_grad()
    if config.training.eval_at_end and last_validation_step != progress.step:
        log(progress.metrics() | validate(model, val_loader, device))
    return progress
