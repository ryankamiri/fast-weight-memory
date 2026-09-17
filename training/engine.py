"""Training and validation loops; fast-weight session state stays inside each call."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
import time
from typing import Callable

import torch

from .config import (
    BridgeMemoryLossConfig,
    CausalLMLossConfig,
    LossConfig,
    TrainingConfig,
)
from architectures.qwen.mlp import FWQwen3MLP


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
    input_tokens: int = 0
    all_token_loss_sum: float = 0.0
    all_token_count: int = 0
    delayed_answer_loss_sum: float = 0.0
    delayed_answer_count: int = 0

    def add(self, output, batch, input_tokens):
        all_count = prediction_count(batch["labels"]) if "labels" in batch else 0
        delayed_count = prediction_count(batch["delayed_labels"]) if "delayed_labels" in batch else 0
        all_loss = getattr(output, "all_token_loss", None)
        delayed_loss = getattr(output, "delayed_answer_loss", None)
        # Small trainer test doubles expose only the historical scalar loss.
        if all_count and all_loss is None and delayed_count == 0:
            all_loss = output.loss
        if all_count and all_loss is None:
            raise ValueError("Model did not return all_token_loss")
        if delayed_count and delayed_loss is None:
            raise ValueError("Model did not return delayed_answer_loss")
        self.batches += 1
        self.input_tokens += input_tokens
        if all_count:
            self.all_token_loss_sum += all_loss.detach().item() * all_count
            self.all_token_count += all_count
        if delayed_count:
            self.delayed_answer_loss_sum += delayed_loss.detach().item() * delayed_count
            self.delayed_answer_count += delayed_count

    def metrics(self, prefix: str, loss_config: LossConfig) -> dict[str, float]:
        metrics = {}
        objective = 0.0
        active_terms = 0
        if self.all_token_count:
            loss = self.all_token_loss_sum / self.all_token_count
            metrics[f"{prefix}/all_token_loss"] = loss
            metrics[f"{prefix}/all_token_perplexity"] = perplexity(loss)
            objective += loss_config.all_tokens_weight * loss
            active_terms += int(loss_config.all_tokens_weight > 0)
        if self.delayed_answer_count:
            if not isinstance(loss_config, BridgeMemoryLossConfig):
                raise TypeError("Delayed-answer metrics require BridgeMemoryLossConfig")
            loss = self.delayed_answer_loss_sum / self.delayed_answer_count
            metrics[f"{prefix}/delayed_answer_loss"] = loss
            metrics[f"{prefix}/delayed_answer_perplexity"] = perplexity(loss)
            objective += loss_config.delayed_answer_weight * loss
            active_terms += int(loss_config.delayed_answer_weight > 0)
        if active_terms == 0:
            raise ValueError("Batch has no active next-token targets")
        metrics[f"{prefix}/loss"] = objective
        # This is a standard perplexity only when the objective has one mean-CE term.
        if active_terms == 1:
            metrics[f"{prefix}/perplexity"] = perplexity(objective)
        return metrics


@dataclass
class CandidateScores:
    correct: int = 0
    vocabulary_correct: int = 0
    target_log_probability_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    count: int = 0

    def add(self, logits, candidates, targets):
        logits = logits.float()
        candidate_scores = logits.gather(1, candidates)
        choices = candidates.gather(1, candidate_scores.argmax(dim=1, keepdim=True)).squeeze(1)
        target_scores = logits.gather(1, targets[:, None]).squeeze(1)
        ranks = (logits > target_scores[:, None]).sum(dim=1) + 1
        self.correct += int((choices == targets).sum().item())
        self.vocabulary_correct += int((logits.argmax(dim=1) == targets).sum().item())
        self.target_log_probability_sum += float(
            (target_scores - torch.logsumexp(logits, dim=1)).sum().item()
        )
        self.reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
        self.count += targets.shape[0]

    def metrics(self, prefix):
        return {
            f"{prefix}_candidate_accuracy": self.correct / self.count,
            f"{prefix}_vocabulary_top_1_accuracy": self.vocabulary_correct / self.count,
            f"{prefix}_mean_target_log_probability": self.target_log_probability_sum / self.count,
            f"{prefix}_mean_target_reciprocal_rank": self.reciprocal_rank_sum / self.count,
        }


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


DIAGNOSTIC_FIELDS = {
    "candidate_token_ids", "target_token_ids", "conditions", "query_variants",
}


def move_batch(batch, device):
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def forward_batch(model, batch, loss_config: LossConfig, *, diagnostics=False):
    inputs = {name: value for name, value in batch.items() if name not in DIAGNOSTIC_FIELDS}
    if diagnostics and "candidate_token_ids" in batch:
        # The final input token is the answer; the preceding hidden state predicts it.
        inputs["logits_to_keep"] = 2
    if isinstance(loss_config, BridgeMemoryLossConfig):
        return model.forward_bridge_memory(
            **inputs,
            all_token_loss_weight=loss_config.all_tokens_weight,
            delayed_answer_loss_weight=loss_config.delayed_answer_weight,
            state=None,
            use_cache=False,
        )
    if not isinstance(loss_config, CausalLMLossConfig):
        raise TypeError(f"Unsupported loss config: {type(loss_config).__name__}")
    return model(**inputs, state=None, use_cache=False)


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
def _validate_pass(
    model, dataloader, device: torch.device, should_stop,
    loss_config: LossConfig | None = None,
) -> dict[str, float] | None:
    loss_config = CausalLMLossConfig() if loss_config is None else loss_config
    was_training = model.training
    model.eval()
    totals = Accumulation()
    candidate_groups = {"all": CandidateScores()}
    records = 0
    try:
        for batch in dataloader:
            if should_stop():
                return None
            batch = move_batch(batch, device)
            B, S = batch["input_ids"].shape
            with precision_context(device):
                output = forward_batch(model, batch, loss_config, diagnostics=True)
            totals.add(output, batch, B * S)
            if "candidate_token_ids" in batch:
                answer_logits = output.logits[:, -2, :]
                candidates = batch["candidate_token_ids"]
                targets = batch["target_token_ids"]
                candidate_groups["all"].add(answer_logits, candidates, targets)
                for index, (condition, variant) in enumerate(zip(
                    batch["conditions"], batch["query_variants"],
                )):
                    name = f"{condition}_{variant}"
                    candidate_groups.setdefault(name, CandidateScores()).add(
                        answer_logits[index:index + 1],
                        candidates[index:index + 1],
                        targets[index:index + 1],
                    )
            records += B
            del output, batch
        if should_stop():
            return None  # Never select a best checkpoint from partial validation.
    finally:
        model.train(was_training)
    if totals.batches == 0:
        raise ValueError("Validation range has no records matching seq_len")
    metrics = totals.metrics("val", loss_config) | {"val/records": records}
    for name, scores in candidate_groups.items():
        if scores.count:
            metrics.update(scores.metrics(f"val/{name}"))
    return metrics


def validate(model, dataloader, device: torch.device, should_stop=lambda: False,
             fast_weight_read_scales=(1.0, 0.5, 0.0),
             loss_config: LossConfig | None = None) -> dict[str, float] | None:
    layers = [module for module in model.modules()
              if isinstance(module, FWQwen3MLP) and module.is_fast_weight_layer]
    previous = [module.fast_weight_read_scale for module in layers]
    try:
        # Standard validation/checkpoint selection always uses full-strength reads.
        for module in layers:
            module.fast_weight_read_scale = 1.0
        metrics = _validate_pass(model, dataloader, device, should_stop, loss_config)
        if metrics is None or not layers:
            return metrics

        losses = {1.0: metrics["val/loss"]}
        for scale in fast_weight_read_scales:
            if scale == 1.0:
                result = metrics
            else:
                for module in layers:
                    module.fast_weight_read_scale = scale
                result = _validate_pass(model, dataloader, device, should_stop, loss_config)
                if result is None:
                    return None
            losses[scale] = result["val/loss"]
            # At scale 1, result and metrics are the same dictionary. Iterate
            # over a snapshot while adding the explicit scale-suffixed view.
            for name, value in list(result.items()):
                if name != "val/records":
                    metrics[f"{name}_fw_read_scale_{scale:g}"] = value
        if 0.0 in losses:
            for scale, loss in losses.items():
                if scale != 0:
                    metrics[f"val/fw_read_loss_improvement_scale_{scale:g}"] = losses[0.0] - loss
        return metrics
    finally:
        for module, scale in zip(layers, previous):
            module.fast_weight_read_scale = scale


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
    on_validation=None,
    progress: Progress | None = None,
) -> Progress:
    """Repeat epochs until max_steps successful updates or an external stop."""
    progress = Progress() if progress is None else progress
    group = Accumulation()
    model.train()
    optimizer.zero_grad()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("Training requires at least one trainable parameter")
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
            batch = move_batch(batch, device)
            B, S = batch["input_ids"].shape
            input_tokens = B * S
            progress.tokens_seen += input_tokens

            with precision_context(device):
                output = forward_batch(model, batch, config.loss)
                loss = output.loss
                if isinstance(config.loss, CausalLMLossConfig):
                    loss = config.loss.all_tokens_weight * loss
            group.add(output, batch, input_tokens)
            # Do not retain the logits or returned fast-weight state across calls.
            del output, batch
            loss_value = loss.detach().item()
            if should_stop():
                del loss
                break
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
            if should_stop():
                break
            if group.batches < config.training.gradient_accumulation_steps:
                # don't backprob yet
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, config.training.max_grad_norm,
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
                metrics = progress.metrics() | group.metrics("train", config.loss) | {
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
                metrics = validate(model, val_loader, device, should_stop,
                                   config.validation.fast_weight_read_scales, config.loss)
                if metrics is not None:
                    log(progress.metrics() | metrics)
                    if on_validation is not None:
                        on_validation(model, progress, metrics)
                    last_validation_step = progress.step
            group_started = time.perf_counter()
            if progress.step >= config.training.max_steps:
                break

        if epoch_batches == 0 and not should_stop():
            raise ValueError("Training range has no records matching seq_len")

    # A partial accumulation group is discarded, never applied on shutdown.
    optimizer.zero_grad()
    if config.training.eval_at_end and last_validation_step != progress.step and not should_stop():
        metrics = validate(model, val_loader, device, should_stop,
                           config.validation.fast_weight_read_scales, config.loss)
        if metrics is not None:
            log(progress.metrics() | metrics)
            if on_validation is not None:
                on_validation(model, progress, metrics)
    return progress
