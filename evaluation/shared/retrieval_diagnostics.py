"""Tokenwise answer loss and real training-forward versus decode-path comparisons."""

from dataclasses import asdict, dataclass
from collections import defaultdict

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from training.engine import precision_context
from evaluation.shared.benchmarks import exact_match


def selection_counts(rows, predictions):
    if set(predictions) != {row["example_id"] for row in rows}:
        raise ValueError("Selection metrics require the entire requested cohort")
    prefix_groups, remap_groups = defaultdict(list), defaultdict(list)
    correct = 0
    positions = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        hit = int(exact_match(predictions[row["example_id"]]["hypothesis"], row["answer"]))
        correct += hit
        metadata = row["metadata"]
        prefix_groups[row["group_id"]].append(hit)
        key = (metadata["family_id"], tuple(metadata["fact_order"]), metadata["query_key"])
        remap_groups[key].append(hit)
        position = str(metadata["queried_fact_order"])
        positions[position]["correct"] += hit
        positions[position]["total"] += 1
    summary = {"questions": {"correct": correct, "total": len(rows)}, "queried_position": dict(positions)}
    for name, groups in (("both_questions", prefix_groups), ("both_remapped_answers", remap_groups)):
        # Consumed cohorts are incomplete: singleton groups cannot count as pairs.
        pairs = [hits for hits in groups.values() if len(hits) == 2]
        summary[name] = {"correct": sum(all(hits) for hits in pairs), "total": len(pairs),
                         "incomplete_groups": sum(len(hits) != 2 for hits in groups.values())}
    return summary


@dataclass
class AnswerLoss:
    label_nll: float
    eos_nll: float
    combined_nll: float
    label_tokens: int
    token_nll: list[float]


def answer_loss(logits: Float[torch.Tensor, "T V"], labels: Int[torch.Tensor, "T"]) -> AnswerLoss:
    if len(labels) < 2 or logits.shape[0] != len(labels):
        raise ValueError("Expected label tokens followed by one EOS target")
    losses = F.cross_entropy(logits.float(), labels, reduction="none")
    if not torch.isfinite(losses).all():
        raise ValueError("Non-finite answer loss")
    return AnswerLoss(float(losses[:-1].mean()), float(losses[-1]),
                      float(losses.mean()), len(labels) - 1, losses.tolist())


def score_answer(model, row):
    device = next(model.parameters()).device
    ids = torch.tensor([row["input_ids"]], device=device)
    prompt_length = len(row["prompt_input_ids"])
    labels = ids[0, prompt_length:]
    model.eval()
    with torch.inference_mode(), precision_context(device):
        output = model(ids, use_cache=False, logits_to_keep=len(labels) + 1)
        # Position P-1 predicts the first label; the final EOS input predicts nothing scored.
        logits = output.logits[0, :-1]
        loss = answer_loss(logits, labels)
        predicted_ids = logits.argmax(dim=-1)
        correct = predicted_ids == labels
        # Later label positions receive the correct preceding tokens. This is
        # teacher-forced answer scoring, distinct from cached greedy generation.
        return {
            **asdict(loss),
            "expected_token_ids": labels.tolist(),
            "predicted_token_ids": predicted_ids.tolist(),
            "first_label_token_correct": bool(correct[0]),
            "all_label_tokens_correct": bool(correct[:-1].all()),
            "label_and_eos_correct": bool(correct.all()),
        }


def compare_logits(training_logits, decoding_logits, labels, training_loss):
    training_logits = training_logits.float()
    decoding_logits = decoding_logits.float()
    if not torch.isfinite(training_logits).all() or not torch.isfinite(decoding_logits).all():
        raise ValueError("Non-finite parity logits")
    difference = (training_logits - decoding_logits).abs()
    train = answer_loss(training_logits, labels)
    decode = answer_loss(decoding_logits, labels)
    tv = (training_logits.softmax(-1) - decoding_logits.softmax(-1)).abs().sum(-1) / 2
    same_argmax = bool(torch.equal(training_logits.argmax(-1), decoding_logits.argmax(-1)))
    metrics = {
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "max_probability_total_variation": float(tv.max()),
        "training_vs_decode_nll_difference": abs(train.combined_nll - decode.combined_nll),
        "fused_vs_explicit_nll_difference": abs(training_loss - train.combined_nll),
        "same_argmax": same_argmax,
        "training_loss": training_loss,
        "training": asdict(train), "decode": asdict(decode),
        "training_argmax": training_logits.argmax(-1).tolist(),
        "decode_argmax": decoding_logits.argmax(-1).tolist(),
    }
    # Conservative BF16 screening, not a proof of gradient or kernel equivalence.
    # Preserve failed screens for investigation, including an argmax near-tie.
    metrics["passed"] = (
        same_argmax and metrics["max_abs_logit_difference"] <= 0.5
        and metrics["mean_abs_logit_difference"] <= 0.03
        and metrics["max_probability_total_variation"] <= 0.02
        and metrics["training_vs_decode_nll_difference"] <= 0.05
        and metrics["fused_vs_explicit_nll_difference"] <= 0.05
    )
    return metrics


def path_parity(model, row):
    device = next(model.parameters()).device
    ids = torch.tensor([row["input_ids"]], device=device)
    labels = torch.tensor([row["labels"]], device=device)
    prompt_length = len(row["prompt_input_ids"])
    answer = ids[0, prompt_length:]
    was_training = model.training
    try:
        model.train()
        # Execute the actual grad-enabled training forward and CUDA fused loss.
        # No optimizer step or backward is performed in this read-only diagnostic.
        with torch.enable_grad(), precision_context(device):
            output = model(ids, labels=labels, use_cache=False, logits_to_keep=len(answer) + 1)
            training_logits = output.logits[0, :-1].detach().float().clone()
            training_loss = float(output.loss.detach())
        del output
        model.eval()
        with torch.inference_mode(), precision_context(device):
            output = model.prefill(ids[:, :prompt_length])
            decoding_logits = [output.logits[0, -1].float()]
            for token in answer[:-1]:
                output = model(token.reshape(1, 1), state=output.state, use_cache=True,
                               logits_to_keep=1, prepend_memory_tokens=False)
                decoding_logits.append(output.logits[0, -1].float())
            return compare_logits(training_logits, torch.stack(decoding_logits), answer, training_loss)
    finally:
        model.train(was_training)
