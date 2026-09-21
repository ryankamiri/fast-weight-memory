import math

from jaxtyping import Float, Int
import torch


def score_logits(logits, example, tokenizer):
    logits: Float[torch.Tensor, "vocab_size"] = logits.float()
    target = int(example["target_token_id"])
    candidates: Int[torch.Tensor, "num_candidates"] = torch.tensor(
        example["candidate_token_ids"],
        dtype=torch.long,
        device=logits.device,
    )
    target_score = logits[target]
    target_rank = int((logits > target_score).sum().item()) + 1
    candidate_scores = logits[candidates]
    candidate_choice = int(candidates[candidate_scores.argmax()].item())
    decoy_scores = candidate_scores[candidates != target]
    top_token = int(logits.argmax().item())
    return {
        "target_rank": target_rank,
        "target_reciprocal_rank": 1.0 / target_rank,
        "target_log_probability": float(
            (target_score - torch.logsumexp(logits, dim=0)).item()
        ),
        "target_vs_best_decoy_margin": float(
            (target_score - decoy_scores.max()).item()
        ),
        "candidate_choice_token_id": candidate_choice,
        "candidate_correct": candidate_choice == target,
        "vocabulary_top_token_id": top_token,
        "vocabulary_top_token": tokenizer.decode([top_token]),
        "vocabulary_top_1_correct": top_token == target,
    }


def aggregate_scores(rows):
    rows = list(rows)
    count = len(rows)
    if count == 0:
        raise ValueError("Cannot summarize an empty score group")
    return {
        "examples": count,
        "candidate_accuracy": sum(row["candidate_correct"] for row in rows) / count,
        "vocabulary_top_1_accuracy": sum(
            row["vocabulary_top_1_correct"] for row in rows
        )
        / count,
        "mean_target_log_probability": math.fsum(
            row["target_log_probability"] for row in rows
        )
        / count,
        "mean_target_reciprocal_rank": math.fsum(
            row["target_reciprocal_rank"] for row in rows
        )
        / count,
        "mean_target_vs_best_decoy_margin": math.fsum(
            row["target_vs_best_decoy_margin"] for row in rows
        )
        / count,
    }


def grouped_score_summary(rows, group_fields):
    rows = list(rows)
    groups = {"all": rows}
    for row in rows:
        key = "/".join(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    return {
        name: aggregate_scores(group)
        for name, group in groups.items()
        if group
    }
