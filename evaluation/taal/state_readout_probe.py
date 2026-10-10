"""Frozen-checkpoint state collection and capacity-controlled held-out readouts.

The classifier sees fixed projections of W and S only, never tokens, KV,
record identity or model answer scores. This is an information probe, not a
replacement for matched causal memory interventions.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import yaml
from transformers import AutoTokenizer

from architectures.taal.qwen.state import NeuralMemoryStates
from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.lifecycle import lifecycle_boundaries
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.state_audit import clone_memory_states, fork_session_state, load_model, score_logits
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import (
    TaalTraceExporter, TaalTraceRecorder, make_trace_tokens, trace_greedy_prediction,
)
from utils.seed import seed_everything


@dataclass(frozen=True)
class StateFeatures:
    weights: torch.Tensor  # [L, P] on CPU
    momentum: torch.Tensor  # [L, P] on CPU


class StateProjector:
    def __init__(self, initial: NeuralMemoryStates, *, dimension: int, seed: int):
        if type(dimension) is not int or dimension < 1:
            raise ValueError("projection_dim must be a positive integer")
        self.layers = sorted(initial)
        first = initial[self.layers[0]]
        self.names = sorted(first.weights)
        self.width = sum(first.weights[name].numel() for name in self.names)
        generator = torch.Generator().manual_seed(seed)
        # The same fixed Gaussian map at every layer, independent of labels.
        self.projection = torch.randn(self.width, dimension, generator=generator) / dimension ** 0.5
        self.projection = self.projection.to(first.weights[self.names[0]].device)

    @torch.inference_mode()
    def __call__(self, states: NeuralMemoryStates) -> StateFeatures:
        projected = {}
        for component in ("weights", "momentum"):
            vectors = []
            for layer in self.layers:
                state = states[layer]
                if state.pending_count:
                    raise ValueError("Probe requires committed chunk-1 states")
                parameters = getattr(state, component)
                vector = torch.cat([parameters[name].float().reshape(-1) for name in self.names])
                if vector.numel() != self.width:
                    raise ValueError("Probe layers must have the same memory parameter shapes")
                vectors.append(vector)
            features = torch.stack(vectors) @ self.projection
            if not bool(torch.isfinite(features).all()):
                raise ValueError("Non-finite memory features")
            projected[component] = features.cpu()
        return StateFeatures(**projected)


def validate_splits(examples: list[dict]) -> list[int]:
    labels = sorted({int(row["target_token_id"]) for row in examples})
    families = {}
    ids = set()
    for row in examples:
        if row["example_id"] in ids:
            raise ValueError("Duplicate probe example")
        ids.add(row["example_id"])
        family, split = row["context_family"], row["probe_split"]
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown probe split")
        if family in families and families[family] != split:
            raise ValueError("Context-family leakage across probe splits")
        families[family] = split
    for split in ("train", "val", "test"):
        selected = [row for row in examples if row["probe_split"] == split]
        counts = [sum(int(row["target_token_id"]) == label for row in selected) for label in labels]
        if not counts or min(counts) == 0 or len(set(counts)) != 1:
            raise ValueError("Every probe split must contain balanced labels")
    return labels


@torch.inference_mode()
def filler_forward(model, ids: list[int], state, *, start: int, end: int,
                   execution_block_size: int, hidden_span: tuple[int, int] | None):
    """Apply a key mask to retained absolute positions, never an old KV length."""
    output = None
    for block_start in range(start, end, execution_block_size):
        block_end = min(block_start + execution_block_size, end)
        tokens = torch.tensor([ids[block_start:block_end]], device=model.device)
        attention_mask = None
        if hidden_span is not None:
            retained = state.past_key_values.layers[0].positions
            incoming = torch.arange(block_start, block_end, device=model.device)
            positions = torch.cat((retained, incoming))
            visible = (positions < hidden_span[0]) | (positions >= hidden_span[1])
            attention_mask = visible.unsqueeze(0)
        output = model(tokens, state=state, use_cache=True, logits_to_keep=1,
                       attention_mask=attention_mask, prepend_memory_tokens=False)
        state = output.state
    if output is None:
        raise ValueError("Filler interval must be nonempty")
    return output


def save_state(path: Path, states: NeuralMemoryStates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Only W/S snapshots, not whole per-token trajectories or copies of KV.
    save_tensor_payload(path, {layer: {
        component: {name: value.detach().cpu().clone() for name, value in getattr(state, component).items()}
        for component in ("weights", "momentum")
    } for layer, state in states.items()})


def save_tensor_payload(path: Path, payload: dict | torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def collect_example(model, tokenizer, example: dict, *, projector: StateProjector,
                    root: Path, exporter: TaalTraceExporter | None,
                    execution_block_size: int, save_memory_states: bool) -> dict[str, StateFeatures]:
    boundaries = lifecycle_boundaries(example, tokenizer)
    ids = example["input_ids"]
    features = {}
    common = TaalTraceRecorder(model, projector.layers, boundaries["fact"])
    state = None
    start = 0
    with common:
        for stage in ("label", "fact"):
            end = boundaries[stage]
            output = model.prefill(torch.tensor([ids[start:end]], device=model.device),
                state=state, execution_block_size=execution_block_size,
                prepend_memory_tokens=start == 0)
            state = output.state
            features[stage] = projector(state.memory_states)
            if save_memory_states:
                save_state(root / "memory_states" / example["example_id"] / f"{stage}.pt", state.memory_states)
            start = end
    acquired = state
    fact_end = boundaries["fact"]
    branches = {
        "normal": None,
        "fact_hidden": (0, fact_end),
        "unrelated_hidden": (fact_end, 2 * fact_end),
    }
    if 2 * fact_end >= boundaries["middle"]:
        raise ValueError("Probe needs an unrelated filler span equal to the fact length")
    for branch, hidden_span in branches.items():
        state = fork_session_state(acquired, clone_memory_states(acquired.memory_states))
        recorder = TaalTraceRecorder(model, projector.layers, len(ids) - fact_end, position_offset=fact_end)
        start = fact_end
        with recorder:
            for stage in ("middle", "final"):
                end = boundaries[stage]
                output = filler_forward(model, ids, state, start=start, end=end,
                    execution_block_size=execution_block_size, hidden_span=hidden_span)
                state = output.state
                features[f"{branch}/{stage}"] = projector(state.memory_states)
                if save_memory_states:
                    save_state(root / "memory_states" / example["example_id"] / f"{branch}-{stage}.pt", state.memory_states)
                start = end
            retained = state.past_key_values.layers[0].positions
            if bool((retained < fact_end).any()):
                raise ValueError("Original fact unexpectedly remains in final query KV")
            query = torch.tensor([ids[start:]], device=model.device)
            output = model.prefill(query, state=state, execution_block_size=execution_block_size,
                                   prepend_memory_tokens=False)
        scores = score_logits(output.logits[0, -1].float(), example, tokenizer)
        outcome = {"evaluation_id": f"{example['example_id']}/{branch}",
                   "example_id": example["example_id"], "audit_condition": branch,
                   "context_family": example["context_family"], "probe_split": example["probe_split"],
                   **scores}
        existing = read_results(root / "results.jsonl", id_field="evaluation_id")
        if outcome["evaluation_id"] not in existing:
            append_result(root / "results.jsonl", outcome)
        if exporter is not None:
            prediction = trace_greedy_prediction(model, tokenizer,
                token_id=int(scores["vocabulary_top_token_id"]), position=len(ids),
                state=output.state, layers=projector.layers, memory_read_scale=1.0)
            exporter.export(TraceEpisode(
                run_id=root.name, example_id=example["example_id"], condition_id=branch,
                checkpoint=exporter.manifest["run_metadata"]["checkpoint"],
                tokenizer_id=exporter.manifest["run_metadata"]["tokenizer"],
                tokens=make_trace_tokens(ids, tokenizer) + [prediction.token],
                writes=common.writes + recorder.writes + prediction.writes,
                reads=common.reads + recorder.reads + prediction.reads,
                internal_prefixes=common.internal_prefixes + recorder.internal_prefixes + prediction.internal_prefixes,
                outcome=outcome, metadata={
                    "expected_answer": example["answer"], "expected_token_id": example["target_token_id"],
                    "query_start_position": boundaries["final"], "filler_key_mask_span": hidden_span,
                    "mask_applies_only_during_filler": True,
                    "query_kv_role": "branch-specific KV, unlike the recipient-KV layer-access audit",
                    "snapshot_boundaries": boundaries,
                },
            ))
    return features


def ridge_readout(train_x, train_y, val_x, val_y, test_x, test_y, *, classes: int, strengths: list[float]):
    """Standardize on train only; choose regularization on val, never test."""
    mean = train_x.mean(0)
    std = train_x.std(0, unbiased=False).clamp_min(1e-6)
    def design(features):
        normalized = (features - mean) / std
        return torch.cat((normalized, torch.ones(len(features), 1)), dim=1).double()
    train, val, test = map(design, (train_x, val_x, test_x))
    labels = torch.nn.functional.one_hot(train_y, classes).double()
    gram, right = train.T @ train, train.T @ labels
    best = None
    for strength in strengths:
        penalty = torch.eye(train.shape[1], dtype=torch.float64) * strength
        penalty[-1, -1] = 0
        weights = torch.linalg.solve(gram + penalty, right)
        accuracy = float(((val @ weights).argmax(1) == val_y).float().mean())
        if best is None or accuracy > best[0]:
            best = (accuracy, strength, weights)
    val_accuracy, strength, weights = best
    prediction = (test @ weights).argmax(1)
    return {
        "val_accuracy": val_accuracy, "test_accuracy": float((prediction == test_y).float().mean()),
        "ridge_strength": strength, "test_predictions": prediction.tolist(),
    }, {"mean": mean, "std": std, "weights": weights}


def fit_probes(examples, features: dict[str, StateFeatures], *, groups: dict[str, list[int]],
               strengths: list[float], shuffled_controls: int, seed: int):
    labels = validate_splits(examples)
    indices = {split: torch.tensor([i for i, row in enumerate(examples) if row["probe_split"] == split])
               for split in ("train", "val", "test")}
    targets = torch.tensor([labels.index(row["target_token_id"]) for row in examples])
    scores, readouts = {}, {}
    for stage, state in features.items():
        for group, layers in groups.items():
            for component in ("weights", "momentum", "both"):
                parts = []
                for name in ("weights", "momentum"):
                    if component in (name, "both"):
                        # Equal capacity for every layer group: mean P features,
                        # not a larger classifier for groups with more layers.
                        parts.append(getattr(state, name)[:, layers].mean(1))
                x = torch.cat(parts, dim=1)
                train_x, val_x, test_x = (x[indices[split]] for split in ("train", "val", "test"))
                train_y, val_y, test_y = (targets[indices[split]] for split in ("train", "val", "test"))
                result, readout = ridge_readout(train_x, train_y, val_x, val_y, test_x, test_y,
                                              classes=len(labels), strengths=strengths)
                generator = torch.Generator().manual_seed(seed)
                shuffled = []
                for _ in range(shuffled_controls):
                    control, _ = ridge_readout(
                        train_x, train_y[torch.randperm(len(train_y), generator=generator)],
                        val_x, val_y[torch.randperm(len(val_y), generator=generator)],
                        test_x, test_y, classes=len(labels), strengths=strengths)
                    shuffled.append(control["test_accuracy"])
                key = f"{stage}/{group}/{component}"
                scores[key] = {**result, "features": x.shape[1], "shuffled_test_accuracies": shuffled,
                              "shuffled_mean": sum(shuffled) / len(shuffled)}
                readouts[key] = readout
    return {"classes": labels, "chance_accuracy": 1 / len(labels),
            "split_counts": {split: len(index) for split, index in indices.items()}, "probes": scores}, readouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    for name in ("save_traces", "save_memory_states"):
        if type(settings.get(name, True)) is not bool:
            raise ValueError(f"{name} must be boolean")
    if type(settings["shuffled_controls"]) is not int or settings["shuffled_controls"] < 1:
        raise ValueError("shuffled_controls must be positive")
    if not settings["ridge_strengths"] or any(value <= 0 for value in settings["ridge_strengths"]):
        raise ValueError("Ridge strengths must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("State collection requires CUDA")
    seed_everything(settings["seed"])
    path = Path(settings["dataset"]["path"])
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    examples = [json.loads(line) for line in path.read_text().splitlines() if line]
    validate_splits(examples)
    manifest = {"config": settings, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint),
                "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dataset_metadata": metadata,
                "torch_version": str(torch.__version__)}
    ensure_manifest(args.output_dir / "manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
    model = load_model(args.checkpoint).to("cuda").eval()
    if model.config.memory_chunk_size != 1:
        raise ValueError("State readout probe requires memory_chunk_size=1")
    torch.set_num_threads(4)
    initial = model.model.initial_memory_states(batch_size=1)
    projector = StateProjector(initial, dimension=settings["projection_dim"], seed=settings["seed"])
    save_tensor_payload(args.output_dir / "projection.pt", projector.projection.cpu())
    if sorted(index for group in settings["layer_groups"].values() for index in group) != projector.layers:
        raise ValueError("Layer groups must partition the checkpoint memory layers")
    exporter = TaalTraceExporter(args.output_dir / "memory_traces", run_metadata={
        **manifest, "tokenizer": metadata["tokenizer"], "tokenizer_revision": metadata["tokenizer_revision"],
        "layers": projector.layers, "capture": "all_examples_all_layers_compact_scalars",
    }) if settings.get("save_traces", True) else None
    if settings.get("save_memory_states", True):
        save_state(args.output_dir / "memory_states/initial.pt", initial)
    feature_rows = []
    for index, example in enumerate(examples):
        destination = args.output_dir / "features" / f"{example['example_id']}.pt"
        if destination.exists():
            saved = torch.load(destination, weights_only=True)
            row = {stage: StateFeatures(**values) for stage, values in saved.items()}
        else:
            row = collect_example(model, tokenizer, example, projector=projector, root=args.output_dir,
                exporter=exporter, execution_block_size=settings["execution_block_size"],
                save_memory_states=settings.get("save_memory_states", True))
            row["initial"] = projector(initial)
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_tensor_payload(destination, {stage: {"weights": values.weights, "momentum": values.momentum}
                                             for stage, values in row.items()})
        feature_rows.append(row)
        print(f"Collected {index + 1}/{len(examples)}: {example['example_id']}", flush=True)
    features = {stage: StateFeatures(
        weights=torch.stack([row[stage].weights for row in feature_rows]),
        momentum=torch.stack([row[stage].momentum for row in feature_rows]),
    ) for stage in feature_rows[0]}
    groups = {**settings["layer_groups"], "all": projector.layers,
              **{f"layer-{layer}": [layer] for layer in projector.layers}}
    summary, readouts = fit_probes(examples, features, groups=groups,
        strengths=settings["ridge_strengths"], shuffled_controls=settings["shuffled_controls"], seed=settings["seed"])
    if exporter is not None and len(exporter.manifest["episodes"]) != len(examples) * 3:
        raise ValueError("Incomplete probe trace export")
    results = read_results(args.output_dir / "results.jsonl", id_field="evaluation_id")
    if len(results) != len(examples) * 3:
        raise ValueError("Incomplete probe outcome scores")
    summary["model_query_scores"] = {}
    for branch in ("normal", "fact_hidden", "unrelated_hidden"):
        for split in ("train", "val", "test"):
            rows = [row for row in results.values() if row["audit_condition"] == branch and row["probe_split"] == split]
            mean_loss = -sum(row["target_log_probability"] for row in rows) / len(rows)
            summary["model_query_scores"][f"{branch}/{split}"] = {
                "examples": len(rows), "mean_answer_loss": mean_loss,
                "answer_perplexity": float(torch.tensor(mean_loss).exp()),
                "vocabulary_accuracy": sum(row["vocabulary_top_1_correct"] for row in rows) / len(rows),
                "candidate_accuracy": sum(row["candidate_correct"] for row in rows) / len(rows),
            }
    save_tensor_payload(args.output_dir / "readouts.pt", readouts)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Complete: {len(summary['probes'])} readouts, {summary['split_counts']}", flush=True)


if __name__ == "__main__":
    main()
