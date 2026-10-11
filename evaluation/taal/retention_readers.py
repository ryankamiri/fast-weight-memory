"""Held-out readers and cross-stage transfer, never a causal retrieval claim."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from evaluation.storage import ensure_manifest
from evaluation.taal.state_readout_probe import (
    StateFeatures, StateProjector, save_tensor_payload, validate_splits,
)


def fit_reader(train, train_y, val, val_y, *, classes, kind, strengths, seed,
               epochs=200):
    if kind == "ridge":
        mean = train.mean(0)
        std = train.std(0, unbiased=False).clamp_min(1e-6)
        train, val = (train - mean) / std, (val - mean) / std
        values = torch.nn.functional.one_hot(train_y, classes).double()
        prior = values.mean(0)
        # Dual ridge solves N x N, not P x P when layers are concatenated.
        gram = train.double() @ train.double().T
        best = None
        for strength in strengths:
            dual = torch.linalg.solve(gram + strength * torch.eye(len(train), dtype=torch.float64), values - prior)
            weights = train.double().T @ dual
            accuracy = float(((val.double() @ weights + prior).argmax(-1) == val_y).float().mean())
            if best is None or accuracy > best[0]:
                best = accuracy, strength, weights
        accuracy, strength, weights = best
        return {"val_accuracy": accuracy, "ridge_strength": strength}, {
            "kind": kind, "mean": mean, "std": std, "weights": weights, "prior": prior,
        }
    if kind != "mlp":
        raise ValueError(f"Unknown reader {kind}")
    torch.manual_seed(seed)
    mean = train.mean(0)
    std = train.std(0, unbiased=False).clamp_min(1e-6)
    train, val = (train - mean) / std, (val - mean) / std
    best = None
    for decay in strengths:
        torch.manual_seed(seed)
        model = torch.nn.Sequential(torch.nn.Linear(train.shape[-1], 32),
                                    torch.nn.SiLU(), torch.nn.Linear(32, classes))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(train), train_y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                accuracy = float((model(val).argmax(-1) == val_y).float().mean())
            if best is None or accuracy > best[0]:
                best = (accuracy, decay, epoch + 1,
                        {name: value.detach().clone() for name, value in model.state_dict().items()})
    accuracy, decay, epoch, parameters = best
    return {"val_accuracy": accuracy, "weight_decay": decay, "epoch": epoch}, {
        "kind": kind, "mean": mean, "std": std, "parameters": parameters,
        "classes": classes, "width": train.shape[-1],
    }


@torch.no_grad()
def predict(reader, x):
    x = (x - reader["mean"]) / reader["std"]
    if reader["kind"] == "ridge":
        return (x.double() @ reader["weights"] + reader["prior"]).argmax(-1)
    model = torch.nn.Sequential(torch.nn.Linear(reader["width"], 32),
                                torch.nn.SiLU(), torch.nn.Linear(32, reader["classes"]))
    model.load_state_dict(reader["parameters"])
    return model(x).argmax(-1)


def reader_sweep(examples, features, settings):
    labels = validate_splits(examples)
    indices = {split: torch.tensor([i for i, row in enumerate(examples)
                                   if row["probe_split"] == split])
               for split in ("train", "val", "test")}
    y = torch.tensor([labels.index(row["target_token_id"]) for row in examples])
    summary, readers = {}, {}
    N, L, P = next(iter(features.values())).weights.shape
    # Pooled versus concatenated layers tests whether pooling destroyed a code.
    for component in ("weights", "momentum", "both"):
        for layout in ("pooled", "layerwise"):
            xs = {}
            for stage, state in features.items():
                parts = []
                for name in ("weights", "momentum"):
                    if component in (name, "both"):
                        value = getattr(state, name)
                        parts.append(value.mean(1) if layout == "pooled" else value.flatten(1))
                xs[stage] = torch.cat(parts, -1)
            for source, x in xs.items():
                for kind in ("ridge", "mlp"):
                    key = f"{source}/{layout}/{component}/{kind}"
                    train, val = x[indices["train"]], x[indices["val"]]
                    strengths = settings["ridge_strengths"] if kind == "ridge" else settings["weight_decays"]
                    fit, reader = fit_reader(train, y[indices["train"]], val, y[indices["val"]],
                        classes=len(labels), kind=kind, strengths=strengths,
                        seed=settings["seed"], epochs=settings["reader_epochs"])
                    transfer = {destination: {
                        "test_accuracy": float((predict(reader, value[indices["test"]]) == y[indices["test"]]).float().mean()),
                        "test_predictions": predict(reader, value[indices["test"]]).tolist(),
                    } for destination, value in xs.items()}
                    controls = []
                    generator = torch.Generator().manual_seed(settings["seed"])
                    for control in range(settings["shuffled_controls"]):
                        _, shuffled = fit_reader(train, y[indices["train"]][torch.randperm(len(train), generator=generator)],
                            val, y[indices["val"]][torch.randperm(len(val), generator=generator)],
                            classes=len(labels), kind=kind, strengths=strengths,
                            seed=settings["seed"] + control, epochs=settings["reader_epochs"])
                        controls.append(float((predict(shuffled, x[indices["test"]]) == y[indices["test"]]).float().mean()))
                    summary[key] = {**fit, "features": x.shape[-1], "transfer": transfer,
                                    "shuffled_test_accuracies": controls}
                    readers[key] = reader
                    print(f"Reader {key}: {transfer[source]['test_accuracy']:.3f}", flush=True)
    return {"classes": labels, "chance": 1 / len(labels), "layers": L,
            "split_counts": {name: len(index) for name, index in indices.items()},
            "readers": summary}, readers


def load_saved_features(source, examples, *, dimension, seed, device):
    initial = torch.load(source / "memory_states/initial.pt", weights_only=True)
    states = {layer: SimpleNamespace(**{name: {key: tensor.to(device) for key, tensor in values.items()}
                                        for name, values in state.items()}, pending_count=0)
              for layer, state in initial.items()}
    projector = StateProjector(states, dimension=dimension, seed=seed)
    stages = {"fact": "fact", "middle": "normal-middle", "final": "normal-final"}
    collected = {stage: [] for stage in stages}
    for i, example in enumerate(examples):
        for stage, filename in stages.items():
            payload = torch.load(source / "memory_states" / example["example_id"] / f"{filename}.pt", weights_only=True)
            states = {layer: SimpleNamespace(**{name: {key: tensor.to(device) for key, tensor in values.items()}
                                               for name, values in state.items()}, pending_count=0)
                      for layer, state in payload.items()}
            collected[stage].append(projector(states))
        print(f"Projected saved state {i + 1}/{len(examples)}", flush=True)
    return {stage: StateFeatures(torch.stack([row.weights for row in rows]),
                                torch.stack([row.momentum for row in rows]))
            for stage, rows in collected.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(4)
    dataset = Path(settings["dataset"]["path"])
    examples = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    source_manifest = json.loads((args.source / "manifest.json").read_text())
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if source_manifest["dataset_sha256"] != digest:
        raise ValueError("Saved-state dataset identity mismatch")
    trace_manifest = args.source / "memory_traces/manifest.json"
    if settings.get("save_traces", True) and not trace_manifest.is_file():
        raise ValueError("Saved-state source traces are missing")
    ensure_manifest(args.output_dir / "manifest.json", {
        "config": settings, "source": str(args.source.resolve()), "source_manifest": source_manifest,
        "trace_source": str(trace_manifest.resolve()), "capture": "saved-state reader, no new model forward",
    })
    features = load_saved_features(args.source, examples, dimension=settings["projection_dim"],
                                   seed=settings["seed"], device="cuda" if torch.cuda.is_available() else "cpu")
    summary, readers = reader_sweep(examples, features, settings)
    save_tensor_payload(args.output_dir / "readers.pt", readers)
    save_tensor_payload(args.output_dir / "features.pt", {
        stage: {"weights": values.weights, "momentum": values.momentum} for stage, values in features.items()})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
