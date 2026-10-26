"""Independent CUDA write/outer-gradient parity with D geometry and saved states."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import NeuralMemory
from evaluation.taal.longmemeval_numerics import write_report
from evaluation.taal.numerical_diagnostic import compare_tensors, tensor_summary


def reference_prediction(keys, weights, config, reading=False):
    predicted = keys
    for index in range(config.depth):
        predicted = predicted @ weights[f"layers.{index}.weight"].transpose(-1, -2)
        if index + 1 < config.depth:
            predicted = F.silu(predicted)
    if config.mlp_output_layernorm or (reading and config.read_layernorm):
        predicted = F.layer_norm(predicted, (config.dim,), eps=config.normalization_eps)
        if config.norm_learned_scale:
            predicted = predicted * (1 + weights["norm_gamma"].unsqueeze(1))
        if config.norm_residual:
            predicted = keys + predicted
    return predicted


def reference_gradient(weights, keys, values, strength, config):
    predicted = reference_prediction(keys, weights, config)
    loss = ((predicted - values).square().mean(-1) * strength).sum()
    gradients = torch.autograd.grad(loss, tuple(weights.values()), create_graph=True)
    return dict(zip(weights, gradients))


def precision_context(device, mode):
    if mode == "fp32":
        return nullcontext()
    return torch.autocast(device.type, dtype=torch.bfloat16)


def write_comparison(config, weights, keys, values, strength, device, mode):
    memory = NeuralMemory(config).to(device).train()
    weights = {
        name: value.detach().to(device).float().clone().requires_grad_()
        for name, value in weights.items()
    }
    keys = keys.detach().to(device).float().clone().requires_grad_()
    values = values.detach().to(device).float().clone().requires_grad_()
    strength = strength.detach().to(device).float().clone().requires_grad_()
    with precision_context(device, mode):
        manual = memory._chunk_gradient(weights, keys, values, strength).chunk_gradients
        automatic = reference_gradient(weights, keys, values, strength, config)
        probe = memory.memory_mlp(keys, weights=weights, return_intermediates=True)
        manual_scalar = sum(g.square().sum() for g in manual.values())
        auto_scalar = sum(g.square().sum() for g in automatic.values())
    inputs = (*weights.values(), keys, values, strength)
    outer_manual = torch.autograd.grad(manual_scalar, inputs, retain_graph=True)
    outer_auto = torch.autograd.grad(auto_scalar, inputs)
    return {
        "mode": mode,
        "gradient": {
            name: compare_tensors(manual[name].detach(), automatic[name].detach())
            for name in weights
        },
        "outer_gradient": [
            compare_tensors(a.detach(), b.detach())
            for a, b in zip(outer_manual, outer_auto)
        ],
        "dtypes": {
            "predicted": str(probe.predicted_values.dtype),
            "linear": [str(v.dtype) for v in probe.values_before_silu],
            "inverse_std": None
            if probe.inverse_std is None
            else str(probe.inverse_std.dtype),
        },
    }


def recurrent_probe(config, device, mode, method, count=206):
    torch.manual_seed(42)
    memory = NeuralMemory(config).to(device).train()
    with torch.no_grad():
        for layer in memory.memory_mlp.layers:
            layer.weight.normal_(std=0.02)
    state = memory.initial_state(1)
    initial = tuple(state.weights.values())
    weights = state.weights
    momentum = state.momentum
    keys = F.normalize(torch.randn(1, count, config.dim, device=device), dim=-1)
    values = torch.randn_like(keys) * 0.03
    records = []
    for index in range(count):
        with precision_context(device, mode):
            key = keys[:, index : index + 1]
            value = values[:, index : index + 1]
            strength = torch.full((1, 1), 0.1, device=device)
            if method == "manual":
                gradient = memory._chunk_gradient(
                    weights, key, value, strength
                ).chunk_gradients
            else:
                gradient = reference_gradient(weights, key, value, strength, config)
            momentum = {name: 0.9 * momentum[name] - gradient[name] for name in weights}
            weights = {name: 0.99 * weights[name] + momentum[name] for name in weights}
            read = reference_prediction(key, weights, config, reading=True)
        if index + 1 in (1, 16, 64, count) or not all(
            torch.isfinite(w).all() for w in weights.values()
        ):
            loss = read.float().square().sum()
            outer = torch.autograd.grad(loss, initial, retain_graph=True)
            records.append(
                {
                    "tokens": index + 1,
                    "read": tensor_summary(read),
                    "weights": {name: tensor_summary(w) for name, w in weights.items()},
                    "outer": [tensor_summary(g) for g in outer],
                }
            )
        if not all(torch.isfinite(w).all() for w in weights.values()):
            break
    return records


def run_precision(source_run, output, device):
    output.mkdir(parents=True, exist_ok=False)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    manifest = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "geometry": {"B": 1, "C": 1, "D": 128, "depth": 2},
        "save_traces": True,
        "trace_kind": "gradient/update tensors summarized per comparison; not a generated-text trajectory",
    }
    write_report(output / "manifest.json", manifest)
    configs = {
        "baseline": NeuralMemoryConfig(128),
        "output_ln": NeuralMemoryConfig(128, mlp_output_layernorm=True),
        "read_ln": NeuralMemoryConfig(128, read_layernorm=True),
        "scale_residual": NeuralMemoryConfig(
            128, mlp_output_layernorm=True, norm_learned_scale=True, norm_residual=True
        ),
    }
    results = {"initialization": {}, "saved_updates": {}, "recurrence": {}}
    for arm, config in configs.items():
        torch.manual_seed(42)
        memory = NeuralMemory(config)
        with torch.no_grad():
            for layer in memory.memory_mlp.layers:
                layer.weight.normal_(std=0.02)
        weights = memory.initial_state(1).weights
        keys = F.normalize(torch.randn(1, 1, 128), dim=-1)
        values = torch.randn_like(keys) * 0.03
        strength = torch.full((1, 1), 0.1)
        for mode in ("fp32", "bf16"):
            name = f"{arm}-{mode}"
            results["initialization"][name] = write_comparison(
                config, weights, keys, values, strength, device, mode
            )
            for method in ("manual", "autograd"):
                results["recurrence"][f"{name}-{method}"] = recurrent_probe(
                    config, device, mode, method
                )
                write_report(output / "results.json", results)
    directory = source_run / "reads-1/updates"
    slices_by_layer = {layer: [] for layer in (4, 8, 12, 24, 27)}
    for path in sorted(directory.glob("*.pt")):
        if path.name.startswith("adapter-"):
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        identity = payload.get("identity", {})
        if identity.get("layer") in slices_by_layer:
            slices_by_layer[identity["layer"]].append((path, payload))
    slices = []
    for index in range(5):
        for candidates in slices_by_layer.values():
            if index < len(candidates):
                slices.append(candidates[index])
    if not slices:
        raise ValueError("No real saved update slices found")
    for path, payload in slices[:24]:
        state, queries, keys, values, inputs, mask, dtype = payload["step"]
        layer = payload["identity"]["layer"]
        adapter = torch.load(
            directory / f"adapter-{layer}.pt", map_location="cpu", weights_only=True
        )
        strength = (
            F.linear(
                inputs.float(),
                adapter["neural_memory.write_strength_projection.weight"].float(),
                adapter["neural_memory.write_strength_projection.bias"].float(),
            )
            .sigmoid()
            .squeeze(-1)
        )
        strength = torch.where(mask, strength, 0)
        record = {
            "identity": payload["identity"],
            "strength_policy": "FP32 from saved control parameters; identical across methods",
        }
        for mode in ("fp32", "bf16"):
            record[mode] = write_comparison(
                configs["baseline"],
                state["weights"],
                keys,
                values,
                strength,
                device,
                mode,
            )
        results["saved_updates"][path.name] = record
        write_report(output / "results.json", results)
    write_report(
        output / "completion.json",
        {"status": "complete", "saved_slices": len(results["saved_updates"])},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_precision(args.source_run, args.output_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
