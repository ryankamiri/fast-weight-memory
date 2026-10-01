"""Inspect the write, forget, read, and BF16 residual paths on one TaaL episode."""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from evaluation.scoring import score_logits
from evaluation.taal.state_audit import load_model, split_episode
from evaluation.taal.trace_export import TaalTraceRecorder


def stable_weight_norm(weights: dict[str, torch.Tensor]) -> float:
    """Use float64 before squaring to avoid false zeros in collapsed states."""
    squared = sum(value.detach().double().square().sum() for value in weights.values())
    return math.sqrt(squared.item())


class LayerProbe:
    def __init__(self, layer_index: int, taal, positions: set[int], read_scale: float):
        self.layer_index = layer_index
        self.taal = taal
        self.positions = positions
        self.read_scale = read_scale
        self.write_position = -taal.config.num_persistent_tokens
        self.read_position = -taal.config.num_persistent_tokens
        self.text_position = 0
        self.value_offset = 0
        self.values = None
        self.events: dict[int, dict] = {}
        memory = taal.neural_memory
        self.handles = [
            memory.value_conv.register_forward_hook(self.record_values),
            memory.memory_mlp.register_forward_hook(
                self.record_memory_mlp, with_kwargs=True
            ),
            memory.write_strength_projection.register_forward_hook(
                self.record_write_strength
            ),
            memory.forget_projection.register_forward_hook(self.record_forget),
            memory.momentum_projection.register_forward_hook(self.record_momentum),
            taal.memory_projection_out.register_forward_hook(self.record_correction),
            taal.register_forward_hook(self.record_post_addition),
        ]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def event(self, position: int) -> dict:
        return self.events.setdefault(position, {
            "layer": self.layer_index,
            "position": position,
        })

    def record_values(self, module, args, output) -> None:
        self.values = F.silu(output[0]).float()
        self.value_offset = 0

    def record_write_strength(self, module, args, output) -> None:
        position = self.write_position
        if position in self.positions:
            self.event(position)["write_strength"] = float(output[0, 0].sigmoid())

    def record_memory_mlp(self, module, args, kwargs, output) -> None:
        keys = args[0]
        _, count, _ = keys.shape
        is_write = kwargs.get("return_intermediates", False)
        if is_write:
            assert self.values is not None
            for index in range(count):
                position = self.write_position + index
                if position in self.positions:
                    event = self.event(position)
                    event["weight_norm_before"] = stable_weight_norm(kwargs["weights"])
                    event["key_norm"] = float(keys[0, index].float().norm())
                    error = (
                        output.predicted_values[0, index].float()
                        - self.values[0, self.value_offset + index]
                    )
                    event["value_prediction_error_norm"] = float(error.norm())
            self.write_position += count
            self.value_offset += count
        else:
            for index in range(count):
                position = self.read_position + index
                if position in self.positions:
                    self.event(position)["raw_memory_read_norm"] = float(
                        output.predicted_values[0, index].float().norm()
                    )
            self.read_position += count

    def record_forget(self, module, args, output) -> None:
        position = self.write_position - 1
        if position in self.positions:
            self.event(position)["forget_coefficient"] = float(output[0, 0].sigmoid())

    def record_momentum(self, module, args, output) -> None:
        position = self.write_position - 1
        if position in self.positions:
            self.event(position)["momentum_retention"] = float(output[0, 0].sigmoid())

    def record_correction(self, module, args, output) -> None:
        gate = torch.tanh(self.taal.residual_gate)
        for index in range(output.shape[1]):
            position = self.text_position + index
            if position in self.positions:
                correction = output[0, index]
                injection = self.read_scale * gate * correction
                event = self.event(position)
                event["projected_correction_norm"] = float(
                    correction.float().norm()
                )
                event["residual_gate"] = float(gate.float())
                event["pre_add_injection_norm"] = float(injection.float().norm())

    def record_post_addition(self, module, args, output) -> None:
        incoming = args[0]
        outgoing = output[0]
        for index in range(incoming.shape[1]):
            position = self.text_position + index
            if position in self.positions:
                before = incoming[0, index].float()
                after = outgoing[0, index].float()
                event = self.event(position)
                event["incoming_norm"] = float(before.norm())
                event["post_add_delta_norm"] = float((after - before).norm())
        self.text_position += incoming.shape[1]


@contextmanager
def forget_override(model, value: float | None):
    if value is None:
        yield
        return
    if not 0 < value < 1:
        raise ValueError("forget override must be strictly between zero and one")
    saved = []
    with torch.no_grad():
        for layer in model.model.layers:
            projection = layer.taal.neural_memory.forget_projection
            saved.append((projection, projection.weight.clone(), projection.bias.clone()))
            projection.weight.zero_()
            projection.bias.fill_(torch.logit(torch.tensor(value)).item())
    try:
        yield
    finally:
        with torch.no_grad():
            for projection, weight, bias in saved:
                projection.weight.copy_(weight)
                projection.bias.copy_(bias)


@torch.inference_mode()
def inspect_episode(model, example: dict, tokenizer, *, forget: float | None) -> dict:
    if model.config.memory_chunk_size != 1:
        raise ValueError("The per-token mechanism probe requires memory_chunk_size=1")
    prefix_ids, query_ids = split_episode(example)
    fact_position = int(example.get("fact_position", 0))
    query_position = len(prefix_ids)
    positions = {
        0, fact_position, fact_position + 1,
        query_position // 4, query_position // 2,
        max(0, query_position - 1), query_position,
        len(example["input_ids"]) - 1,
    }
    probes = [
        LayerProbe(index, layer.taal, positions, 1.0)
        for index, layer in enumerate(model.model.layers)
    ]
    layers = list(range(len(probes)))
    try:
        with forget_override(model, forget):
            prefix_trace = TaalTraceRecorder(model, layers, len(prefix_ids))
            with prefix_trace:
                prefix = model.prefill(
                    torch.tensor([prefix_ids], dtype=torch.long, device=model.device),
                    execution_block_size=model.config.working_memory_size,
                )
            query_trace = TaalTraceRecorder(
                model, layers, len(query_ids), position_offset=query_position
            )
            with query_trace:
                output = model.prefill(
                    torch.tensor([query_ids], dtype=torch.long, device=model.device),
                    execution_block_size=model.config.working_memory_size,
                    state=prefix.state,
                    prepend_memory_tokens=False,
                )
        logits = output.logits[0, -1].float()
        score = score_logits(logits, example, tokenizer)
        events = [
            probe.events[position]
            for probe in probes
            for position in sorted(probe.events)
        ]
        event_index = {(event["layer"], event["position"]): event for event in events}
        for write in prefix_trace.writes + query_trace.writes:
            if write.position in positions:
                event_index[(write.layer, write.position)].update({
                    "proposed_write_norm": write.proposed_write_norm,
                    "net_weight_change_norm": write.net_weight_change_norm,
                })
        return {
            "example_id": example["example_id"],
            "forget_override": forget,
            "positions": sorted(positions),
            "outcome": score,
            "events": events,
        }
    finally:
        for probe in probes:
            probe.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--example-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-forget", type=float, default=0.0001)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("The checkpoint probe requires a CUDA GPU")
    metadata = json.loads(args.dataset.with_suffix(".metadata.json").read_text())
    examples = (
        json.loads(line) for line in args.dataset.read_text().splitlines()
    )
    example = next(
        (row for row in examples if row["example_id"] == args.example_id), None
    )
    if example is None:
        parser.error(f"No example named {args.example_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"], revision=metadata["tokenizer_revision"]
    )
    model = load_model(args.checkpoint).to("cuda").eval()
    print(f"Loaded {args.checkpoint} on {torch.cuda.get_device_name(0)}", flush=True)
    baseline = inspect_episode(model, example, tokenizer, forget=None)
    print("Completed checkpoint-control replay", flush=True)
    low_forget = inspect_episode(
        model, example, tokenizer, forget=args.compare_forget
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "replays": [baseline, low_forget],
    }, indent=2) + "\n")
    print(f"Saved mechanism probe to {args.output}", flush=True)


if __name__ == "__main__":
    main()
