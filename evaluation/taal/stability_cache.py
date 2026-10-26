"""Compare rolling KV execution against independent full-history band-masked Qwen."""

import argparse
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.taal.decoder_diagnostic import load_replay_source
from evaluation.taal.longmemeval_numerics import write_report
from evaluation.taal.numerical_diagnostic import compare_tensors


def band_mask(S, window, device, dtype):
    positions = torch.arange(S, device=device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < window)
    mask = torch.zeros(S, S, device=device, dtype=dtype)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


def bypass_memory(adapter):
    def forward(hidden_states, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = adapter.neural_memory.initial_state(hidden_states.shape[0])
        return hidden_states, memory_state

    return forward


@torch.inference_mode()
def compare_cache(model, ids, block_sizes=(1, 32)):
    settings = Qwen3Config.from_dict(model.config.to_dict())
    settings._attn_implementation = "sdpa"
    reference = (
        Qwen3ForCausalLM(settings)
        .to(device=ids.device, dtype=next(model.parameters()).dtype)
        .eval()
    )
    host_weights = {
        name: value
        for name, value in model.state_dict().items()
        if ".taal." not in name
    }
    reference.load_state_dict(host_weights, strict=True)
    _, S = ids.shape
    mask = band_mask(
        S, model.config.working_memory_size, ids.device, next(model.parameters()).dtype
    )
    expected = reference(
        ids,
        attention_mask=mask,
        position_ids=torch.arange(S, device=ids.device)[None],
        use_cache=False,
        output_hidden_states=True,
        logits_to_keep=min(S, 128),
    )
    expected_hidden = [value.cpu() for value in expected.hidden_states]
    expected_logits = expected.logits.cpu()
    del expected, reference, mask
    results = {}
    for size in block_sizes:
        hidden = [[] for _ in expected_hidden]
        logits = []
        state = None
        with ExitStack() as stack:
            for layer in model.model.layers:
                stack.enter_context(
                    patch.object(layer, "forward_memory", bypass_memory(layer.taal))
                )
            for start in range(0, S, size):
                output = model(
                    ids[:, start : start + size],
                    state=state,
                    use_cache=True,
                    memory_read_scale=0.0,
                    output_hidden_states=True,
                    logits_to_keep=0,
                )
                state = output.state
                for index, value in enumerate(output.hidden_states):
                    hidden[index].append(value.cpu())
                if start + size > S - 128:
                    logits.append(output.logits.cpu())
        actual_logits = torch.cat(logits, dim=1)[:, -min(S, 128) :]
        layers = []
        for index, (pieces, target) in enumerate(zip(hidden, expected_hidden)):
            actual = torch.cat(pieces, dim=1)
            delta = (actual.double() - target.double()).norm(dim=-1)
            relative = delta / target.double().norm(dim=-1).clamp_min(1e-12)
            threshold = 0.05 if actual.dtype == torch.bfloat16 else 0.001
            bad = (relative > threshold).nonzero()
            layers.append(
                {
                    "layer_boundary": index,
                    "comparison": compare_tensors(actual, target),
                    "diagnostic_relative_threshold": threshold,
                    "first_above_threshold": None if not len(bad) else int(bad[0, 1]),
                }
            )
        results[str(size)] = {
            "layers": layers,
            "logits": compare_tensors(actual_logits, expected_logits),
            "tail_argmax_agreement": float(
                (actual_logits.argmax(-1) == expected_logits.argmax(-1)).float().mean()
            ),
            "tokens_seen": state.tokens_seen,
        }
    return results


def run_cache(checkpoint, source_run, output, device):
    output.mkdir(parents=True, exist_ok=False)
    source = load_replay_source(source_run, 96, [2])
    tokens = source.example["input_ids"] + source.continuation
    config = TaalQwen3Config.from_pretrained(checkpoint)
    write_report(
        output / "manifest.json",
        {
            "checkpoint": str(checkpoint),
            "source": str(source_run),
            "tokens": len(tokens),
            "memory": "bypassed identically; cache-only correctness test",
            "reference": "HF Qwen3 full history, independently constructed explicit local causal mask, no cache",
            "save_traces": True,
            "trace_kind": "per-layer numerical comparisons; no memory branch is evaluated",
            "interpretation": "Finite-precision differences are reported, not automatically classified as bugs.",
        },
    )
    for mode, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        model = (
            TaalQwen3ForCausalLM.from_pretrained(
                checkpoint, config=config, dtype=dtype, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        for length in (256, len(tokens)):
            ids = torch.tensor([tokens[:length]], dtype=torch.long, device=device)
            result = compare_cache(model, ids)
            write_report(output / f"{mode}-{length}.json", result)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_report(output / "completion.json", {"status": "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_cache(args.checkpoint, args.source_run, args.output_dir, torch.device("cuda"))


if __name__ == "__main__":
    main()
