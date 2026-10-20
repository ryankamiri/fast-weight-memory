"""Observe ordinary Qwen sublayers on identical forced tokens without changing math."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path

from jaxtyping import Float, Int
import torch

from evaluation.taal.numerical_diagnostic import tensor_summary


@dataclass
class FixedReplaySource:
    manifest: dict
    example: dict
    continuation: list[int]
    reference: dict


def load_replay_source(directory, decode_tokens, layers):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    example = json.loads((directory / "example.json").read_text())
    prompt = example["input_ids"]
    if example["prompt_length"] != len(prompt) or decode_tokens < 1:
        raise ValueError("Invalid source prompt length or continuation count")

    observed = {}
    reference = {}
    with (directory / "reads-1/updates/updates.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            position = row["position"]
            if position is None:
                continue
            if row["layer"] == 0:
                if position in observed:
                    raise ValueError("Duplicate source token position")

                observed[position] = row["token_id"]
            if row["layer"] in layers:
                reference[(row["layer"], position)] = {
                    name: row["metrics"].get(name) for name in ("hidden_in", "hidden_out")
                }
    end = len(prompt) + decode_tokens
    if any(position not in observed for position in range(end)):
        raise ValueError("Source does not contain every requested forced token")
    if [observed[position] for position in range(len(prompt))] != prompt:
        raise ValueError("Source update tokens disagree with saved prompt")

    continuation = [observed[position] for position in range(len(prompt), end)]
    return FixedReplaySource(manifest, example, continuation, reference)


@torch.inference_mode()
def replay_fixed_tokens(model, prompt: Int[torch.Tensor, "1 S"], continuation: list[int],
                        block_size: int, read_scale: float):
    """Return diagnostic predictions, but always consume the supplied continuation."""
    _, S = prompt.shape
    output = model.prefill(prompt, execution_block_size=block_size, memory_read_scale=read_scale)
    predictions = []
    for index, token_id in enumerate(continuation):
        logits: Float[torch.Tensor, "V"] = output.logits[0, -1].float()
        predictions.append({
            "position": S + index, "forced_token_id": token_id,
            "predicted_token_id": int(logits.argmax()),
            "forced_token_log_probability": float(logits.log_softmax(-1)[token_id]),
        })
        token: Int[torch.Tensor, "1 1"] = prompt.new_tensor([[token_id]])
        output = model(token, state=output.state, use_cache=True, logits_to_keep=1,
                       memory_read_scale=read_scale, prepend_memory_tokens=False)
    return output, predictions


class DecoderRecorder:
    """Observe selected late tokens and save bounded ordinary-decoder replay slices."""

    def __init__(self, model, output_dir, layers, start, snapshot_positions=(), snapshot_layer=2):
        if model.training or model.config.memory_chunk_size != 1 or start < 0 or not layers or len(layers) != len(set(layers)):
            raise ValueError("Recorder requires eval mode, distinct layers and nonnegative start")
        if any(layer < 0 or layer >= len(model.model.layers) for layer in layers):
            raise ValueError("Recorder layer is out of range")

        self.model = model
        self.output_dir = Path(output_dir)
        self.layers = layers
        self.start = start
        self.snapshot_positions = set(snapshot_positions)
        self.snapshot_layer = snapshot_layer
        self.stack = ExitStack()
        self.current = {}
        self.rows = []
        self.handles = []
        self.position = 0
        self.token_ids = []
        self.active = False

    def _begin(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        B, S = ids.shape
        if B != 1:
            raise ValueError("Decoder diagnostic requires B=1")

        state = kwargs.get("state")
        self.position = 0 if state is None else state.tokens_seen
        self.token_ids = ids[0].tolist()
        self.active = self.position + S > self.start

    def _layer_begin(self, layer):
        def begin(module, args, kwargs):
            if not self.active:
                return

            cache = kwargs["past_key_values"]
            positions = kwargs["cache_position"]
            cached_layer = cache.layers[layer]
            keys, persistent = cached_layer.attention_metadata(positions, cache.persistent_mask(positions))
            self.current[layer] = {
                "vectors": {}, "controls": {name: [] for name in ("write", "forget", "momentum")},
                "persistent_count": module.taal.config.num_persistent_tokens if kwargs.get("prepend_memory_tokens", True) else 0,
                "cache": {
                    "seen_before": cached_layer.cumulative_length,
                    "query_positions": positions.tolist(), "key_positions": keys.tolist(),
                    "persistent": persistent.tolist(),
                    "mask_allowed": (kwargs["attention_mask"][0, 0] == 0).tolist(),
                    "keys_before": None if cached_layer.keys is None else tensor_summary(cached_layer.keys),
                    "values_before": None if cached_layer.values is None else tensor_summary(cached_layer.values),
                },
            }
            if layer == self.snapshot_layer and any(int(pos) in self.snapshot_positions for pos in positions):
                self.current[layer]["snapshot"] = {
                    "block_start": self.position, "token_ids": list(self.token_ids),
                    "cache_position": positions.detach().cpu().clone(),
                    "position_ids": kwargs["position_ids"].detach().cpu().clone(),
                    "position_embeddings": tuple(value.detach().cpu().clone() for value in kwargs["position_embeddings"]),
                    "attention_mask": kwargs["attention_mask"].detach().cpu().clone(),
                    "keys_before": None if cached_layer.keys is None else cached_layer.keys.detach().cpu().clone(),
                    "values_before": None if cached_layer.values is None else cached_layer.values.detach().cpu().clone(),
                    "key_positions": keys.detach().cpu().clone(),
                }
        return begin

    def _observe(self, layer, name, before=False):
        def observe(module, args, output=None):
            if not self.active:
                return

            value = args[0] if before else output
            if isinstance(value, tuple):
                value = value[0]
            self.current[layer]["vectors"][name] = value.detach()
        return observe

    def _control(self, layer, name):
        def observe(module, args, output):
            if self.active:
                self.current[layer]["controls"][name].append(output.detach().sigmoid().reshape(-1))
        return observe

    def _finish(self, layer):
        def finish(module, args, output):
            if not self.active:
                return

            current = self.current.pop(layer)
            vectors = current["vectors"]
            vectors["decoder_output"] = output[0].detach()
            vectors["memory_delta"] = vectors["memory_output"] - vectors["memory_input"]
            vectors["attention_recomposition_error"] = vectors["attention_residual"] - (
                vectors["memory_output"] + vectors["attention_output"])
            vectors["mlp_recomposition_error"] = vectors["decoder_output"] - (
                vectors["attention_residual"] + vectors["mlp_output"])
            controls = {name: torch.cat(values).cpu().tolist() for name, values in current["controls"].items()}
            for offset, token_id in enumerate(self.token_ids):
                position = self.position + offset
                if position < self.start:
                    continue

                row = {
                    "layer": layer, "position": position, "token_id": token_id,
                    "block_start": self.position,
                    "components": {name: tensor_summary(value[0, offset]) for name, value in vectors.items()},
                    "controls": {name: values[offset + current["persistent_count"]]
                                 if math.isfinite(values[offset + current["persistent_count"]]) else None
                                 for name, values in controls.items()},
                    "cache": {**current["cache"], "allowed_positions": [
                        pos for pos, allowed in zip(current["cache"]["key_positions"],
                                                   current["cache"]["mask_allowed"][offset]) if allowed]},
                }
                self.rows.append(row)
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            self.stream.flush()
            if "snapshot" in current:
                current["snapshot"]["vectors"] = {name: value.cpu().clone() for name, value in vectors.items()}
                torch.save(current["snapshot"], self.output_dir / f"layer-{layer}-position-{self.position}.pt")
                weights_path = self.output_dir / f"decoder-{layer}.pt"
                if not weights_path.exists():
                    torch.save({name: value.detach().cpu().clone() for name, value in module.state_dict().items()
                                if not name.startswith("taal.")}, weights_path)
        return finish

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.stream = self.stack.enter_context((self.output_dir / "components.jsonl").open("w"))
        self.handles.append(self.model.model.register_forward_pre_hook(self._begin, with_kwargs=True))
        for index in self.layers:
            layer = self.model.model.layers[index]
            self.handles.append(layer.register_forward_pre_hook(self._layer_begin(index), with_kwargs=True))
            outputs = {
                "memory_output": layer.taal, "attention_input": layer.input_layernorm,
                "attention_q_proj": layer.self_attn.q_proj, "attention_k_proj": layer.self_attn.k_proj,
                "attention_v_proj": layer.self_attn.v_proj, "attention_q_norm": layer.self_attn.q_norm,
                "attention_k_norm": layer.self_attn.k_norm, "attention_output": layer.self_attn,
                "mlp_input": layer.post_attention_layernorm, "mlp_gate": layer.mlp.gate_proj,
                "mlp_up": layer.mlp.up_proj, "mlp_output": layer.mlp,
            }
            inputs = {
                "attention_context": layer.self_attn.o_proj,
                "attention_residual": layer.post_attention_layernorm, "mlp_product": layer.mlp.down_proj,
            }
            # The TaaL input is captured by its positional hidden-state argument.
            self.handles.append(layer.taal.register_forward_pre_hook(self._observe(index, "memory_input", True)))
            for name, module in outputs.items():
                self.handles.append(module.register_forward_hook(self._observe(index, name)))
            for name, module in inputs.items():
                self.handles.append(module.register_forward_pre_hook(self._observe(index, name, True)))
            for name, projection in (
                ("write", layer.taal.neural_memory.write_strength_projection),
                ("forget", layer.taal.neural_memory.forget_projection),
                ("momentum", layer.taal.neural_memory.momentum_projection),
            ):
                self.handles.append(projection.register_forward_hook(self._control(index, name)))
            self.handles.append(layer.register_forward_hook(self._finish(index), prepend=True))
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        partial = {str(layer): {name: tensor_summary(value) for name, value in item["vectors"].items()}
                   for layer, item in self.current.items()}
        (self.output_dir / "summary.json").write_text(json.dumps({
            "rows": len(self.rows), "layers": self.layers, "start": self.start,
            "exception": None if error[0] is None else error[0].__name__, "partial_components": partial,
        }, indent=2, allow_nan=False) + "\n")
        self.stack.__exit__(*error)
