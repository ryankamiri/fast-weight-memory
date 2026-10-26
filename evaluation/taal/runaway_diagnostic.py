"""Dense scalar observations and bounded replay slices, without altering memory math."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.taal.numerical_diagnostic import cpu_payload


def norm64(value):
    return value.detach().double().norm()


def weights_norm(weights):
    return torch.stack([norm64(value).square() for value in weights.values()]).sum().sqrt()


def candidate_reasons(row, initial_norm):
    """Screening thresholds locate replay candidates, not a causal onset verdict."""
    metrics = row["metrics"]
    before = metrics.get("weight_before")
    after = metrics.get("weight_after")
    reasons = []
    if before is not None and after is not None:
        if after >= 2 * max(before, 1e-30) and after >= 10 * max(initial_norm, 1e-30):
            reasons.append("weight_growth")
    before_loss = metrics.get("same_key_mse_before")
    after_loss = metrics.get("same_key_mse_after")
    if before_loss is not None and after_loss is not None:
        if after_loss > 1 and after_loss >= 100 * max(before_loss, 1e-30):
            reasons.append("same_key_loss_growth")
    incoming = metrics.get("hidden_in")
    injection = metrics.get("injection")
    if incoming is not None and injection is not None:
        if injection >= 10 * max(incoming, 1e-30):
            reasons.append("read_dominates_residual")
    return reasons


@dataclass
class UpdateCapture:
    identity: dict
    step: tuple
    metrics: dict


class RunawayRecorder:
    """Record every C1 update and residual read; retain only bounded state slices."""

    def __init__(self, monitor, output_dir, tail_size=8, allow_transition_ablation=False):
        if tail_size < 1:
            raise ValueError("tail_size must be positive")

        self.monitor = monitor
        self.allow_transition_ablation = allow_transition_ablation
        self.output_dir = Path(output_dir)
        self.tail_size = tail_size
        self.stack = ExitStack()
        self.pending = {}
        self.tails = {}
        self.cursor = {}
        self.current = {}
        self.corrections = {}
        self.initial_norms = {}
        self.candidates = []
        self.seen = set()
        self.adapters_saved = set()
        self.update_count = 0
        self.read_count = 0
        self.stream = None

    def _begin(self, layer):
        def begin(adapter, args, kwargs):
            self.cursor[layer] = -adapter.config.num_persistent_tokens if kwargs.get("prepend_memory_tokens", True) else 0
            self.pending[layer] = []
            self.corrections.pop(layer, None)
        return begin

    def _identity(self, layer, cursor):
        block = self.monitor.blocks[layer]
        internal = cursor < 0
        position = None if internal else block.start + cursor
        return {
            "layer": layer, "block_start": block.start, "position": position,
            "internal_index": cursor + self.monitor.model.config.num_persistent_tokens if internal else None,
            "token_id": None if internal else block.token_ids[cursor],
            "phase": "internal" if internal else ("prompt" if position < self.monitor.prompt_length else "generated"),
        }

    def _gradient(self, layer, original):
        def gradient(weights, keys, values, strength, **kwargs):
            result = original(weights, keys, values, strength, **kwargs)
            current = self.current[layer]
            current["gradient"] = result.chunk_gradients
            current["strength"] = strength.detach()
            return result
        return gradient

    def _prediction(self, layer):
        def prediction(module, args, kwargs, result):
            if kwargs.get("return_intermediates", False) and layer in self.current:
                self.current[layer]["prediction"] = result.predicted_values.detach()
        return prediction

    def _control(self, layer, name):
        def control(module, args, output):
            self.current[layer][name] = output.detach().sigmoid().float().reshape(())
        return control

    def _correction(self, layer):
        def correction(module, args, output):
            self.corrections[layer] = output.detach()
        return correction

    def _process(self, layer, memory, original):
        def process(state, queries, keys, values, inputs, mask, output_dtype):
            B, C, D = keys.shape
            if B != 1 or C != 1 or state.pending_count:
                raise ValueError("Runaway diagnostic requires B=1, C=1 and no pending writes")

            self.current[layer] = {}
            step = (state, queries, keys, values, inputs, mask, output_dtype)
            read, next_state = original(*step)
            observed = self.current.pop(layer)
            # This extra read probes the just-written key; it is never fed to the model.
            after_prediction = memory.memory_mlp(keys.float(), weights=next_state.weights, return_intermediates=True).predicted_values
            before_mse = (observed["prediction"].double() - values.double()).square().mean()
            after_mse = (after_prediction.double() - values.double()).square().mean()
            delta = {name: next_state.weights[name].double() - weight.double() for name, weight in state.weights.items()}
            metrics = {
                "weight_before": weights_norm(state.weights), "weight_after": weights_norm(next_state.weights),
                "gradient": weights_norm(observed["gradient"]), "weight_delta": weights_norm(delta),
                "momentum_before": weights_norm(state.momentum), "momentum_after": weights_norm(next_state.momentum),
                "write_strength": observed["strength"].reshape(()).double(),
                "forget": observed["forget"].double(), "momentum_retention": (observed["retention"] if memory.momentum_enabled else torch.zeros_like(observed["retention"])).double(),
                "same_key_mse_before": before_mse, "same_key_mse_after": after_mse,
                "query": norm64(queries), "key": norm64(keys), "value": norm64(values),
                "memory_input": norm64(inputs), "raw_read": norm64(read),
            }
            for index, name in enumerate(state.weights):
                metrics[f"w{index}_before"] = norm64(state.weights[name])
                metrics[f"w{index}_after"] = norm64(next_state.weights[name])
                metrics[f"g{index}"] = norm64(observed["gradient"][name])
            identity = {**self._identity(layer, self.cursor[layer]), "update_order": self.update_count}
            capture = UpdateCapture(identity, step, metrics)
            self.pending[layer].append(capture)
            self.tails[layer].append(capture)
            self.cursor[layer] += 1
            self.update_count += 1
            return read, next_state
        return process

    def _save_slice(self, layer, capture, name):
        adapter = self.monitor.model.model.layers[layer].taal
        if layer not in self.adapters_saved:
            torch.save(cpu_payload(adapter.state_dict()), self.output_dir / f"adapter-{layer}.pt")
            self.adapters_saved.add(layer)
        path = self.output_dir / f"{name}.pt"
        torch.save({"identity": capture.identity, "step": cpu_payload(capture.step)}, path)
        return path.name

    def _write_rows(self, layer, captures):
        if not captures:
            return

        names = list(captures[0].metrics)
        # One host transfer per adapter block, not a synchronization per scalar.
        packed = torch.stack([torch.stack([capture.metrics[name] for name in names]) for capture in captures]).cpu().tolist()
        for capture, values in zip(captures, packed):
            row = {**capture.identity, "metrics": {name: value if math.isfinite(value) else None for name, value in zip(names, values)},
                   "nonfinite_metrics": {name: str(value) for name, value in zip(names, values) if not math.isfinite(value)}}
            self.stream.write(json.dumps(row, allow_nan=False) + "\n")
            for reason in candidate_reasons(row, self.initial_norms[layer]):
                key = (layer, row["phase"], reason)
                if key not in self.seen:
                    self.seen.add(key)
                    filename = self._save_slice(layer, capture, f"candidate-{layer}-{row['phase']}-{reason}")
                    self.candidates.append({**row, "reason": reason, "slice": filename})
        self.stream.flush()

    def _finish(self, layer):
        def finish(adapter, args, output):
            captures = self.pending[layer]
            block = self.monitor.blocks[layer]
            correction = self.corrections.get(layer)
            if correction is not None:
                scale = block.kwargs.get("memory_read_scale", 1.0)
                injection = scale * torch.tanh(adapter.residual_gate) * correction
                for capture in captures:
                    position = capture.identity["position"]
                    if position is not None:
                        offset = position - block.start
                        capture.metrics.update(hidden_in=norm64(block.hidden_states[:, offset]),
                                               injection=norm64(injection[:, offset]),
                                               hidden_out=norm64(output[0][:, offset]))
                        self.read_count += 1
            # Internal rows have no Qwen residual injection and remain separate.
            internal = [capture for capture in captures if capture.identity["position"] is None]
            text = [capture for capture in captures if capture.identity["position"] is not None]
            self._write_rows(layer, internal)
            self._write_rows(layer, text)
            self.pending[layer] = []
        return finish

    def __enter__(self):
        model = self.monitor.model
        if model.training or model.config.memory_chunk_size != 1:
            raise ValueError("Runaway diagnostic requires eval mode and C1")

        for layer in model.model.layers:
            memory = layer.taal.neural_memory
            if not self.allow_transition_ablation and not all((memory.updates_enabled, memory.forgetting_enabled, memory.momentum_enabled, layer.taal.reads_enabled)):
                raise ValueError("Runaway diagnostic expects the unmodified recurrence and enabled reads")

        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.stream = self.stack.enter_context((self.output_dir / "updates.jsonl").open("w"))
        for index, layer in enumerate(model.model.layers):
            adapter = layer.taal
            memory = adapter.neural_memory
            self.tails[index] = deque(maxlen=self.tail_size)
            self.initial_norms[index] = float(weights_norm(dict(memory.memory_mlp.named_parameters())))
            self.stack.enter_context(patch.object(memory, "_process_chunk", self._process(index, memory, memory._process_chunk)))
            self.stack.enter_context(patch.object(memory, "_chunk_gradient", self._gradient(index, memory._chunk_gradient)))
            handles = [
                adapter.register_forward_pre_hook(self._begin(index), with_kwargs=True),
                adapter.register_forward_hook(self._finish(index), prepend=True),
                memory.memory_mlp.register_forward_hook(self._prediction(index), with_kwargs=True),
                memory.forget_projection.register_forward_hook(self._control(index, "forget")),
                memory.momentum_projection.register_forward_hook(self._control(index, "retention")),
                adapter.memory_projection_out.register_forward_hook(self._correction(index)),
            ]
            for handle in handles:
                self.stack.callback(handle.remove)
        return self

    def __exit__(self, *error):
        try:
            for layer, captures in self.pending.items():
                self._write_rows(layer, captures)
            tail_files = []
            for layer, captures in self.tails.items():
                for index, capture in enumerate(captures):
                    tail_files.append(self._save_slice(layer, capture, f"tail-{layer}-{index}"))
            summary = {
                "updates": self.update_count, "reads": self.read_count,
                "initial_weight_norms": self.initial_norms, "tail_size_per_layer": self.tail_size,
                "tail_slices": tail_files, "first_candidates_per_layer_phase": self.candidates,
                "interpretation": "Threshold crossings are replay candidates, not proven causes. All update rows are retained.",
                "ordering": "update_order is actual block/layer execution order; position excludes internal tokens.",
                "exception": None if error[0] is None else error[0].__name__,
                "norms": "FP64 observation reductions only; model math and dtypes unchanged.",
            }
            (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        finally:
            self.stack.__exit__(*error)
            self.tails.clear()
            self.pending.clear()
            self.current.clear()
