"""Evaluation-only block capture, exact-operation replay, and independent write checks."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
import traceback
from unittest.mock import patch

from jaxtyping import Float
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value] if value.is_floating_point() else []
    if is_dataclass(value):
        return [tensor for field in fields(value) for tensor in tensors(getattr(value, field.name))]
    if isinstance(value, dict):
        return [tensor for child in value.values() for tensor in tensors(child)]
    if isinstance(value, (tuple, list)):
        return [tensor for child in value for tensor in tensors(child)]
    return []


def all_finite(value):
    found = tensors(value)
    return not found or bool(torch.stack([torch.isfinite(tensor).all() for tensor in found]).all())


def tensor_summary(value):
    detached = value.detach().double()
    mask = torch.isfinite(detached)
    valid = detached[mask]
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "nan": int(detached.isnan().sum()), "inf": int(detached.isinf().sum()),
        "max_abs_finite": float(valid.abs().max()) if valid.numel() else None,
        "norm_finite": float(valid.norm()) if valid.numel() else None,
        "first_bad_index": (~mask).nonzero()[0].tolist() if not bool(mask.all()) else None,
    }


def cpu_payload(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if is_dataclass(value):
        return {field.name: cpu_payload(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {key: cpu_payload(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_payload(child) for child in value)
    if isinstance(value, list):
        return [cpu_payload(child) for child in value]
    return value


class NumericalFailure(RuntimeError):
    def __init__(self, report):
        super().__init__(f"First observed non-finite output: {report.get('operation', report.get('stage'))}")
        self.report = report


@dataclass
class MemoryBlock:
    layer: int
    start: int
    token_ids: list[int]
    hidden_states: Float[torch.Tensor, "1 S D_model"]
    kwargs: dict


class BlockMonitor:
    """Check production block boundaries without changing operations or tensor values."""

    def __init__(self, model, prompt_length):
        self.model = model
        self.prompt_length = prompt_length
        self.start = 0
        self.token_ids = []
        self.blocks = {}
        self.last_block = None
        self.recent = deque(maxlen=128)
        self.samples = []
        self.handles = []

    def _begin(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        state = kwargs.get("state")
        self.start = 0 if state is None else state.tokens_seen
        self.token_ids = ids[0].tolist()
        if self.start % 256 == 0:
            print(f"NUMERICAL_PROGRESS position={self.start} phase={'prompt' if self.start < self.prompt_length else 'generated'}", flush=True)

    def _capture(self, layer):
        def capture(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            options = {key: value for key, value in kwargs.items() if key != "hidden_states"}
            self.blocks[layer] = MemoryBlock(layer, self.start, self.token_ids, hidden, options)
        return capture

    def _check(self, stage, layer):
        def check(module, args, output):
            good = all_finite(output)
            event = {"layer": layer, "start": self.start, "length": len(self.token_ids), "stage": stage, "finite": good}
            self.recent.append(event)
            if stage == "taal" and (self.start % 256 == 0 or not good):
                event = {**event, "tensors": [tensor_summary(value) for value in tensors(output)]}
                self.samples.append(event)
            if not good:
                self.last_block = self.blocks.get(layer)
                raise NumericalFailure({**event, "recent": list(self.recent)})
        return check

    def __enter__(self):
        if self.model.training or self.model.config.memory_chunk_size != 1:
            raise ValueError("Numerical diagnostics require evaluation and memory chunk1")

        for name, parameter in self.model.named_parameters():
            if not all_finite(parameter):
                raise NumericalFailure({"stage": "checkpoint_parameter", "parameter": name, "tensor": tensor_summary(parameter)})

        self.handles.append(self.model.model.register_forward_pre_hook(self._begin, with_kwargs=True))
        for index, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.taal.register_forward_pre_hook(self._capture(index), with_kwargs=True))
            for stage, module in (
                ("taal", layer.taal), ("attention", layer.self_attn),
                ("qwen_mlp", layer.mlp), ("decoder", layer),
            ):
                self.handles.append(module.register_forward_hook(self._check(stage, index)))
        self.handles.append(self.model.lm_head.register_forward_hook(self._check("logits", None)))
        return self

    def __exit__(self, *error):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FirstNonfinite(TorchDispatchMode):
    """Inspect actual ATen outputs only during bounded replay, not the full episode."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.position = None
        self.internal_index = None
        self.step = None
        self.steps = deque(maxlen=16)
        self.module_stack = []
        self.operations = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        self.operations += 1
        if not all_finite(result):
            stack = [f"{entry.filename}:{entry.lineno} {entry.line}" for entry in traceback.extract_stack()
                     if "/architectures/" in entry.filename]
            raise NumericalFailure({
                "operation": str(func), "operation_index": self.operations,
                "layer": self.block.layer, "block_start": self.block.start,
                "position": self.position, "internal_index": self.internal_index,
                "module_stack": list(self.module_stack), "source": stack,
                "inputs": [tensor_summary(value) for value in tensors((args, kwargs))],
                "outputs": [tensor_summary(value) for value in tensors(result)],
            })
        return result


def independent_gradient(weights, keys, values, strength, dtype):
    if keys.shape[0] != 1:
        raise ValueError("Independent diagnostic reference requires B=1")

    with torch.inference_mode(False), torch.enable_grad():
        reference = {name: value.detach().to(dtype=dtype).clone().requires_grad_(True) for name, value in weights.items()}
        predicted = keys.detach().to(dtype=dtype).clone()
        for index, weight in enumerate(reference.values()):
            predicted = F.linear(predicted[0], weight[0]).unsqueeze(0)
            if index + 1 < len(reference):
                predicted = F.silu(predicted)
        loss = ((predicted - values.detach().to(dtype=dtype).clone()).square().mean(-1)
                * strength.detach().to(dtype=dtype).clone()).sum()
        gradients = torch.autograd.grad(loss, tuple(reference.values()))
    return {"loss": tensor_summary(loss), "gradients": dict(zip(reference, gradients))}


def compare_tensors(actual, reference):
    result = {"actual": tensor_summary(actual), "reference": tensor_summary(reference)}
    if all_finite((actual, reference)):
        delta = actual.double() - reference.double()
        result.update(max_abs_error=float(delta.abs().max()),
                      relative_l2_error=float(delta.norm() / reference.double().norm().clamp_min(1e-30)))
    return result


@torch.no_grad()
def check_write(memory, step):
    state, queries, keys, values, inputs, mask, output_dtype = step
    if not all_finite((state, keys, values, inputs)):
        return {"status": "nonfinite_input_before_write", "no_finite_oracle_comparison": True}

    B, C, D = keys.shape
    if B != 1 or C != 1 or state.pending_count:
        raise ValueError("Write oracle requires B=1, C=1, and no pending chunk")

    strength = memory.write_strength_projection(inputs).sigmoid().reshape(B, C).float()
    strength = torch.where(mask, strength, 0.0)
    manual = memory._chunk_gradient(state.weights, keys, values, strength).chunk_gradients
    automatic = independent_gradient(state.weights, keys, values, strength, torch.float32)
    double = independent_gradient(state.weights, keys, values, strength, torch.float64)
    mean_input = inputs.float().mean(1).to(memory.forget_projection.weight.dtype)
    forget = memory.forget_projection(mean_input).sigmoid().reshape(B, 1, 1).float()
    momentum = memory.momentum_projection(mean_input).sigmoid().reshape(B, 1, 1).float()
    updated = memory._update(state, inputs, keys, values, mask)
    comparisons = {}
    for name, weight in state.weights.items():
        expected_momentum = momentum * state.momentum[name] - automatic["gradients"][name]
        expected_weight = (1 - forget) * weight + expected_momentum
        double_weight = ((1 - forget.double()) * weight.double() + momentum.double()
                         * state.momentum[name].double() - double["gradients"][name])
        comparisons[name] = {
            "manual_vs_autograd_fp32": compare_tensors(manual[name], automatic["gradients"][name]),
            "manual_vs_autograd_fp64": compare_tensors(manual[name], double["gradients"][name]),
            "updated_vs_autograd_fp32": compare_tensors(updated.weights[name], expected_weight),
            "updated_fp64": tensor_summary(double_weight),
        }
    return {"status": "compared", "write_strength": tensor_summary(strength),
            "forget": tensor_summary(forget), "momentum_retention": tensor_summary(momentum),
            "old_weights": {name: tensor_summary(value) for name, value in state.weights.items()},
            "old_momentum": {name: tensor_summary(value) for name, value in state.momentum.items()},
            "fp32_loss": automatic["loss"], "fp64_loss": double["loss"], "parameters": comparisons}


@torch.inference_mode()
def replay_block(adapter, block, snapshot_path=None):
    mode = FirstNonfinite(block)
    memory = adapter.neural_memory
    original = memory._process_chunk
    persistent = adapter.config.num_persistent_tokens if block.kwargs.get("prepend_memory_tokens", True) else 0
    cursor = -persistent

    def process(state, queries, keys, values, inputs, mask, output_dtype):
        nonlocal cursor
        mode.position = block.start + cursor if cursor >= 0 else None
        mode.internal_index = cursor + persistent if cursor < 0 else None
        mode.step = (state, queries, keys, values, inputs, mask, output_dtype)
        result = original(*mode.step)
        with _disable_current_modes():
            mode.steps.append({
                "position": mode.position, "internal_index": mode.internal_index,
                "before": [tensor_summary(value) for value in state.weights.values()],
                "after": [tensor_summary(value) for value in result[1].weights.values()],
                "read": tensor_summary(result[0]),
            })
        cursor += 1
        mode.position = None
        mode.internal_index = None
        return result

    def enter_module(name):
        def enter(module, args):
            mode.module_stack.append(name)
        return enter

    def exit_module(module, args, output):
        mode.module_stack.pop()

    failure = None
    with ExitStack() as stack:
        stack.enter_context(patch.object(memory, "_process_chunk", process))
        stack.enter_context(patch.object(memory, "trace_observer", None))
        stack.enter_context(patch.object(adapter, "trace_observer", None))
        for name, module in adapter.named_modules():
            handle = module.register_forward_pre_hook(enter_module(name or "adapter"))
            stack.callback(handle.remove)
            handle = module.register_forward_hook(exit_module, always_call=True)
            stack.callback(handle.remove)
        try:
            with mode:
                adapter(block.hidden_states, **block.kwargs)
        except NumericalFailure as error:
            failure = error.report
    result = {"reproduced": failure is not None, "first_nonfinite": failure,
              "preceding_finite_steps": list(mode.steps)}
    if mode.step is not None:
        result["write_oracle"] = check_write(memory, mode.step)
    if snapshot_path is not None:
        payload = {"block": cpu_payload(block), "adapter_state_dict": cpu_payload(adapter.state_dict()),
                   "step": cpu_payload(mode.step), "report": result}
        torch.save(payload, Path(snapshot_path))
    return result
