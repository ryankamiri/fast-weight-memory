"""Compare identical tokens through full and cached TaaL without optimizer updates."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

from jaxtyping import Float, Int
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from evaluation.shared.retrieval_diagnostics import answer_loss, compare_logits
from evaluation.taal.numerical_diagnostic import cpu_payload


PRECISIONS = ("bf16_auto", "bf16_math", "fp32_auto", "fp32_math")
PATHS = ("full_eval", "full_train", "full_cache", "blocked", "blocked_alternate")


@contextmanager
def execution_precision(device, precision):
    if precision not in PRECISIONS:
        raise ValueError(f"Unknown precision: {precision}")

    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        # FP32 controls must not silently retain TF32 matrix multiplication.
        if precision.startswith("fp32"):
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        attention = sdpa_kernel(SDPBackend.MATH) if precision.endswith("math") else nullcontext()
        with attention, torch.autocast(device.type, dtype=torch.bfloat16,
                                      enabled=precision.startswith("bf16")):
            yield {
                "precision": precision,
                "autocast": torch.is_autocast_enabled(device.type),
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "sdpa_enabled": {
                    "math": torch.backends.cuda.math_sdp_enabled(),
                    "flash": torch.backends.cuda.flash_sdp_enabled(),
                    "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
                },
            }
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@dataclass
class ExecutionResult:
    logits: Float[torch.Tensor, "T V"]
    memory: dict
    kv: dict
    tokens_seen: int
    features: dict[str, Float[torch.Tensor, "S D"]] = field(default_factory=dict)
    feature_dtypes: dict[str, list[str]] = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    fused_loss: float | None = None


class ExecutionRecorder:
    """Align module outputs by text position; persistent memory tokens stay separate."""

    def __init__(self, model, token_count):
        self.model = model
        self.token_count = token_count
        self.start = 0
        self.length = 0
        self.chunks = {}
        self.dtypes = {}
        self.handles = []
        self.features = {}
        self.expected_keys = set()

    def begin(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None:
            ids = args[0]
        state = kwargs.get("state")
        self.start = 0 if state is None else state.tokens_seen
        self.length = ids.shape[1]
        if ids.shape[0] != 1:
            raise ValueError("Execution recorder requires B=1")

    def observe(self, key, before=False, internal=False):
        def capture(module, args, output=None):
            value = args[0] if before else output
            if isinstance(value, tuple):
                value = value[0]
            if internal:
                value = value[:, -self.length:]
            if value.ndim != 3 or value.shape[:2] != (1, self.length):
                raise ValueError(f"Unexpected token shape at {key}: {tuple(value.shape)}")

            # Copy only detached values; hooks never change a forward result.
            self.chunks.setdefault(key, []).append((self.start, value[0].detach().cpu().clone()))
            self.dtypes.setdefault(key, []).append(str(value.dtype))
        return capture

    def __enter__(self):
        self.handles.append(self.model.model.register_forward_pre_hook(self.begin, with_kwargs=True))
        for index, layer in enumerate(self.model.model.layers):
            memory = layer.taal.neural_memory
            stages = [
                ("input", layer.taal, True, False),
                ("memory_projection_in", layer.taal.memory_projection_in, False, False),
                ("memory_query_projection", memory.query_projection, False, True),
                ("memory_query_conv", memory.query_conv, False, True),
                ("memory_key_projection", memory.key_projection, False, True),
                ("memory_key_conv", memory.key_conv, False, True),
                ("memory_value_projection", memory.value_projection, False, True),
                ("memory_value_conv", memory.value_conv, False, True),
                ("memory_read", memory, False, True),
                ("memory_projection_out", layer.taal.memory_projection_out, False, False),
                ("memory_residual", layer.taal, False, False),
                ("attention_norm", layer.input_layernorm, False, False),
                ("attention_q", layer.self_attn.q_proj, False, False),
                ("attention_k", layer.self_attn.k_proj, False, False),
                ("attention_v", layer.self_attn.v_proj, False, False),
                ("attention_output", layer.self_attn, False, False),
                ("mlp_norm", layer.post_attention_layernorm, False, False),
                ("mlp_gate", layer.mlp.gate_proj, False, False),
                ("mlp_up", layer.mlp.up_proj, False, False),
                ("mlp_output", layer.mlp, False, False),
                ("decoder_output", layer, False, False),
            ]
            for stage, (name, module, before, internal) in enumerate(stages):
                key = f"L{index:02}/{stage:02}-{name}"
                self.expected_keys.add(key)
                hook = self.observe(key, before, internal)
                if before:
                    self.handles.append(module.register_forward_pre_hook(hook))
                else:
                    self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        if exc_type is not None:
            return False
        if set(self.chunks) != self.expected_keys:
            raise ValueError("Execution trace is missing an expected layer or stage")

        for key, chunks in self.chunks.items():
            cursor = 0
            values = []
            for start, value in chunks:
                if start != cursor:
                    raise ValueError(f"Missing or duplicated positions at {key}: {start} vs {cursor}")
                values.append(value)
                cursor += value.shape[0]
            if cursor != self.token_count:
                raise ValueError(f"Incomplete token coverage at {key}")
            self.features[key] = torch.cat(values).float()
        return False


def execute_path(model, row, path, precision, read_scale=1.0, record=True) -> ExecutionResult:
    if path not in PATHS:
        raise ValueError(f"Unknown execution path: {path}")
    if model.config.memory_chunk_size != 1:
        raise ValueError("This diagnostic preserves the original chunk1 recipe")

    device = next(model.parameters()).device
    ids: Int[torch.Tensor, "1 S"] = torch.tensor([row["input_ids"]], device=device)
    prompt_length = len(row["prompt_input_ids"])
    answer: Int[torch.Tensor, "T"] = ids[0, prompt_length:]
    if len(answer) < 2 or ids[0, :prompt_length].tolist() != row["prompt_input_ids"]:
        raise ValueError("Expected exact prompt followed by label and EOS")

    was_training = model.training
    training = path == "full_train"
    model.train(training)
    recorder = ExecutionRecorder(model, ids.shape[1])
    recording = recorder if record else nullcontext()
    gradient_mode = torch.enable_grad() if training else torch.inference_mode()
    try:
        with gradient_mode, execution_precision(device, precision) as settings, recording:
            fused_loss = None
            if path.startswith("full"):
                labels = torch.tensor([row["labels"]], device=device) if training else None
                output = model(ids, labels=labels, use_cache=path == "full_cache",
                               logits_to_keep=len(answer) + 1, memory_read_scale=read_scale)
                logits = output.logits[0, :-1].detach().float().cpu().clone()
                if output.loss is not None:
                    fused_loss = float(output.loss.detach())
            else:
                block_size = model.config.working_memory_size
                if path == "blocked_alternate":
                    block_size = min(37, max(1, block_size - 1))
                output = model.prefill(ids[:, :prompt_length], execution_block_size=block_size,
                                       memory_read_scale=read_scale)
                predictions = []
                for token in answer:
                    predictions.append(output.logits[0, -1].detach().float().cpu().clone())
                    # Consume EOS too, so final memory and KV correspond to identical tokens.
                    output = model(token.reshape(1, 1), state=output.state, use_cache=True,
                                   logits_to_keep=1, memory_read_scale=read_scale,
                                   prepend_memory_tokens=False)
                logits = torch.stack(predictions)
            state = output.state
            memory = cpu_payload(state.memory_states)
            kv = {}
            if state.past_key_values is not None:
                for index, layer in enumerate(state.past_key_values.layers):
                    kv[index] = {
                        "keys": layer.keys.detach().cpu().clone(),
                        "values": layer.values.detach().cpu().clone(),
                        "positions": layer.positions.detach().cpu().clone(),
                    }
            tokens_seen = state.tokens_seen
        return ExecutionResult(logits, memory, kv, tokens_seen, recorder.features,
                               recorder.dtypes, settings, fused_loss)
    finally:
        model.train(was_training)


def tensor_difference(reference: Float[torch.Tensor, "..."], actual: Float[torch.Tensor, "..."]):
    if reference.shape != actual.shape:
        raise ValueError(f"Comparison shape mismatch: {reference.shape} vs {actual.shape}")
    reference = reference.float()
    actual = actual.float()
    if not bool(torch.isfinite(reference).all() and torch.isfinite(actual).all()):
        raise ValueError("Non-finite tensor in execution comparison")

    difference = (actual - reference).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": float(difference.norm() / reference.norm().clamp_min(1e-12)),
        "equal": bool(torch.equal(actual, reference)),
        "close": bool(torch.allclose(actual, reference, atol=1e-5, rtol=1e-4)),
    }


def flatten_state(state, prefix=""):
    flattened = {}
    for key, value in state.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            flattened.update(flatten_state(value, name))
        elif isinstance(value, torch.Tensor):
            flattened[name] = value
        elif value is not None:
            flattened[name] = torch.tensor([value])
    return flattened


def compare_execution(reference, actual, labels):
    if reference.tokens_seen != actual.tokens_seen:
        raise ValueError("Compared paths consumed different token counts")
    if set(reference.features) != set(actual.features):
        raise ValueError("Compared paths have different observation stages")

    loss = answer_loss(reference.logits, labels).combined_nll
    metrics = compare_logits(reference.logits, actual.logits, labels, loss)
    stages = []
    snapshots = {}
    first_exact = None
    first_tolerance = None
    # Order by model layer, operation and then text position, not hook arrival order.
    for key in sorted(reference.features):
        left = reference.features[key]
        right = actual.features[key]
        summary = tensor_difference(left, right)
        difference = (right - left).abs()
        exact = (difference > 0).any(dim=-1)
        close = torch.isclose(left, right, atol=1e-5, rtol=1e-4).all(dim=-1)
        row = {"stage": key, **summary,
               "max_abs_by_token": difference.amax(dim=-1).tolist(),
               "outside_tolerance_by_token": (~close).tolist(),
               "reference_dtypes": reference.feature_dtypes[key],
               "actual_dtypes": actual.feature_dtypes[key]}
        stages.append(row)
        for label, mask in (("first_exact", exact), ("first_tolerance", ~close)):
            if label in snapshots or not bool(mask.any()):
                continue
            position = int(mask.nonzero()[0])
            identity = {"stage": key, "position": position,
                        **tensor_difference(left[position], right[position])}
            snapshots[label] = {"identity": identity, "reference": left[position], "actual": right[position]}
            if label == "first_exact":
                first_exact = identity
            else:
                first_tolerance = identity
    left_flat = flatten_state(reference.memory)
    right_flat = flatten_state(actual.memory)
    if set(left_flat) != set(right_flat):
        raise ValueError("Final memory state fields differ")
    memory_difference = {}
    for key in left_flat:
        memory_difference[key] = tensor_difference(left_flat[key], right_flat[key])
    metrics.update({"first_exact_difference": first_exact, "first_tolerance_difference": first_tolerance,
                    "observation_tolerance": {"atol": 1e-5, "rtol": 1e-4},
                    "ordering": "layer, operation, text position; first observed, not proof of root cause",
                    "final_state": {"memory": memory_difference}, "tokens_seen": actual.tokens_seen})
    return {"metrics": metrics, "stages": stages, "snapshots": snapshots}
