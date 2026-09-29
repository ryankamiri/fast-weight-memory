"""Restartable, disk-backed pre-query snapshots for the TaaL state audit."""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import transformers

from architectures.shared.qwen.cache import SlidingWindowKVCache
from architectures.taal.qwen.state import NeuralMemoryStates, TaalModelState
from architectures.titans.state import NeuralMemoryState
from evaluation.storage import ensure_manifest
from evaluation.taal.trace_contract import InternalPrefix, ReadEvent, WriteEvent


@dataclass
class PrefixTrace:
    writes: list[WriteEvent]
    reads: list[ReadEvent]
    internal_prefixes: list[InternalPrefix]


@dataclass
class SavedPrefix:
    state: TaalModelState
    trace: PrefixTrace | None


def checkpoint_fingerprint(checkpoint: Path) -> str:
    """Check contents, not just the path: a checkpoint directory can be replaced."""
    files = sorted(
        path for path in checkpoint.iterdir()
        if path.is_file() and (
            path.suffix in (".safetensors", ".bin")
            or path.name == "config.json"
            or path.name.endswith(".index.json")
        )
    )
    if not files or not any(path.suffix in (".safetensors", ".bin") for path in files):
        raise ValueError("Prefix snapshots require a local checkpoint with model weights")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        with path.open("rb") as file:
            while block := file.read(8 * 1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _cpu_tensor(value: torch.Tensor | None, batch_index: int | None = None):
    if value is None:
        return None
    if batch_index is not None:
        value = value[batch_index:batch_index + 1]
    # Clone so a session slice does not serialize the entire batch's storage.
    return value.detach().to("cpu").clone().contiguous()


def _pack_memory(states: NeuralMemoryStates, batch_index: int) -> dict:
    packed = {}
    for layer_index, state in states.items():
        packed[layer_index] = {}
        for field in fields(state):
            value = getattr(state, field.name)
            if isinstance(value, dict):
                value = {
                    name: _cpu_tensor(tensor, batch_index)
                    for name, tensor in value.items()
                }
            elif isinstance(value, torch.Tensor):
                value = _cpu_tensor(value, batch_index)
            packed[layer_index][field.name] = value
    return packed


def _unpack_memory(packed: dict, device: torch.device | str) -> NeuralMemoryStates:
    states = {}
    for layer_index, values in packed.items():
        restored = {}
        for name, value in values.items():
            if isinstance(value, dict):
                value = {key: tensor.to(device).clone() for key, tensor in value.items()}
            elif isinstance(value, torch.Tensor):
                value = value.to(device).clone()
            restored[name] = value
        states[layer_index] = NeuralMemoryState(**restored)
    return states


class PrefixStore:
    """Store tensors and plain metadata, never pickled model/cache instances.

    Each file contains one episode's own KV, complete neural state, and optional
    prefix trace. A completed file can be reused after interruption without
    recomputing the long prefix. Query conditions always fork the restored state.
    """

    def __init__(self, directory: Path, identity: dict):
        self.directory = directory
        manifest = {
            "schema": "taal-audit-prefix/v1",
            "torch_version": str(torch.__version__),
            "transformers_version": transformers.__version__,
            **identity,
        }
        # Model configs can contain integer mapping keys or tuples. Compare the
        # JSON-normalized representation that is actually persisted on disk.
        ensure_manifest(directory / "manifest.json", json.loads(json.dumps(manifest)))

    def path(self, example: dict) -> Path:
        identity = json.dumps(example, sort_keys=True, separators=(",", ":"))
        return self.directory / f"{hashlib.sha256(identity.encode()).hexdigest()}.pt"

    def contains(self, example: dict) -> bool:
        return self.path(example).is_file()

    def save(
        self,
        example: dict,
        state: TaalModelState,
        trace: PrefixTrace | None,
        *,
        batch_index: int = 0,
    ) -> None:
        cache = state.past_key_values
        if not isinstance(cache, SlidingWindowKVCache):
            raise ValueError("Prefix snapshot requires a SlidingWindowKVCache")
        if any(not layer.is_initialized for layer in cache.layers):
            raise ValueError("Prefix snapshot requires initialized KV layers")
        B, _, _, _ = cache.layers[0].keys.shape
        if not 0 <= batch_index < B:
            raise ValueError("Prefix snapshot batch_index is outside the session batch")
        payload = {
            "example": example,
            "tokens_seen": state.tokens_seen,
            "memory_states": _pack_memory(state.memory_states, batch_index),
            "cache": {
                "window_size": cache.window_size,
                "max_persistent_kv_tokens": cache.max_persistent_kv_tokens,
                "persistent_positions": _cpu_tensor(cache.persistent_positions),
                "layers": [{
                    "keys": _cpu_tensor(layer.keys, batch_index),
                    "values": _cpu_tensor(layer.values, batch_index),
                    "positions": _cpu_tensor(layer.positions),
                    "is_persistent": _cpu_tensor(layer.is_persistent),
                    "cumulative_length": layer.cumulative_length,
                } for layer in cache.layers],
            },
            "trace": None if trace is None else asdict(trace),
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as file:
                temporary = Path(file.name)
                torch.save(payload, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path(example))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _load_payload(self, example: dict) -> dict:
        # mmap lets a swapped-state read access neural tensors without bringing
        # that other episode's entire KV cache into RAM or onto the GPU.
        payload = torch.load(self.path(example), map_location="cpu", weights_only=True, mmap=True)
        if payload["example"] != example:
            raise ValueError("Prefix snapshot belongs to a different episode")
        return payload

    def load_memory(self, example: dict, device: torch.device | str) -> NeuralMemoryStates:
        return _unpack_memory(self._load_payload(example)["memory_states"], device)

    def load(self, example: dict, device: torch.device | str) -> SavedPrefix:
        payload = self._load_payload(example)
        saved_cache = payload["cache"]
        cache = SlidingWindowKVCache(
            num_layers=len(saved_cache["layers"]),
            window_size=saved_cache["window_size"],
            max_persistent_kv_tokens=saved_cache["max_persistent_kv_tokens"],
        )
        if saved_cache["persistent_positions"] is not None:
            cache.persistent_positions = saved_cache["persistent_positions"].to(device).clone()
        for layer, saved_layer in zip(cache.layers, saved_cache["layers"]):
            keys = saved_layer["keys"].to(device).clone()
            layer.lazy_initialization(keys)
            layer.keys = keys
            layer.values = saved_layer["values"].to(device).clone()
            layer.positions = saved_layer["positions"].to(device).clone()
            layer.is_persistent = saved_layer["is_persistent"].to(device).clone()
            layer.cumulative_length = saved_layer["cumulative_length"]
        trace = payload["trace"]
        return SavedPrefix(
            state=TaalModelState(
                past_key_values=cache,
                tokens_seen=payload["tokens_seen"],
                memory_states=_unpack_memory(payload["memory_states"], device),
            ),
            trace=None if trace is None else PrefixTrace(
                writes=[WriteEvent(**event) for event in trace["writes"]],
                reads=[ReadEvent(**event) for event in trace["reads"]],
                internal_prefixes=[InternalPrefix(**event) for event in trace["internal_prefixes"]],
            ),
        )
