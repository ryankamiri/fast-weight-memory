from contextlib import AbstractContextManager
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.titans.state import NeuralMemoryState
from evaluation.taal.trace_contract import (
    InternalPrefix,
    ReadEvent,
    SCHEMA_VERSION,
    TraceComparison,
    TraceEpisode,
    TraceToken,
    WriteEvent,
)


def _parameter_norm(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.float().square().sum() for value in parameters.values()]).sum().sqrt()


def _difference_norm(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([
        (left[name].float() - right[name].float()).square().sum()
        for name in left
    ]).sum().sqrt()


class _LayerRecorder:
    def __init__(self, layer: int, offset: int, chunk_size: int):
        self.layer = layer
        self.offset = offset
        self.chunk_size = chunk_size
        self.cursor = 0
        self.segment_index = -1
        self.writes: list[WriteEvent] = []
        self.reads: list[ReadEvent] = []
        self.prefixes: list[InternalPrefix] = []
        self._active = False
        self._prefix_start: dict[str, torch.Tensor] | None = None
        self._pending_writes: list[torch.Tensor] = []
        self._pending_boundaries: list[bool] = []
        self._pending_prefix_change: torch.Tensor | None = None

    def begin_call(self, *, text_length: int, persistent_count: int) -> None:
        if text_length < 1 or persistent_count < 0:
            raise ValueError("Invalid TaaL trace call lengths")
        if self._active:
            raise RuntimeError("A previous TaaL trace call was not completed")
        if persistent_count:
            self.segment_index += 1
        elif self.segment_index < 0:
            self.segment_index = 0
        self._active = True
        self._text_length = text_length
        self._persistent_count = persistent_count
        self._write_index = 0
        self._pending_writes = []
        self._pending_boundaries = []
        self._pending_prefix_change = None

    def record_write(
        self,
        before: NeuralMemoryState,
        token_gradients: dict[str, torch.Tensor],
        after: NeuralMemoryState,
        write_mask: torch.Tensor,
        write_strength: torch.Tensor,
    ) -> None:
        if not self._active:
            raise RuntimeError("TaaL write arrived outside a trace call")
        B, C = write_mask.shape
        if B != 1:
            raise ValueError("TaaL trace export currently requires one session per batch")
        if self._write_index + C > self._persistent_count + self._text_length:
            raise RuntimeError("More writes than memory input tokens")
        old = {name: value[0].detach() for name, value in before.weights.items()}
        new = {name: value[0].detach() for name, value in after.weights.items()}
        committed = after.pending_count == 0
        for token_index in range(C):
            index = self._write_index
            internal = index < self._persistent_count
            boundary = committed and token_index == C - 1
            grad = {
                name: value[0, token_index].detach()
                for name, value in token_gradients.items()
            }
            if internal and index == 0:
                self._prefix_start = {name: value.clone() for name, value in old.items()}
            with torch.no_grad():
                proposed = _parameter_norm(grad)
                previous = _parameter_norm(old)
                net = _difference_norm(new, old) if boundary else proposed.new_zeros(())
                if self.chunk_size == 1:
                    other = torch.stack([
                        (new[name].float() - old[name].float() + grad[name].float())
                        .square().sum()
                        for name in old
                    ]).sum().sqrt()
                    dot = torch.stack([
                        (-grad[name].float() * (new[name].float() - old[name].float())).sum()
                        for name in old
                    ]).sum()
                else:
                    # For a multi-token update, comparing one token's proposed
                    # gradient to the entire chunk's net movement is invalid.
                    other = proposed.new_full((), float("nan"))
                    dot = proposed.new_full((), float("nan"))
                # Transfer scalars together after the forward call, not once
                # per token, to avoid forcing repeated GPU synchronization.
                self._pending_writes.append(torch.stack((
                    proposed,
                    previous,
                    net,
                    other,
                    dot,
                    write_strength[0, token_index].detach().float(),
                    write_mask[0, token_index].detach().float(),
                )))
            self._pending_boundaries.append(boundary)
            if internal and index + 1 == self._persistent_count:
                assert self._prefix_start is not None
                end_weights = new if boundary else old
                self._pending_prefix_change = _difference_norm(
                    end_weights, self._prefix_start
                )
                self._prefix_start = None
            self._write_index += 1

    def record_reads(
        self,
        incoming: torch.Tensor,
        injection: torch.Tensor,
        read_scale: float,
        residual_gate: torch.Tensor,
    ) -> None:
        if not self._active or self._write_index != self._persistent_count + self._text_length:
            raise RuntimeError("TaaL read/write trace lengths disagree")
        with torch.no_grad():
            write_metrics = torch.stack(self._pending_writes).cpu().tolist()
            incoming_norms = incoming[0].detach().float().norm(dim=-1)
            injected_norms = injection[0].detach().float().norm(dim=-1)
            magnitudes = torch.stack((incoming_norms, injected_norms), dim=-1).cpu().tolist()
        for index, metrics in enumerate(write_metrics):
            proposed, previous, net, other, dot, strength, mask = metrics
            internal = index < self._persistent_count
            alignment = (
                dot / (proposed * net)
                if self.chunk_size == 1 and proposed > 0 and net > 0
                else None
            )
            self.writes.append(WriteEvent(
                layer=self.layer,
                position=(
                    None if internal
                    else self.offset + self.cursor + index - self._persistent_count
                ),
                segment_index=self.segment_index,
                internal_index=index if internal else None,
                write_enabled=bool(mask),
                write_strength=strength,
                proposed_write_norm=proposed,
                previous_weight_norm=previous,
                net_weight_change_norm=net,
                other_movement_norm=other if self.chunk_size == 1 else None,
                write_to_net_alignment=alignment,
                chunk_size=self.chunk_size,
                chunk_boundary=self._pending_boundaries[index],
            ))
        if self._persistent_count:
            assert self._pending_prefix_change is not None
            self.prefixes.append(InternalPrefix(
                layer=self.layer,
                segment_index=self.segment_index,
                before_position=self.offset + self.cursor,
                count=self._persistent_count,
                net_weight_change_norm=float(self._pending_prefix_change.cpu().item()),
            ))
        gate = float(residual_gate.detach().float().cpu().item())
        for index, (incoming_norm, injected_norm) in enumerate(magnitudes):
            self.reads.append(ReadEvent(
                layer=self.layer,
                position=self.offset + self.cursor + index,
                segment_index=self.segment_index,
                enabled=read_scale != 0,
                read_scale=float(read_scale),
                residual_gate=gate,
                incoming_norm=incoming_norm,
                injected_norm=injected_norm,
                relative_injection=(
                    injected_norm / incoming_norm if incoming_norm > 0 else None
                ),
                state_timing=(
                    "post-write"
                    if self._pending_boundaries[self._persistent_count + index]
                    else "pre-write"
                ),
            ))
        self.cursor += self._text_length
        self._pending_writes = []
        self._pending_boundaries = []
        self._pending_prefix_change = None
        self._active = False


class TaalTraceRecorder(AbstractContextManager):
    """Attach only during one eval-mode, batch-one prompt/continuation."""

    def __init__(
        self,
        model: TaalQwen3ForCausalLM,
        layers: list[int],
        token_count: int,
        position_offset: int = 0,
    ):
        if model.training:
            raise ValueError("TaaL trace export requires model.eval()")
        if not layers or len(layers) != len(set(layers)):
            raise ValueError("Select one or more distinct TaaL layers")
        if token_count < 1 or position_offset < 0:
            raise ValueError("Trace token count must be positive and offset nonnegative")
        decoder_layers = model.model.layers
        if any(layer < 0 or layer >= len(decoder_layers) for layer in layers):
            raise ValueError("Trace layer index is out of range")
        self.model = model
        self.token_count = token_count
        self.recorders = {
            layer: _LayerRecorder(
                layer,
                position_offset,
                decoder_layers[layer].taal.neural_memory.config.chunk_size,
            )
            for layer in sorted(layers)
        }
        self._attached = False

    def __enter__(self):
        for layer in self.recorders:
            taal = self.model.model.layers[layer].taal
            if taal.trace_observer is not None or taal.neural_memory.trace_observer is not None:
                raise RuntimeError("A TaaL trace observer is already attached")
        for layer, recorder in self.recorders.items():
            taal = self.model.model.layers[layer].taal
            taal.trace_observer = recorder
            taal.neural_memory.trace_observer = recorder.record_write
        self._attached = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._attached:
            for layer in self.recorders:
                taal = self.model.model.layers[layer].taal
                taal.trace_observer = None
                taal.neural_memory.trace_observer = None
            self._attached = False
        if exc_type is None:
            for recorder in self.recorders.values():
                if recorder._active or recorder.cursor != self.token_count:
                    raise RuntimeError("TaaL trace did not cover the declared token count")
        return False

    @property
    def writes(self) -> list[WriteEvent]:
        return [event for recorder in self.recorders.values() for event in recorder.writes]

    @property
    def reads(self) -> list[ReadEvent]:
        return [event for recorder in self.recorders.values() for event in recorder.reads]

    @property
    def internal_prefixes(self) -> list[InternalPrefix]:
        return [prefix for recorder in self.recorders.values() for prefix in recorder.prefixes]


def make_trace_tokens(token_ids: list[int], tokenizer, *, offset: int = 0, phase: str = "prompt") -> list[TraceToken]:
    if phase not in ("prompt", "generated"):
        raise ValueError("Token phase must be prompt or generated")
    return [
        TraceToken(
            position=offset + index,
            token_id=int(token_id),
            text=tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            phase=phase,
        )
        for index, token_id in enumerate(token_ids)
    ]


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(payload, file, ensure_ascii=False, allow_nan=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


class TaalTraceExporter:
    def __init__(self, root: Path, *, run_metadata: dict):
        self.root = root
        self.manifest_path = root / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest["schema"] != SCHEMA_VERSION or self.manifest["run_metadata"] != run_metadata:
                raise ValueError("Trace manifest differs from this run; use a new trace directory")
            self.manifest.setdefault("comparisons", {})
        else:
            self.manifest = {
                "schema": SCHEMA_VERSION,
                "run_metadata": run_metadata,
                "episodes": {},
                "comparisons": {},
            }
            _atomic_json(self.manifest_path, self.manifest)

    def export(self, episode: TraceEpisode) -> Path:
        payload = episode.to_dict()
        key = f"{episode.example_id}/{episode.condition_id}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        relative = Path("episodes") / f"{digest}.json.gz"
        path = self.root / relative
        existing = self.manifest["episodes"].get(key)
        if existing is not None and existing != str(relative):
            raise ValueError(f"Trace index disagrees for {key}")
        self._write_gzip(path, payload)
        self.manifest["episodes"][key] = str(relative)
        _atomic_json(self.manifest_path, self.manifest)
        return path

    def export_comparison(self, comparison: TraceComparison) -> Path:
        if comparison.schema != SCHEMA_VERSION:
            raise ValueError(f"Unsupported comparison schema: {comparison.schema}")
        if comparison.difference_log_probability != (
            comparison.baseline_log_probability - comparison.variant_log_probability
        ):
            raise ValueError("Comparison log-probability difference is inconsistent")
        key = (
            f"{comparison.example_id}/{comparison.baseline_condition}/"
            f"{comparison.variant_condition}"
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        relative = Path("comparisons") / f"{digest}.json.gz"
        existing = self.manifest["comparisons"].get(key)
        if existing is not None and existing != str(relative):
            raise ValueError(f"Comparison index disagrees for {key}")
        path = self.root / relative
        self._write_gzip(path, comparison.__dict__)
        self.manifest["comparisons"][key] = str(relative)
        _atomic_json(self.manifest_path, self.manifest)
        return path

    @staticmethod
    def _write_gzip(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary_file:
            temporary = Path(temporary_file.name)
        try:
            with temporary.open("wb") as raw_file:
                with gzip.open(raw_file, mode="wt", encoding="utf-8") as file:
                    json.dump(
                        payload,
                        file,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    file.write("\n")
                raw_file.flush()
                os.fsync(raw_file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
