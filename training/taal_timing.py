"""Opt-in, one-microbatch TaaL timing without per-token synchronization."""

import json
import time
from collections import defaultdict

import torch


class TaalStageTimings:
    """Aggregate every invocation, including each token's write and read.

    These are unsynchronized host call spans. They include Python, dispatch,
    and any implicit waits, but are not active-GPU-compute measurements.
    """

    def __init__(self, layer: int):
        self.layer = layer
        self.samples = defaultdict(lambda: [0, 0.0, 0.0])

    def __call__(self, stage: str, seconds: float) -> None:
        sample = self.samples[stage]
        sample[0] += 1
        sample[1] += seconds
        sample[2] = max(sample[2], seconds)

    def summary(self) -> dict:
        return {
            stage: {
                "calls": count,
                "total_s": round(total, 4),
                "mean_ms": round(1000 * total / count, 4),
                "max_ms": round(1000 * maximum, 4),
            }
            for stage, (count, total, maximum) in sorted(self.samples.items())
        }


class TaalMicrobatchTimer:
    """Measure layer/memory spans and emit a parseable Slurm-log record.

    CUDA events measure elapsed GPU-timeline time, including gaps while Python
    dispatches tiny operations; they are not a measure of active GPU compute.
    CPU dispatch spans are also recorded. One synchronization after the forward
    makes the overall wall time and event readings complete.
    """

    def __init__(self, model, device: torch.device, sequence_length: int):
        if model.model.gradient_checkpointing:
            raise ValueError("TaaL timing currently expects non-checkpointed decoder calls")
        self.model = model
        self.device = device
        self.sequence_length = sequence_length
        self.marks: dict[int, dict[str, tuple[float, torch.cuda.Event | None]]] = {}
        self.handles = []
        self.stage_timers: dict[int, TaalStageTimings] = {}
        self.started_at = 0.0

    def _mark(self, layer: int, name: str) -> None:
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.marks.setdefault(layer, {})[name] = (time.perf_counter(), event)

    def __enter__(self):
        self.started_at = time.perf_counter()
        for index, layer in enumerate(self.model.model.layers):
            observer = TaalStageTimings(index)
            self.stage_timers[index] = observer
            layer.taal.timing_observer = observer
            if hasattr(layer.taal, "neural_memory"):
                layer.taal.neural_memory.timing_observer = observer
            self.handles.append(layer.register_forward_pre_hook(
                lambda _module, _args, index=index: self._mark(index, "layer_start")
            ))
            self.handles.append(layer.taal.register_forward_pre_hook(
                lambda _module, _args, index=index: self._mark(index, "memory_start")
            ))
            self.handles.append(layer.taal.register_forward_hook(
                lambda _module, _args, _output, index=index: self._mark(index, "memory_end")
            ))
            self.handles.append(layer.register_forward_hook(
                lambda _module, _args, _output, index=index: self._mark(index, "layer_end")
            ))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for layer in self.model.model.layers:
            layer.taal.timing_observer = None
            if hasattr(layer.taal, "neural_memory"):
                layer.taal.neural_memory.timing_observer = None
        return False

    @staticmethod
    def _span(marks, start: str, end: str) -> tuple[float, float | None]:
        start_cpu, start_event = marks[start]
        end_cpu, end_event = marks[end]
        gpu_ms = (
            start_event.elapsed_time(end_event)
            if start_event is not None and end_event is not None else None
        )
        return end_cpu - start_cpu, gpu_ms

    def forward_record(self) -> dict:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        layers = []
        expected_layers = len(self.model.model.layers)
        if len(self.marks) != expected_layers:
            raise RuntimeError("TaaL timing missed one or more decoder layers")
        for index in range(expected_layers):
            marks = self.marks[index]
            if len(marks) != 4:
                raise RuntimeError(f"TaaL timing is incomplete for layer {index}")
            memory_cpu, memory_gpu = self._span(marks, "memory_start", "memory_end")
            decoder_cpu, decoder_gpu = self._span(marks, "memory_end", "layer_end")
            layers.append({
                "layer": index,
                "memory_cpu_dispatch_s": round(memory_cpu, 4),
                "decoder_cpu_dispatch_s": round(decoder_cpu, 4),
                "memory_gpu_timeline_ms": round(memory_gpu, 2) if memory_gpu is not None else None,
                "decoder_gpu_timeline_ms": round(decoder_gpu, 2) if decoder_gpu is not None else None,
                "stage_host_spans": self.stage_timers[index].summary(),
            })
        persistent = self.model.config.num_persistent_tokens
        aggregate: dict[str, list[float]] = {}
        for timer in self.stage_timers.values():
            for stage, (count, total, maximum) in timer.samples.items():
                combined = aggregate.setdefault(stage, [0, 0.0, 0.0])
                combined[0] += count
                combined[1] += total
                combined[2] = max(combined[2], maximum)
        return {
            "kind": "taal_first_microbatch_forward",
            "sequence_tokens": self.sequence_length,
            "layers": expected_layers,
            "expected_memory_update_and_read_calls": expected_layers * (self.sequence_length + persistent),
            "forward_wall_s": round(time.perf_counter() - self.started_at, 4),
            "memory_cpu_dispatch_s": round(sum(row["memory_cpu_dispatch_s"] for row in layers), 4),
            "decoder_cpu_dispatch_s": round(sum(row["decoder_cpu_dispatch_s"] for row in layers), 4),
            "memory_gpu_timeline_ms": round(sum(row["memory_gpu_timeline_ms"] or 0 for row in layers), 2),
            "decoder_gpu_timeline_ms": round(sum(row["decoder_gpu_timeline_ms"] or 0 for row in layers), 2),
            "stage_timing_semantics": "unsynchronized host call spans; nested *_total stages overlap children; not active GPU utilization",
            "stage_host_spans_all_layers": {
                stage: {
                    "calls": int(count),
                    "total_s": round(total, 4),
                    "mean_ms": round(1000 * total / count, 4),
                    "max_ms": round(1000 * maximum, 4),
                }
                for stage, (count, total, maximum) in sorted(aggregate.items())
            },
            "per_layer": layers,
        }


def print_taal_timing(record: dict) -> None:
    print("TAAL_TIMING " + json.dumps(record, separators=(",", ":")), flush=True)
