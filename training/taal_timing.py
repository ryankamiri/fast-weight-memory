"""Opt-in, one-microbatch TaaL timing without per-token synchronization."""

import json
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch


class TaalStageTimings:
    """Aggregate every invocation, including each token's write and read.

    These are unsynchronized host call spans. They include Python, dispatch,
    and any implicit waits, but are not active-GPU-compute measurements.
    Write prediction, output-loss derivative formation (write_loss), and the
    explicit reverse chain rule (gradient_calculation) are timed separately.
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
        self.marks: dict[int, list[dict]] = {}
        self.backward_marks: dict[int, dict] = {}
        self.handles = []
        self.stage_timers: dict[int, TaalStageTimings] = {}
        self.started_at = 0.0

    def _mark(self, layer: int, name: str) -> None:
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        calls = self.marks.setdefault(layer, [])
        if name == "layer_start":
            calls.append({})
        calls[-1][name] = (time.perf_counter(), event)

    def _backward_mark(self, layer, name, gradient):
        event = None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        self.backward_marks.setdefault(layer, {})[name] = (time.perf_counter(), event)
        return gradient

    def _watch_memory_backward(self, index, inputs, output):
        # Tensor hooks survive removal of the forward instrumentation. Their span
        # brackets reverse computation from memory reads to memory input gradients;
        # it is an elapsed interval, not an exclusive operator attribution.
        if inputs[0].requires_grad and output[0].requires_grad:
            output[0].register_hook(lambda gradient: self._backward_mark(index, "start", gradient))
            inputs[0].register_hook(lambda gradient: self._backward_mark(index, "end", gradient))

    def __enter__(self):
        self.started_at = time.perf_counter()
        for index, layer in enumerate(self.model.model.layers):
            observer = TaalStageTimings(index)
            self.stage_timers[index] = observer
            layer.taal.timing_observer = observer
            if hasattr(layer.taal, "neural_memory"):
                layer.taal.neural_memory.timing_observer = observer
                layer.taal.neural_memory.memory_mlp.timing_observer = observer
                self.handles.append(layer.taal.neural_memory.register_forward_hook(
                    lambda _module, inputs, output, index=index: self._watch_memory_backward(index, inputs, output)
                ))
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
                layer.taal.neural_memory.memory_mlp.timing_observer = None
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
            memory_cpu = decoder_cpu = 0.0
            memory_gpu = decoder_gpu = None
            for marks in self.marks[index]:
                if len(marks) != 4:
                    raise RuntimeError(f"TaaL timing is incomplete for layer {index}")
                cpu, gpu = self._span(marks, "memory_start", "memory_end")
                memory_cpu += cpu
                if gpu is not None:
                    memory_gpu = (memory_gpu or 0) + gpu
                cpu, gpu = self._span(marks, "memory_end", "layer_end")
                decoder_cpu += cpu
                if gpu is not None:
                    decoder_gpu = (decoder_gpu or 0) + gpu
            layers.append({
                "layer": index,
                "forward_calls": len(self.marks[index]),
                "memory_cpu_dispatch_s": round(memory_cpu, 4),
                "decoder_cpu_dispatch_s": round(decoder_cpu, 4),
                "memory_gpu_timeline_ms": round(memory_gpu, 2) if memory_gpu is not None else None,
                "decoder_gpu_timeline_ms": round(decoder_gpu, 2) if decoder_gpu is not None else None,
                "stage_host_spans": self.stage_timers[index].summary(),
            })
        # Observed calls remain correct for multi-block prefill and query resumes
        # that do not prepend internal tokens, unlike a sequence-length estimate.
        memory_calls = sum(
            timer.samples.get("gradient_execution", [0])[0]
            for timer in self.stage_timers.values()
        )
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
            "expected_memory_update_and_read_calls": memory_calls or expected_layers * (self.sequence_length + self.model.config.num_persistent_tokens),
            "memory_dim": getattr(self.model.config, "memory_dim", None),
            "memory_depth": getattr(self.model.config, "memory_depth", None),
            "memory_chunk_size": getattr(self.model.config, "memory_chunk_size", None),
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

    def backward_record(self) -> dict:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        layers = []
        for index, marks in sorted(self.backward_marks.items()):
            if set(marks) != {"start", "end"}:
                raise RuntimeError(f"Memory backward timing is incomplete for layer {index}")
            cpu, gpu = self._span(marks, "start", "end")
            layers.append({"layer": index, "wall_s": round(cpu, 4), "gpu_timeline_ms": gpu})
        return {
            "per_layer_neural_memory_backward": layers,
            "backward_timing_semantics": "elapsed intervals between memory-output and memory-input gradient hooks; may overlap other autograd work",
        }


class TaalKernelSampler:
    """One bounded CPU/CUDA operator trace, not the enormous full training graph."""

    def __init__(self, model, device, output_path, *, warmup_calls=16, sample_calls=64):
        self.memory_mlp = model.model.layers[0].taal.neural_memory.memory_mlp
        self.device = device
        self.output_path = Path(output_path)
        self.warmup_calls = warmup_calls
        self.sample_calls = sample_calls
        self.calls = 0
        self.profiler = None
        self.finished = False
        self.handles = []

    def _start(self, _module, _args):
        self.calls += 1
        if self.calls == self.warmup_calls + 1:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if self.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(activities=activities)
            self.profiler.__enter__()

    def _stop(self, _module, _args, _output):
        if self.profiler is not None and self.calls == self.warmup_calls + self.sample_calls:
            self._finish()

    def _finish(self):
        self.profiler.__exit__(None, None, None)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.profiler.export_chrome_trace(str(self.output_path))
        averages = self.profiler.key_averages()
        print_taal_timing({
            "kind": "taal_bounded_kernel_sample", "layer": 0,
            "mlp_calls": min(self.sample_calls, self.calls - self.warmup_calls),
            "trace_path": str(self.output_path),
            "self_cpu_total_us": sum(event.self_cpu_time_total for event in averages),
            "self_device_total_us": sum(event.self_device_time_total for event in averages),
            "semantics": "bounded forward sample includes intervening writes/reads; profiler overhead and warmup excluded from performance conclusions; inspect timeline for launch gaps",
        })
        print(averages.table(sort_by="self_cpu_time_total", row_limit=15), flush=True)
        if self.device.type == "cuda":
            print(averages.table(sort_by="self_device_time_total", row_limit=15), flush=True)
        self.finished = True
        self.profiler = None

    def __enter__(self):
        self.handles = [
            self.memory_mlp.register_forward_pre_hook(self._start),
            self.memory_mlp.register_forward_hook(self._stop),
        ]
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        if self.profiler is not None:
            if exc_type is None:
                self._finish()
            else:
                self.profiler.__exit__(exc_type, exc_value, traceback)
                self.profiler = None
        return False


class TaalEvaluationTimings:
    """Opt-in phase boundaries separate prefix computation from tracing/IO."""

    def __init__(self, model, device, enabled):
        self.model, self.device, self.enabled = model, device, enabled
        self.samples = defaultdict(lambda: [0, 0.0])

    @contextmanager
    def phase(self, name, *, tokens=0, detailed=False):
        if not self.enabled:
            yield
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        timer = TaalMicrobatchTimer(self.model, self.device, tokens) if detailed else None
        with timer if timer is not None else nullcontext():
            yield
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        if timer is not None:
            record = timer.forward_record()
            record.update(kind="taal_eval_forward", phase=name)
            print_taal_timing(record)
        sample = self.samples[name]
        sample[0] += 1
        sample[1] += elapsed
        print_taal_timing({"kind": "taal_eval_phase", "phase": name,
                           "call": sample[0], "tokens": tokens, "wall_s": round(elapsed, 4)})


def print_taal_timing(record: dict) -> None:
    print("TAAL_TIMING " + json.dumps(record, separators=(",", ":")), flush=True)
