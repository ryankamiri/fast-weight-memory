"""Saved-key reconstruction and controlled answer-token read-address audits.

Evaluation-only hooks leave the architecture/checkpoint unchanged. Oracle reads
replay native answer-token write inputs at every layer, preventing lower-layer
read changes from confounding upper-layer memory updates.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer

from architectures.taal.qwen.state import NeuralMemoryStates
from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.lifecycle import lifecycle_boundaries
from evaluation.taal.prefix_store import PrefixStore, PrefixTrace, checkpoint_fingerprint
from evaluation.scoring import grouped_score_summary, score_logits
from evaluation.taal.state_audit import clone_memory_states, fork_session_state, load_model
from evaluation.taal.state_readout_probe import save_state, save_tensor_payload
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder, make_trace_tokens, trace_greedy_prediction
from utils.seed import seed_everything


CONDITIONS = ("native", "oracle_relevant", "oracle_wrong", "answer_reads_off")
STAGES = ("fact", "middle", "final")


@dataclass
class NativeWrite:
    before: NeuralMemoryStates
    after: NeuralMemoryStates
    keys: torch.Tensor
    values: torch.Tensor
    inputs: torch.Tensor
    mask: torch.Tensor
    query: torch.Tensor


class AddressCapture:
    """Capture actual post-convolution/SiLU/normalization inputs, batch/chunk one.

    In oracle mode, the final read address changes but the update receives the
    exact native before-state, keys, values, controls input and mask. The returned
    W/S must equal the native post-write state exactly at every selected layer.
    """

    def __init__(self, model, *, start: int, persistent: bool = False, capture_writes: bool = False,
                 reference: dict[int, NativeWrite] | None = None,
                 addresses: dict[int, torch.Tensor] | None = None):
        if model.training or model.config.memory_chunk_size != 1:
            raise ValueError("Address diagnostics require eval and memory_chunk_size=1")
        if (reference is None) != (addresses is None):
            raise ValueError("An oracle requires both native writes and read addresses")
        self.model = model
        self.layers = sorted(model.model.initial_memory_states(batch_size=1))
        count = model.config.num_persistent_tokens if persistent else 0
        self.positions = {layer: start - count for layer in self.layers}
        self.events = {layer: [] for layer in self.layers}
        self.writes = {}
        self.reference = reference
        self.addresses = addresses
        self.capture_writes = capture_writes
        self.originals = {}
        if reference is not None and (set(reference) != set(self.layers) or set(addresses) != set(self.layers)):
            raise ValueError("Oracle must cover exactly the native memory layers")

    def _wrap(self, layer, original):
        def process(memory, state, queries, keys, values, inputs, write_mask, output_dtype):
            B, C, D = queries.shape
            if (B, C) != (1, 1):
                raise ValueError("Address capture requires B=C=1")
            native_query = queries
            if self.reference is not None:
                if self.events[layer]:
                    raise ValueError("Oracle scope is one answer-predicting token")
                native = self.reference[layer]
                state = clone_memory_states(native.before)[layer]
                keys, values, inputs, write_mask = native.keys, native.values, native.inputs, native.mask
                queries = self.addresses[layer].to(queries.device).reshape(1, 1, D)
            before = None
            if self.capture_writes:
                before = clone_memory_states({layer: state})
            output, after = original(state, queries, keys, values, inputs, write_mask, output_dtype)
            if self.reference is not None:
                for name, expected in self.reference[layer].after[layer].weights.items():
                    torch.testing.assert_close(after.weights[name], expected, rtol=0, atol=0)
                    torch.testing.assert_close(after.momentum[name], self.reference[layer].after[layer].momentum[name], rtol=0, atol=0)
            if self.capture_writes:
                self.writes[layer] = NativeWrite(before, clone_memory_states({layer: after}),
                    keys.detach().clone(), values.detach().clone(), inputs.detach().clone(),
                    write_mask.detach().clone(), native_query.detach().clone())
            position = self.positions[layer]
            self.events[layer].append({"position": position if position >= 0 else None,
                "internal_index": position + self.model.config.num_persistent_tokens if position < 0 else None,
                "query": queries[0, 0].detach().float().clone(),
                "formed_query": native_query[0, 0].detach().float().clone(),
                "key": keys[0, 0].detach().float().clone(),
                "value": values[0, 0].detach().float().clone(),
                "read_output": output[0, 0].detach().float().clone(),
                "write_enabled": write_mask[0, 0].detach().clone(),
                "native_write_replayed": self.reference is not None})
            self.positions[layer] += 1
            return output, after
        return process

    def __enter__(self):
        for layer in self.layers:
            memory = self.model.model.layers[layer].taal.neural_memory
            self.originals[layer] = memory._process_chunk
            memory._process_chunk = MethodType(self._wrap(layer, memory._process_chunk), memory)
        return self

    def __exit__(self, *_):
        for layer, original in self.originals.items():
            self.model.model.layers[layer].taal.neural_memory._process_chunk = original

    def payload(self):
        result = {}
        for layer, events in self.events.items():
            vectors = torch.stack([torch.stack([event[name] for name in
                ("query", "formed_query", "key", "value", "read_output")]) for event in events]).cpu()
            if not bool(torch.isfinite(vectors).all()):
                raise ValueError("Non-finite address vector")
            masks = torch.stack([event["write_enabled"] for event in events]).cpu().tolist()
            result[str(layer)] = [{**{name: value for name, value in event.items()
                if not isinstance(value, torch.Tensor)}, **dict(zip(
                    ("query", "formed_query", "key", "value", "read_output"), numbers)),
                    "write_enabled": bool(mask)}
                for event, numbers, mask in zip(events, vectors.tolist(), masks)]
        return {"schema": "taal-address-vectors/v1", "layers": result,
                "vector_stage": "post convolution, SiLU and Q/K normalization"}


def select_heldout(rows, split):
    selected = [row for row in rows if row["probe_split"] == split]
    if not selected or len({row["example_id"] for row in selected}) != len(selected):
        raise ValueError("Need unique selected held-out examples")
    donors = {}
    for row in selected:
        matches = [other for other in selected if other["pair_group"] == row["pair_group"]
                   and other["target_token_id"] != row["target_token_id"]]
        if len(matches) != 1:
            raise ValueError("Need exactly one opposing-label donor in each selected pair")
        donors[row["example_id"]] = matches[0]
    return selected, donors


def reconstruction(output, value):
    output, value = output.float().reshape(-1), value.float().reshape(-1)
    metrics = {"mse": float((output - value).square().mean()),
               "relative_error": float((output - value).norm() / value.norm().clamp_min(1e-12)),
               "cosine": float(F.cosine_similarity(output, value, dim=0)),
               "output_norm": float(output.norm()), "value_norm": float(value.norm())}
    if not all(torch.isfinite(torch.tensor(number)).item() for number in metrics.values()):
        raise ValueError("Non-finite reconstruction metric")
    return metrics


def vector_path(example_id, suffix):
    digest = hashlib.sha256(example_id.encode()).hexdigest()[:24]
    return Path("vectors") / f"{digest}-{suffix}.json.gz"


@torch.inference_mode()
def collect_prefix(model, tokenizer, example, root, exporter, store, settings):
    boundaries = lifecycle_boundaries(example, tokenizer)
    ids, driver = example["input_ids"], len(example["input_ids"]) - 1
    capture = AddressCapture(model, start=0, persistent=True)
    recorder = TaalTraceRecorder(model, capture.layers, driver)
    state, start = None, 0
    with recorder, capture:
        for stage, end in [(stage, boundaries[stage]) for stage in STAGES] + [("driver", driver)]:
            if end <= start:
                raise ValueError("Snapshot stages must precede the answer-predicting token")
            state = model.prefill(torch.tensor([ids[start:end]], device=model.device), state=state,
                execution_block_size=settings["execution_block_size"], prepend_memory_tokens=start == 0).state
            if settings["mode"] == "reconstruction" and stage in STAGES:
                save_state(root / "snapshots" / example["example_id"] / f"{stage}.pt", state.memory_states)
            if stage == "final" and bool((state.past_key_values.layers[0].positions < boundaries["fact"]).any()):
                raise ValueError("Original fact remains in query KV")
            start = end
    fact_addresses = {}
    for layer, events in capture.events.items():
        fact = next(event for event in events if event["position"] == example["fact_position"])
        fact_addresses[layer] = {name: fact[name].cpu() for name in ("key", "value")}
    save_tensor_payload(root / "fact_addresses" / f"{example['example_id']}.pt", fact_addresses)
    relative = vector_path(example["example_id"], "prefix")
    exporter._write_gzip(exporter.root / relative, capture.payload())
    store.save(example, state, PrefixTrace(recorder.writes, recorder.reads, recorder.internal_prefixes))


@torch.inference_mode()
def audit_example(model, tokenizer, example, donor, root, exporter, store, settings):
    saved = store.load(example, model.device)
    own = torch.load(root / "fact_addresses" / f"{example['example_id']}.pt", weights_only=True)
    wrong = torch.load(root / "fact_addresses" / f"{donor['example_id']}.pt", weights_only=True)
    driver, ids = len(example["input_ids"]) - 1, example["input_ids"]
    layers = sorted(own)
    native_writes = None
    conditions = ("native",) if settings["mode"] == "reconstruction" else CONDITIONS
    for condition in conditions:
        addresses = None
        if condition.startswith("oracle_"):
            source = own if condition == "oracle_relevant" else wrong
            addresses = {layer: values["key"] for layer, values in source.items()}
        elif condition == "answer_reads_off":
            addresses = {layer: write.query for layer, write in native_writes.items()}
        capture = AddressCapture(model, start=driver, capture_writes=True,
            reference=native_writes if addresses is not None else None, addresses=addresses)
        recorder = TaalTraceRecorder(model, layers, 1, position_offset=driver)
        scale = 0.0 if condition == "answer_reads_off" else 1.0
        state = fork_session_state(saved.state, clone_memory_states(saved.state.memory_states))
        with recorder, capture:
            output = model.prefill(torch.tensor([[ids[-1]]], device=model.device), state=state,
                memory_read_scale=scale, execution_block_size=1, prepend_memory_tokens=False)
        if condition == "native":
            native_writes = capture.writes
        outcome = {"evaluation_id": f"{example['example_id']}/{condition}",
            "example_id": example["example_id"], "audit_condition": condition,
            "pair_group": example["pair_group"], "probe_split": example["probe_split"],
            "donor_example_id": donor["example_id"], **score_logits(output.logits[0, -1], example, tokenizer)}
        outcome["address_geometry"] = {
            str(layer): {
                "native_query_relevant_key_cosine": float(F.cosine_similarity(
                    native_writes[layer].query.float().reshape(-1).cpu(), own[layer]["key"].reshape(-1), dim=0)),
                "native_query_wrong_key_cosine": float(F.cosine_similarity(
                    native_writes[layer].query.float().reshape(-1).cpu(), wrong[layer]["key"].reshape(-1), dim=0)),
            }
            for layer in layers
        }
        feedback = AddressCapture(model, start=len(ids))
        with feedback:
            prediction = trace_greedy_prediction(model, tokenizer, token_id=outcome["vocabulary_top_token_id"],
                position=len(ids), state=output.state, layers=layers, memory_read_scale=1.0)
        relative = vector_path(example["example_id"], condition)
        exporter._write_gzip(exporter.root / relative, {"answer_driver": capture.payload(),
            "generated_feedback": feedback.payload(), "scored_position": driver})
        exporter.export(TraceEpisode(root.name, example["example_id"], condition,
            exporter.manifest["run_metadata"]["checkpoint"], exporter.manifest["run_metadata"]["tokenizer"],
            make_trace_tokens(ids, tokenizer) + [prediction.token],
            saved.trace.writes + recorder.writes + prediction.writes,
            saved.trace.reads + recorder.reads + prediction.reads,
            saved.trace.internal_prefixes + recorder.internal_prefixes + prediction.internal_prefixes,
            outcome, {"expected_answer": example["answer"], "expected_token_id": example["target_token_id"],
                "query_start_position": example["final_query_position"], "intervention_position": driver,
                "prefix_vectors": str(vector_path(example["example_id"], "prefix")),
                "condition_vectors": str(relative), "native_write_inputs_fixed": True,
                "read_scope": "answer-predicting token only; generated feedback is native",
                "starting_kv": "same recipient; downstream activations/KV can change causally",
                "oracle": "historical saved key, not a deployable reader"}))
        existing = read_results(root / "results.jsonl", id_field="evaluation_id")
        if outcome["evaluation_id"] not in existing:
            append_result(root / "results.jsonl", outcome)


@torch.inference_mode()
def mapping_rows(model, example, donor, root):
    own = torch.load(root / "fact_addresses" / f"{example['example_id']}.pt", weights_only=True)
    wrong = torch.load(root / "fact_addresses" / f"{donor['example_id']}.pt", weights_only=True)
    rows, vectors = [], {}
    for stage in STAGES:
        snapshot = torch.load(root / "snapshots" / example["example_id"] / f"{stage}.pt", weights_only=True)
        for layer in sorted(own):
            memory = model.model.layers[layer].taal.neural_memory
            weights = {name: value.to(model.device) for name, value in snapshot[layer]["weights"].items()}
            for address, source in (("relevant_key", own), ("wrong_key", wrong)):
                key = source[layer]["key"].to(model.device).reshape(1, 1, -1)
                result = memory.memory_mlp(key, weights=weights).predicted_values
                row = {"evaluation_id": f"{example['example_id']}/{stage}/{layer}/{address}",
                    "example_id": example["example_id"], "stage": stage, "layer": layer,
                    "address": address, "donor_example_id": donor["example_id"],
                    **reconstruction(result, own[layer]["value"].to(model.device))}
                rows.append(row)
                vectors[row["evaluation_id"]] = result.reshape(-1).cpu().tolist()
    return rows, vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if settings["mode"] not in ("reconstruction", "oracle") or settings.get("save_traces") is not True:
        raise ValueError("Select a supported diagnostic with save_traces=true")
    seed_everything(settings["seed"])
    dataset = Path(settings["dataset"]["path"])
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    examples, donors = select_heldout(rows, settings["dataset"]["probe_split"])
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text())
    identity = {"config": settings, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(), "dataset_metadata": metadata}
    ensure_manifest(args.output_dir / "manifest.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"], local_files_only=True)
    model = load_model(args.checkpoint).to("cuda").eval()
    exporter = TaalTraceExporter(args.output_dir / "memory_traces", run_metadata={**identity,
        "tokenizer": metadata["tokenizer"], "capture": "all selected examples/layers; scalar traces plus address vectors"})
    store = PrefixStore(args.output_dir / "prefixes", identity)
    print(f"ADDRESS_DIAGNOSTIC_START mode={settings['mode']} examples={len(examples)} gpu={torch.cuda.get_device_name(0)}", flush=True)
    for index, example in enumerate(examples):
        if not store.contains(example):
            collect_prefix(model, tokenizer, example, args.output_dir, exporter, store, settings)
        print(f"Collected {index + 1}/{len(examples)} {example['example_id']}", flush=True)
    for index, example in enumerate(examples):
        donor = donors[example["example_id"]]
        audit_example(model, tokenizer, example, donor, args.output_dir, exporter, store, settings)
        if settings["mode"] == "reconstruction":
            metrics, vectors = mapping_rows(model, example, donor, args.output_dir)
            existing = read_results(args.output_dir / "mapping_results.jsonl", id_field="evaluation_id")
            for row in metrics:
                if row["evaluation_id"] not in existing:
                    append_result(args.output_dir / "mapping_results.jsonl", row)
            exporter._write_gzip(exporter.root / vector_path(example["example_id"], "reconstruction"), vectors)
        print(f"Audited {index + 1}/{len(examples)} {example['example_id']}", flush=True)
    results = read_results(args.output_dir / "results.jsonl", id_field="evaluation_id")
    expected = len(examples) * (1 if settings["mode"] == "reconstruction" else len(CONDITIONS))
    if len(results) != expected or len(exporter.manifest["episodes"]) != expected:
        raise ValueError("Incomplete score/trace counts")
    if settings["mode"] == "reconstruction":
        mapping_count = len(read_results(args.output_dir / "mapping_results.jsonl", id_field="evaluation_id"))
        if mapping_count != len(examples) * len(STAGES) * len(model.model.initial_memory_states(batch_size=1)) * 2:
            raise ValueError("Incomplete reconstruction coverage")
    summary = {"completed": True, "examples": len(examples), "episodes": expected,
        "condition_counts": {name: sum(row["audit_condition"] == name for row in results.values())
            for name in sorted({row["audit_condition"] for row in results.values()})}}
    summary["scores"] = grouped_score_summary(results.values(), ("audit_condition",))
    if settings["mode"] == "oracle":
        summary["paired_target_log_probability_vs_native"] = {
            condition: sum(
                results[f"{example['example_id']}/{condition}"]["target_log_probability"]
                - results[f"{example['example_id']}/native"]["target_log_probability"]
                for example in examples
            ) / len(examples)
            for condition in CONDITIONS if condition != "native"
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
