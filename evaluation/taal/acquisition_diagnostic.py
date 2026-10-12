"""Separate immediate label acquisition from subsequent memory drift.

Native trajectories are unchanged. Offline MLP readouts compare the actual
label transition with its fresh-gradient-only and unchanged-weight controls.
"""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.state import NeuralMemoryStates
from evaluation.scoring import grouped_score_summary, score_logits
from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.address_diagnostic import AddressCapture, NativeWrite, reconstruction, vector_path
from evaluation.taal.lifecycle import lifecycle_boundaries
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.state_audit import clone_memory_states, load_model, select_examples
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import (
    TaalTraceExporter, TaalTraceRecorder, make_trace_tokens, trace_greedy_prediction,
)
from training.config import load_config
from training.train import initialize_taal_gates, load_model as load_fresh_model
from utils.seed import seed_everything


STAGES = ("before_label", "fresh_only", "actual_after_label", "after_fact", "middle", "pre_query")


@dataclass
class AcquisitionRecord:
    example: dict
    label_writes: dict[int, NativeWrite]
    snapshots: dict[str, NeuralMemoryStates]
    outcome: dict


def opposing_donors(examples: list[dict]) -> dict[str, dict]:
    donors = {}
    for example in examples:
        matches = [other for other in examples
                   if other["record_name"] == example["record_name"]
                   and other["target_token_id"] != example["target_token_id"]]
        if len(matches) != 1:
            raise ValueError("Acquisition requires exactly one opposing-label donor per record")
        donors[example["example_id"]] = matches[0]
    return donors


@torch.inference_mode()
def collect_acquisition(model, tokenizer, example, exporter, execution_block_size: int) -> AcquisitionRecord:
    ids = example["input_ids"]
    label = example["fact_position"]
    boundaries = lifecycle_boundaries(example, tokenizer)
    layers = sorted(model.model.initial_memory_states(batch_size=1))
    recorder = TaalTraceRecorder(model, layers, len(ids))
    capture = AddressCapture(model, start=label, capture_writes=True)
    snapshots = {}
    state, start = None, 0
    with recorder:
        for stage, end in (
            ("before_label", label), ("actual_after_label", label + 1),
            ("after_fact", boundaries["fact"]), ("middle", boundaries["middle"]),
            ("pre_query", boundaries["final"]), ("answer_driver", len(ids)),
        ):
            if end <= start:
                raise ValueError("Acquisition stages must be strictly ordered")
            if stage == "actual_after_label":
                with capture:
                    output = model.prefill(torch.tensor([ids[start:end]], device=model.device),
                        state=state, execution_block_size=1, prepend_memory_tokens=False)
            else:
                output = model.prefill(torch.tensor([ids[start:end]], device=model.device),
                    state=state, execution_block_size=execution_block_size,
                    prepend_memory_tokens=start == 0)
            state = output.state
            if stage in ("after_fact", "middle", "pre_query"):
                snapshots[stage] = clone_memory_states(state.memory_states, "cpu")
            if stage == "pre_query":
                if bool((state.past_key_values.layers[0].positions < boundaries["fact"]).any()):
                    raise ValueError("Original fact remains in query KV; not an evicted-fact test")
            start = end
    label_writes = {
        layer: NativeWrite(
            clone_memory_states(write.before, "cpu"), clone_memory_states(write.after, "cpu"),
            write.keys.cpu(), write.values.cpu(), write.inputs.cpu(), write.mask.cpu(), write.query.cpu(),
        )
        for layer, write in capture.writes.items()
    }
    outcome = {"evaluation_id": f"{example['example_id']}/native",
               "example_id": example["example_id"], "condition": "native",
               **score_logits(output.logits[0, -1], example, tokenizer)}
    prediction = trace_greedy_prediction(model, tokenizer,
        token_id=outcome["vocabulary_top_token_id"], position=len(ids), state=state,
        layers=layers, memory_read_scale=1.0)
    metadata = exporter.manifest["run_metadata"]
    exporter.export(TraceEpisode(
        exporter.root.parent.name, example["example_id"], "native", metadata["checkpoint"],
        metadata["tokenizer"], make_trace_tokens(ids, tokenizer) + [prediction.token],
        recorder.writes + prediction.writes, recorder.reads + prediction.reads,
        recorder.internal_prefixes + prediction.internal_prefixes, outcome,
        {"expected_answer": example["answer"], "expected_token_id": example["target_token_id"],
         "query_start_position": example["final_query_position"], "label_position": label,
         "acquisition_vectors": str(vector_path(example["example_id"], "acquisition")),
         "control_scope": "Offline weight readouts only; native trajectory is unchanged"},
    ))
    return AcquisitionRecord(example, label_writes, snapshots, outcome)


@torch.inference_mode()
def acquisition_rows(model, own: AcquisitionRecord, donor: AcquisitionRecord) -> tuple[list[dict], list[dict]]:
    rows, vectors = [], []
    for layer, write in own.label_writes.items():
        memory = model.model.layers[layer].taal.neural_memory
        before = clone_memory_states(write.before, model.device)[layer]
        keys, values, inputs, mask = [value.to(model.device)
                                    for value in (write.keys, write.values, write.inputs, write.mask)]
        B, C, D = keys.shape
        if (B, C) != (1, 1):
            raise ValueError("Immediate acquisition requires B=C=1")
        strength = memory.write_strength_projection(inputs).sigmoid().reshape(B, C).float()
        strength = torch.where(mask, strength, 0.0)
        gradient = memory._chunk_gradient(before.weights, keys, values, strength).chunk_gradients
        fresh = replace(before, weights={name: weight - gradient[name]
                                        for name, weight in before.weights.items()})
        actual = clone_memory_states(write.after, model.device)[layer]
        replayed = memory._update(before, inputs, keys, values, mask)
        for name in actual.weights:
            torch.testing.assert_close(replayed.weights[name], actual.weights[name], rtol=0, atol=0)
            torch.testing.assert_close(replayed.momentum[name], actual.momentum[name], rtol=0, atol=0)
        states = {"before_label": before, "fresh_only": fresh, "actual_after_label": actual}
        states.update({stage: clone_memory_states({layer: snapshot[layer]}, model.device)[layer]
                       for stage, snapshot in own.snapshots.items()})
        wrong_write = donor.label_writes[layer]
        own_value = values.reshape(D)
        wrong_value = wrong_write.values.to(model.device).reshape(D)
        controls = {
            "write_strength": float(strength[0, 0]),
            "forget": float(memory.forget_projection(inputs[:, 0]).sigmoid()[0, 0]),
            "momentum_retention": float(memory.momentum_projection(inputs[:, 0]).sigmoid()[0, 0]),
            "fresh_gradient_norm": float(torch.stack([g.square().sum() for g in gradient.values()]).sum().sqrt()),
            "value_pair_distance": float((own_value - wrong_value).norm()),
        }
        for stage in STAGES:
            for address, key in (("own", keys), ("opposing", wrong_write.keys.to(model.device))):
                output = memory.memory_mlp(key.float(), weights=states[stage].weights).predicted_values
                relevant = reconstruction(output, own_value)
                wrong = reconstruction(output, wrong_value)
                row = {
                    "evaluation_id": f"{own.example['example_id']}/{layer}/{stage}/{address}",
                    "example_id": own.example["example_id"], "donor_example_id": donor.example["example_id"],
                    "layer": layer, "stage": stage, "address": address, **controls, **relevant,
                    "opposing_value_mse": wrong["mse"],
                    "value_selectivity_margin": wrong["mse"] - relevant["mse"],
                    "native_update_replay_exact": True,
                }
                rows.append(row)
                vectors.append({**row, "key": key.float().reshape(-1).cpu().tolist(),
                                "own_value": own_value.cpu().tolist(),
                                "opposing_value": wrong_value.cpu().tolist(),
                                "read_output": output.float().reshape(-1).cpu().tolist()})
    return rows, vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if settings.get("save_traces") is not True:
        raise ValueError("Acquisition diagnostics require save_traces=true")
    dataset = Path(settings["dataset"]["path"])
    examples = select_examples([json.loads(line) for line in dataset.read_text().splitlines()], settings["dataset"])
    donors = opposing_donors(examples)
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text())
    identity = {"config": settings, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest()}
    ensure_manifest(args.output_dir / "manifest.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"], local_files_only=True)
    summaries = {}
    for model_name in ("trained", "fresh"):
        seed_everything(settings["seed"])
        if model_name == "trained":
            model = load_model(args.checkpoint).to("cuda").eval()
            checkpoint = str(args.checkpoint)
        else:
            config = load_config(settings["fresh_training_config"])
            model = load_fresh_model(config).to(device="cuda", dtype=torch.bfloat16).eval()
            initialize_taal_gates(model, config.training.initial_taal_gate)
            checkpoint = "fresh-seed-42-reference-not-original-initial-snapshot"
        if model.config.memory_chunk_size != 1:
            raise ValueError("Immediate acquisition requires memory_chunk_size=1")
        layers = sorted(model.model.initial_memory_states(batch_size=1))
        root = args.output_dir / model_name
        exporter = TaalTraceExporter(root / "memory_traces", run_metadata={
            **identity, "checkpoint": checkpoint, "tokenizer": metadata["tokenizer"],
            "tokenizer_revision": metadata["tokenizer_revision"], "layers": layers,
            "reference": model_name, "precision": "BF16 slow parameters, FP32 fast weights",
            "capture": "All selected episodes/layers, native traces plus acquisition-vector sidecars",
        })
        print(f"ACQUISITION_START model={model_name} examples={len(examples)} gpu={torch.cuda.get_device_name(0)}", flush=True)
        records = {}
        for index, example in enumerate(examples):
            records[example["example_id"]] = collect_acquisition(
                model, tokenizer, example, exporter, settings["execution_block_size"])
            print(f"Collected {model_name} {index + 1}/{len(examples)}", flush=True)
        mapping = []
        saved_mapping = read_results(root / "mapping_results.jsonl", id_field="evaluation_id")
        saved_scores = read_results(root / "results.jsonl", id_field="evaluation_id")
        for example in examples:
            record = records[example["example_id"]]
            rows, vectors = acquisition_rows(model, record, records[donors[example["example_id"]]["example_id"]])
            for row in rows:
                if row["evaluation_id"] not in saved_mapping:
                    append_result(root / "mapping_results.jsonl", row)
            mapping.extend(rows)
            exporter._write_gzip(exporter.root / vector_path(example["example_id"], "acquisition"),
                                 {"schema": "taal-acquisition-vectors/v1", "rows": vectors})
            if record.outcome["evaluation_id"] not in saved_scores:
                append_result(root / "results.jsonl", record.outcome)
        expected = len(examples) * len(layers) * len(STAGES) * 2
        saved_count = len(read_results(root / "mapping_results.jsonl", id_field="evaluation_id"))
        if (len(mapping) != expected or saved_count != expected
            or len(exporter.manifest["episodes"]) != len(examples)
            or len(read_results(root / "results.jsonl", id_field="evaluation_id")) != len(examples)):
            raise ValueError("Incomplete acquisition score/trace coverage")
        summaries[model_name] = {
            "completed": True, "examples": len(examples), "trace_episodes": len(examples),
            "mapping_rows": len(mapping), "scores": grouped_score_summary(
                [record.outcome for record in records.values()], ("condition",)),
            "mapping": {
                f"{stage}/{address}": {
                    field: sum(row[field] for row in mapping if row["stage"] == stage and row["address"] == address)
                    / (len(examples) * len(layers))
                    for field in ("relative_error", "mse", "cosine", "value_selectivity_margin")
                } for stage in STAGES for address in ("own", "opposing")
            },
        }
        del model, records
        torch.cuda.empty_cache()
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    print("ACQUISITION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
