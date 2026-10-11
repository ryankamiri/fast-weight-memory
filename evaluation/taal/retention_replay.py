"""Fixed-input memory recurrence replay and matched recipient-KV query audits.

Shadow W/S states consume the *normal* run's keys, values and control inputs.
They do not feed back into the transformer during filler. Query traces are
real model forwards after transplanting each shadow state into normal KV.
"""

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
import yaml
from transformers import AutoTokenizer

from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.lifecycle import lifecycle_boundaries
from evaluation.taal.prefix_store import PrefixStore, PrefixTrace, checkpoint_fingerprint
from architectures.titans.state import NeuralMemoryState
from evaluation.taal.state_audit import clone_memory_states, fork_session_state, load_model, score_logits
from evaluation.taal.state_readout_probe import (
    StateFeatures, StateProjector, fit_probes, save_tensor_payload, validate_splits,
)
from evaluation.taal.trace_contract import TraceEpisode, WriteEvent
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder, make_trace_tokens, trace_greedy_prediction


POLICIES = ("normal", "no_fresh", "no_forget", "no_momentum", "frozen")


def scalar_write(layer, position, old, gradients, new, mask, strength, *, updates_enabled=True):
    # Accumulate on device. Transfer only after the normal filler forward ends.
    proposed = sum(value.double().square().sum() for value in gradients.values()).sqrt()
    previous = sum(value.double().square().sum() for value in old.weights.values()).sqrt()
    delta = {name: new.weights[name] - old.weights[name] for name in old.weights}
    net = sum(value.double().square().sum() for value in delta.values()).sqrt()
    other = sum((delta[name] + gradients[name]).double().square().sum() for name in delta).sqrt()
    dot = sum((-gradients[name].double() * delta[name].double()).sum() for name in delta)
    return (layer, position, updates_enabled, torch.stack((
        proposed, previous, net, other, dot, mask[0, 0].double(), strength[0, 0].double(),
    )).detach())


class FixedInputReplay:
    def __init__(self, model, acquired, *, start, policies=POLICIES):
        if model.training or model.config.memory_chunk_size != 1:
            raise ValueError("Fixed-input replay requires eval mode and chunk size one")
        if not policies or policies[0] != "normal" or not set(policies) <= set(POLICIES):
            raise ValueError("Replay must include normal first and valid controls")
        self.model = model
        self.policies = policies
        self.states = {policy: clone_memory_states(acquired) for policy in policies}
        self.positions = {layer: start for layer in acquired}
        self.events = {policy: [] for policy in policies}
        self.inputs = {layer: [] for layer in acquired}
        self.originals = {}

    def _wrap(self, layer, original):
        def update(memory, state, inputs, keys, values, write_mask):
            B, C, D = keys.shape
            if (B, C) != (1, 1):
                raise ValueError("Replay captures exactly one session/token per update")
            normal = original(state, inputs, keys, values, write_mask)
            self.inputs[layer].append(torch.cat((inputs.float(), keys, values), -1).detach())
            observer = memory.trace_observer
            previous = (memory.updates_enabled, memory.forgetting_enabled, memory.momentum_enabled)
            position = self.positions[layer]
            try:
                for policy in self.policies:
                    memory.updates_enabled = policy != "frozen"
                    memory.forgetting_enabled = policy != "no_forget"
                    memory.momentum_enabled = policy != "no_momentum"
                    mask = torch.zeros_like(write_mask) if policy == "no_fresh" else write_mask
                    def record(old, gradients, new, mask, strength, *, updates_enabled=True):
                        per_token = {name: value[:, 0] for name, value in gradients.items()}
                        self.events[policy].append(scalar_write(layer, position, old, per_token, new, mask, strength,
                                                               updates_enabled=updates_enabled))
                    memory.trace_observer = record
                    self.states[policy][layer] = original(self.states[policy][layer], inputs, keys, values, mask)
            finally:
                memory.trace_observer = observer
                memory.updates_enabled, memory.forgetting_enabled, memory.momentum_enabled = previous
            self.positions[layer] += 1
            return normal
        return update

    def __enter__(self):
        for layer in self.states["normal"]:
            memory = self.model.model.layers[layer].taal.neural_memory
            self.originals[layer] = memory._update
            memory._update = MethodType(self._wrap(layer, memory._update), memory)
        return self

    def __exit__(self, *_):
        for layer, original in self.originals.items():
            self.model.model.layers[layer].taal.neural_memory._update = original

    def writes(self, policy):
        events = self.events[policy]
        if not events:
            return []
        numbers = torch.stack([event[-1] for event in events]).cpu().tolist()
        writes = []
        for (layer, position, enabled, _), (proposed, previous, net, other, dot, mask, strength) in zip(events, numbers):
            writes.append(WriteEvent(layer, position, 0, None, bool(mask), strength, proposed, previous,
                net, other, dot / (proposed * net) if proposed > 0 and net > 0 else None,
                updates_enabled=enabled))
        return writes


@contextmanager
def transplanted_filler_state(normal, shadow):
    # Preserve recipient convolution histories and KV. Only W/S are replayed.
    states = {layer: replace(normal.memory_states[layer], weights=shadow[layer].weights,
                              momentum=shadow[layer].momentum) for layer in shadow}
    yield fork_session_state(normal, states)


@torch.inference_mode()
def collect(model, tokenizer, example, root, exporter, projector, settings, store):
    existing = read_results(root / "results.jsonl", id_field="evaluation_id")
    ids = example["input_ids"]
    boundaries = lifecycle_boundaries(example, tokenizer)
    start, end = boundaries["fact"], boundaries["final"]
    if example.get("distractor_end", start) > end - model.config.working_memory_size:
        raise ValueError("Competing fact remains attention-visible at query")
    common = TaalTraceRecorder(model, projector.layers, start)
    with common:
        acquired = model.prefill(torch.tensor([ids[:start]], device=model.device),
            execution_block_size=settings["execution_block_size"]).state
    features = {"fact": projector(acquired.memory_states)}
    filler = TaalTraceRecorder(model, projector.layers, end - start, position_offset=start)
    replay = FixedInputReplay(model, acquired.memory_states, start=start, policies=tuple(settings["policies"]))
    with filler, replay:
        state = acquired
        for stage in ("middle", "final"):
            stop = boundaries[stage]
            state = model.prefill(torch.tensor([ids[start:stop]], device=model.device), state=state,
                execution_block_size=settings["execution_block_size"], prepend_memory_tokens=False).state
            for policy in replay.policies:
                features[f"{policy}/{stage}"] = projector(replay.states[policy])
            start = stop
    if bool((state.past_key_values.layers[0].positions < boundaries["fact"]).any()):
        raise ValueError("Original fact remains in KV")
    if any(position != end for position in replay.positions.values()):
        raise ValueError("Replay token count mismatch")
    for layer, shadow in replay.states["normal"].items():
        for name, value in shadow.weights.items():
            torch.testing.assert_close(value, state.memory_states[layer].weights[name], rtol=0, atol=0)
            torch.testing.assert_close(shadow.momentum[name], state.memory_states[layer].momentum[name], rtol=0, atol=0)
    store.save(example, state, PrefixTrace(common.writes + filler.writes, common.reads + filler.reads, common.internal_prefixes))
    save_tensor_payload(root / "replay_inputs" / f"{example['example_id']}.pt", {
        layer: torch.cat(values, 1).cpu() for layer, values in replay.inputs.items()})
    # Matched donor uses the same record and context family, opposing original
    # label. Derive its shadow bank in the outer collector, never from KV.
    save_tensor_payload(root / "final_states" / f"{example['example_id']}.pt", {
        policy: {layer: {component: {name: value.cpu() for name, value in getattr(memory, component).items()}
                        for component in ("weights", "momentum")}
                 for layer, memory in bank.items()} for policy, bank in replay.states.items()})
    for policy in replay.policies:
        for scale in (1.0, 0.0):
            condition = policy if scale else f"{policy}_reads_off"
            query_recorder = TaalTraceRecorder(model, projector.layers, len(ids) - end, position_offset=end)
            with transplanted_filler_state(state, replay.states[policy]) as query_state, query_recorder:
                output = model.prefill(torch.tensor([ids[end:]], device=model.device), state=query_state,
                    memory_read_scale=scale, execution_block_size=settings["execution_block_size"], prepend_memory_tokens=False)
            scores = score_logits(output.logits[0, -1], example, tokenizer)
            outcome = {"evaluation_id": f"{example['example_id']}/{condition}", "example_id": example["example_id"],
                       "condition": condition, "context_family": example["context_family"],
                       "probe_split": example["probe_split"], **scores}
            if "distractor_token_id" in example:
                logp = output.logits[0, -1].float().log_softmax(-1)
                outcome["distractor_log_probability"] = float(logp[example["distractor_token_id"]])
                outcome["distractor_generated"] = scores["vocabulary_top_token_id"] == example["distractor_token_id"]
            if outcome["evaluation_id"] not in existing:
                append_result(root / "results.jsonl", outcome)
            prediction = trace_greedy_prediction(model, tokenizer, token_id=scores["vocabulary_top_token_id"],
                position=len(ids), state=output.state, layers=projector.layers, memory_read_scale=scale)
            exporter.export(TraceEpisode(root.name, example["example_id"], condition,
                exporter.manifest["run_metadata"]["checkpoint"], exporter.manifest["run_metadata"]["tokenizer"],
                make_trace_tokens(ids, tokenizer) + [prediction.token],
                common.writes + (filler.writes if policy == "normal" else replay.writes(policy)) + query_recorder.writes + prediction.writes,
                common.reads + (filler.reads if policy == "normal" else []) + query_recorder.reads + prediction.reads,
                common.internal_prefixes + prediction.internal_prefixes, outcome,
                {"expected_answer": example["answer"], "query_start_position": end,
                 "filler_semantics": "fixed normal inputs; shadow updates do not feed back into Qwen",
                 "shadow_filler_reads": "unmeasured, not zero", "query_kv": "same normal recipient",
                 "distractor_answer": example.get("distractor_answer"), "snapshot_boundaries": boundaries}))
    return features


@torch.inference_mode()
def swapped_queries(model, tokenizer, examples, root, exporter, projector, settings, store):
    existing = read_results(root / "results.jsonl", id_field="evaluation_id")
    for example in examples:
        donors = [row for row in examples if row["context_family"] == example["context_family"]
                  and row["record_name"] == example["record_name"]
                  and row.get("filler_variant") == example.get("filler_variant")
                  and row["target_token_id"] != example["target_token_id"]]
        if len(donors) != 1:
            raise ValueError("Need exactly one matched opposing-label donor")
        donor = donors[0]
        saved = store.load(example, model.device)
        payload = torch.load(root / "final_states" / f"{donor['example_id']}.pt", weights_only=True)
        end, ids = example["final_query_position"], example["input_ids"]
        for policy in settings["policies"]:
            condition = f"{policy}_swapped"
            identity = f"{example['example_id']}/{condition}"
            bank = {layer: NeuralMemoryState(**{component: {name: value.to(model.device) for name, value in values.items()}
                                                for component, values in memory.items()})
                    for layer, memory in payload[policy].items()}
            recorder = TaalTraceRecorder(model, projector.layers, len(ids) - end, position_offset=end)
            with transplanted_filler_state(saved.state, bank) as query_state, recorder:
                output = model.prefill(torch.tensor([ids[end:]], device=model.device), state=query_state,
                    execution_block_size=settings["execution_block_size"], prepend_memory_tokens=False)
            scores = score_logits(output.logits[0, -1], example, tokenizer)
            logp = output.logits[0, -1].float().log_softmax(-1)
            outcome = {"evaluation_id": identity, "example_id": example["example_id"], "condition": condition,
                       "context_family": example["context_family"], "probe_split": example["probe_split"],
                       "donor_example_id": donor["example_id"], "donor_answer": donor["answer"],
                       "donor_log_probability": float(logp[donor["target_token_id"]]), **scores}
            if identity not in existing:
                append_result(root / "results.jsonl", outcome)
            prediction = trace_greedy_prediction(model, tokenizer, token_id=scores["vocabulary_top_token_id"],
                position=len(ids), state=output.state, layers=projector.layers, memory_read_scale=1.0)
            exporter.export(TraceEpisode(root.name, example["example_id"], condition,
                exporter.manifest["run_metadata"]["checkpoint"], exporter.manifest["run_metadata"]["tokenizer"],
                make_trace_tokens(ids, tokenizer) + [prediction.token], saved.trace.writes + recorder.writes + prediction.writes,
                saved.trace.reads + recorder.reads + prediction.reads, saved.trace.internal_prefixes + prediction.internal_prefixes,
                outcome, {"expected_answer": example["answer"], "query_start_position": end,
                          "prefix_semantics": "normal recipient prefix before donor transplant",
                          "donor_distractor_answer": donor.get("distractor_answer"),
                          "swap_scope": "W/S only; recipient KV and convolution retained"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())
    if settings.get("save_traces", True) is not True:
        raise ValueError("These diagnostic evaluations require traces")
    torch.set_num_threads(4)
    dataset = Path(settings["dataset"]["path"])
    examples = [json.loads(line) for line in dataset.read_text().splitlines() if line]
    # Interference variants are paired, not duplicate training observations.
    variants = sorted({row.get("filler_variant", "neutral") for row in examples})
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text())
    manifest = {"config": settings, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_fingerprint(args.checkpoint), "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "dataset_metadata": metadata}
    ensure_manifest(args.output_dir / "manifest.json", manifest)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"], local_files_only=True)
    model = load_model(args.checkpoint).to("cuda").eval()
    projector = StateProjector(model.model.initial_memory_states(batch_size=1), dimension=settings["projection_dim"], seed=settings["seed"])
    exporter = TaalTraceExporter(args.output_dir / "memory_traces", run_metadata={**manifest,
        "tokenizer": metadata["tokenizer"], "layers": projector.layers, "capture": "all_examples_all_layers_compact_scalars"})
    store = PrefixStore(args.output_dir / "prefixes", manifest)
    all_features = {}
    for index, example in enumerate(examples):
        destination = args.output_dir / "features" / f"{example['example_id']}.pt"
        if destination.exists():
            row = {stage: StateFeatures(**values) for stage, values in torch.load(destination, weights_only=True).items()}
        else:
            row = collect(model, tokenizer, example, args.output_dir, exporter, projector, settings, store)
            save_tensor_payload(destination, {stage: {"weights": values.weights, "momentum": values.momentum}
                                             for stage, values in row.items()})
        all_features[example["example_id"]] = row
        print(f"Completed {index + 1}/{len(examples)} {example['example_id']}", flush=True)
    swapped_queries(model, tokenizer, examples, args.output_dir, exporter, projector, settings, store)
    summary = {}
    for variant in variants:
        selected = [row for row in examples if row.get("filler_variant", "neutral") == variant]
        validate_splits(selected)
        features = {stage: StateFeatures(
            torch.stack([all_features[row["example_id"]][stage].weights for row in selected]),
            torch.stack([all_features[row["example_id"]][stage].momentum for row in selected]))
            for stage in all_features[selected[0]["example_id"]]}
        summary[variant], _ = fit_probes(selected, features, groups={"all": projector.layers,
            **{f"layer-{layer}": [layer] for layer in projector.layers}}, strengths=settings["ridge_strengths"],
            shuffled_controls=settings["shuffled_controls"], seed=settings["seed"])
        if variant == "competing":
            distractor = [{**row, "target_token_id": row["distractor_token_id"]} for row in selected]
            summary["distractor"], _ = fit_probes(distractor, features, groups={"all": projector.layers},
                strengths=settings["ridge_strengths"], shuffled_controls=settings["shuffled_controls"], seed=settings["seed"])
    expected = len(examples) * len(settings["policies"]) * 3
    rows = read_results(args.output_dir / "results.jsonl", id_field="evaluation_id")
    if len(rows) != expected or len(exporter.manifest["episodes"]) != expected:
        raise ValueError("Incomplete condition scores or trace exports")
    summary["condition_counts"] = {condition: sum(row["condition"] == condition for row in rows.values())
                                   for condition in sorted({row["condition"] for row in rows.values()})}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
