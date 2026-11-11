"""Preflight and run the bounded workshop selection/continuation experiments."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml

from data.dataloader import BridgeMemoryCollator
from data.taal_workshop import PARENT, ROOT, MODEL_ID, MODEL_REVISION
from data.taal_d_heldout import D_CHECKPOINT_SHA256
from evaluation.scoring import score_logits
from evaluation.storage import append_result, ensure_manifest, read_results
from evaluation.taal.interventions import audit_conditions
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.state_audit import shared_prefix_examples, split_episode, swap_sources
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import TaalTraceExporter, make_trace_tokens
from training.config import load_config
from training.engine import forward_batch, move_batch, precision_context
from training.optimization import build_optimizer, clip_training_gradients
from training.train import configure_trainable_parameters, load_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate_bundle(root: Path = ROOT, frozen: Path = Path("data/manifests/taal_workshop.json")) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest != json.loads(frozen.read_text()):
        raise ValueError("Workshop dataset manifest differs from the committed split")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Workshop dataset checksum mismatch: {name}")
    splits = {
        name: {doc["document_sha256"] for doc in docs}
        for name, docs in manifest["documents"].items()
    }
    if any(splits[a] & splits[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Natural-text document overlap between splits")
    rows = read_jsonl(root / "evaluation.jsonl")
    if dict(Counter(row["condition"] for row in rows)) != manifest["evaluation_cells"]:
        raise ValueError("Unexpected evaluation cell counts")
    swap_sources(rows, "pair_group")
    shared_prefix_examples(rows, "prefix_group")
    for row in rows:
        prefix, _ = split_episode(row)
        if any(end > len(prefix) - 32 for _, end in row["fact_spans"]):
            raise ValueError(f"Fact still in raw KV window: {row['example_id']}")
    narrow = read_jsonl(root / "narrow.jsonl")
    diverse = read_jsonl(root / "diverse.jsonl")
    if len(narrow) != 144 or len(diverse) != 144 or narrow[128:] != diverse[128:]:
        raise ValueError("Continuation arms must share the same 16 development records")
    for left, right in zip(narrow[:128], diverse[:128]):
        if (left["source_example_id"], left["target_token_id"]) != (right["source_example_id"], right["target_token_id"]):
            raise ValueError("Training record/answer order differs between arms")
    for collection in (narrow, diverse, rows):
        for row in collection:
            digest = row["filler_document_sha256"]
            if digest and digest not in splits[row["evaluation_split"]]:
                raise ValueError(f"Filler document belongs to the wrong split: {row['example_id']}")
    print("WORKSHOP_DATA_VALID " + json.dumps({"files": manifest["files"], "cells": manifest["evaluation_cells"]}), flush=True)
    return manifest


def validate_parent(checkpoint: Path = Path(PARENT)) -> dict:
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    if metadata["progress"]["step"] != 449 or metadata["reason"] == "exception":
        raise ValueError("Workshop continuation requires the successful D449 parent")
    digest = checkpoint_fingerprint(checkpoint)
    if digest != D_CHECKPOINT_SHA256:
        raise ValueError("Parent weights/config differ from verified D449")
    print("WORKSHOP_PARENT_VALID " + json.dumps({"checkpoint": str(checkpoint), "sha256": digest, "step": 449}), flush=True)
    return metadata


def validate_continuation(checkpoint: Path, arm: str, step: int) -> dict:
    metadata = json.loads((checkpoint / "training_metadata.json").read_text())
    settings = load_config(f"training/configs/taal/workshop_{arm}.yaml")
    expected = settings.to_dict()
    saved = metadata["training_config"]
    for section in ("model", "data", "optimizer", "scheduler", "loss", "training", "validation", "checkpoints"):
        if saved[section] != expected[section]:
            raise ValueError(f"Continuation {arm} changed {section}")
    if metadata["progress"]["step"] != step or metadata["reason"] == "exception":
        raise ValueError(f"Expected a usable {arm} checkpoint at {step} successful updates")
    for name in ("step", "skipped_updates", "nan_losses", "nan_gradients", "infinite_losses", "infinite_gradients", "tokens_seen"):
        value = metadata["progress"][name]
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid training progress field: {name}")
    if json.loads((checkpoint / "config.json").read_text()).get("memory_max_write_grad_norm") != 1.0:
        raise ValueError("Continuation checkpoint lost fresh-write clipping")
    print("WORKSHOP_CHECKPOINT_VALID " + json.dumps({"arm": arm, "progress": metadata["progress"], "reason": metadata["reason"]}), flush=True)
    return metadata


def gpu_training_check(arm: str) -> dict:
    """Discard a real longest-row forward/backward/update before starting training."""
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Workshop pilot requires a BF16 CUDA allocation")
    settings = load_config(f"training/configs/taal/workshop_{arm}.yaml")
    device = torch.device("cuda")
    torch.manual_seed(settings.training.seed)
    model = load_model(settings).to(device).train()
    configure_trainable_parameters(model, settings.training.trainable_parameters)
    optimizer = build_optimizer(model, settings)
    rows = read_jsonl(Path(settings.data.dataset_id))[:128]
    row = max(rows, key=lambda row: len(row["input_ids"]))
    batch = move_batch(BridgeMemoryCollator(0.0, 1.0)([row]), device)
    start = time.monotonic()
    with precision_context(device):
        output = forward_batch(model, batch, settings.loss)
    if not torch.isfinite(output.loss):
        raise ValueError("Longest-row training preflight loss is nonfinite")
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError("Longest-row training preflight gradient is missing or nonfinite")
    clip_training_gradients(model, settings)
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Training preflight update produced nonfinite weights")
    torch.cuda.synchronize()
    result = {"arm": arm, "gpu": torch.cuda.get_device_name(), "tokens": len(row["input_ids"]) + 1,
              "loss": output.loss.item(), "seconds": time.monotonic() - start,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    print("WORKSHOP_GPU_TRAIN_VALID " + json.dumps(result), flush=True)
    return result


def summarize_audit(directory: Path, examples: list[dict]) -> dict:
    rows = read_jsonl(directory / "results.jsonl")
    conditions = audit_conditions("standard")
    expected = {f"{example['example_id']}/{condition.name}" for example in examples for condition in conditions}
    actual = {row["evaluation_id"] for row in rows}
    manifest = json.loads((directory / "memory_traces/manifest.json").read_text())
    if actual != expected or len(rows) != len(expected) or set(manifest["episodes"]) != expected:
        raise ValueError("Workshop audit requires every unique score and trace")
    if len(manifest["comparisons"]) != 5 * len(examples):
        raise ValueError("Missing paired memory-control comparisons")
    cells = {}
    for cell in sorted({row["condition"] for row in examples}):
        cells[cell] = {}
        for condition in conditions:
            selected = [row for row in rows if row["dataset_condition"] == cell and row["audit_condition"] == condition.name]
            loss = -sum(row["target_log_probability"] for row in selected) / len(selected)
            cells[cell][condition.name] = {
                "examples": len(selected), "greedy_correct": sum(row["vocabulary_top_1_correct"] for row in selected),
                "donor_correct": sum(row["donor_target_vocabulary_top_1"] for row in selected),
                "answer_loss": loss, "answer_perplexity": math.exp(loss),
                "mean_own_minus_donor_logp": sum(row["own_minus_donor_label_log_probability"] for row in selected) / len(selected),
            }
    by_id = {row["evaluation_id"]: row for row in rows}
    pairs, selection = defaultdict(list), defaultdict(list)
    for example in examples:
        own = by_id[f"{example['example_id']}/correct_full"]
        swapped = by_id[f"{example['example_id']}/swapped"]
        correct = bool(own["vocabulary_top_1_correct"] and swapped["donor_target_vocabulary_top_1"])
        pairs[example["pair_group"]].append(correct)
        if example["selection_group"]:
            selection[example["selection_group"]].append(correct)
    summary = {"rows": len(rows), "traces": len(manifest["episodes"]), "cells": cells,
               "matched_pairs": len(pairs), "pairs_both_directions_correct": sum(all(values) for values in pairs.values()),
               "selection_groups": len(selection), "selection_both_queries_both_assignments_correct": sum(all(values) for values in selection.values()),
               "numerical_failures": 0, "complete": True}
    (directory / "workshop_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def run_audit(checkpoint: Path, directory: Path, *, original_unclipped: bool = False) -> dict:
    settings = yaml.safe_load(Path("evaluation/configs/taal/workshop.yaml").read_text())
    examples = read_jsonl(ROOT / "evaluation.jsonl")
    if original_unclipped:
        settings.pop("memory_max_write_grad_norm")
        settings["dataset"]["end"] = 16
        examples = examples[:16]
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "resolved_audit.yaml"
    ensure_manifest(directory / "workshop_setup.json", {"checkpoint": str(checkpoint), "settings": settings})
    config.write_text(yaml.safe_dump(settings, sort_keys=False))
    subprocess.run([sys.executable, "-m", "evaluation.taal.state_audit", "--config", str(config),
                    "--checkpoint", str(checkpoint), "--output-dir", str(directory)], check=True)
    return summarize_audit(directory, examples)


@torch.inference_mode()
def score_full_context_reference(model, tokenizer, rows: list[dict], directory: Path) -> None:
    results = read_results(directory / "results.jsonl", "example_id")
    exporter = TaalTraceExporter(directory / "memory_traces", run_metadata={
        "model": MODEL_ID, "revision": MODEL_REVISION, "layers": [], "capture": "full_context_host_reference_no_neural_memory",
    })
    for row in rows:
        if row["example_id"] in results:
            continue
        ids = torch.tensor([row["input_ids"]], device=model.device)
        logits = model(ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise ValueError(f"Nonfinite full-context reference: {row['example_id']}")
        prediction = int(logits.argmax())
        outcome = {"example_id": row["example_id"], "dataset_condition": row["condition"], **score_logits(logits, row, tokenizer)}
        exporter.export(TraceEpisode(
            run_id=directory.name, example_id=row["example_id"], condition_id="full_context_qwen",
            checkpoint=MODEL_ID, tokenizer_id=MODEL_ID,
            tokens=make_trace_tokens(row["input_ids"], tokenizer) + make_trace_tokens([prediction], tokenizer, offset=len(row["input_ids"]), phase="generated"),
            writes=[], reads=[], internal_prefixes=[], outcome=outcome,
            metadata={"neural_memory": "not_present", "full_fact_visible": True,
                      "expected_answer": row["answer"], "expected_token_id": row["target_token_id"]},
        ))
        append_result(directory / "results.jsonl", outcome)
    final = read_results(directory / "results.jsonl", "example_id")
    trace_ids = json.loads((directory / "memory_traces/manifest.json").read_text())["episodes"]
    expected_ids = {row["example_id"] for row in rows}
    expected_traces = {f"{example_id}/full_context_qwen" for example_id in expected_ids}
    if set(final) != expected_ids or set(trace_ids) != expected_traces:
        raise ValueError("Full-context reference has incomplete scores/traces")
    if any(not (directory / "memory_traces" / relative).is_file() for relative in trace_ids.values()):
        raise ValueError("Full-context reference is missing a trace file")
    print("WORKSHOP_FULL_CONTEXT_COMPLETE " + json.dumps({"examples": len(final)}), flush=True)


def full_context_reference(directory: Path) -> None:
    """Unadapted host with every source fact visible; no memory signals exist."""
    rows = read_jsonl(ROOT / "evaluation.jsonl")
    ensure_manifest(directory / "manifest.json", {"model": MODEL_ID, "revision": MODEL_REVISION,
                    "dataset_sha256": hashlib.sha256((ROOT / "evaluation.jsonl").read_bytes()).hexdigest(),
                    "attention": "full_context", "parameter_dtype": "bfloat16"})
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
                                               attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    score_full_context_reference(model, tokenizer, rows, directory)


def run_baselines(directory: Path) -> None:
    validate_parent()
    run_audit(Path(PARENT), directory / "D449-unclipped-reference", original_unclipped=True)
    run_audit(Path(PARENT), directory / "D449-clip1-update0")
    full_context_reference(directory / "full-context-qwen")


def run_matched_audits(directory: Path, narrow_run: str, diverse_run: str) -> None:
    """Bundle both dependent audits to stay inside Explorer's eight-job limit."""
    failures = []
    available = []
    # The primary matched endpoint comes first; step100 is a learning-curve control.
    for step in (200, 100):
        checkpoints = {arm: Path("checkpoints") / run / f"step-{step}"
                       for arm, run in (("narrow", narrow_run), ("diverse", diverse_run))}
        if not all(checkpoint.exists() for checkpoint in checkpoints.values()):
            print(f"WORKSHOP_MATCHED_STEP_MISSING step={step}; no unequal-step substitution", flush=True)
            continue
        available.append(step)
        for arm, checkpoint in checkpoints.items():
            try:
                metadata = validate_continuation(checkpoint, arm, step)
                destination = directory / f"{arm}-step{step}"
                summary = run_audit(checkpoint, destination)
                (destination / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                print("WORKSHOP_AUDIT_COMPLETE " + json.dumps({"arm": arm, "step": step, "rows": summary["rows"]}), flush=True)
            except (ValueError, subprocess.CalledProcessError) as error:
                failures.append({"arm": arm, "step": step, "error": str(error)})
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if not available or failures or 200 not in available:
        raise RuntimeError(f"Matched comparison incomplete; available steps={available}; failures={failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "preflight", "baseline", "matched"))
    parser.add_argument("--arm", choices=("narrow", "diverse"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--narrow-run")
    parser.add_argument("--diverse-run")
    args = parser.parse_args()
    validate_bundle()
    if args.mode in ("validate", "preflight"):
        validate_parent()
    if args.mode == "preflight":
        if args.arm is None:
            parser.error("preflight requires --arm")
        gpu_training_check(args.arm)
    elif args.mode in ("baseline", "matched"):
        if args.output_dir is None:
            parser.error("audit modes require --output-dir")
        if args.mode == "baseline":
            run_baselines(args.output_dir)
        else:
            if not args.narrow_run or not args.diverse_run:
                parser.error("matched audits require both parent run IDs")
            run_matched_audits(args.output_dir, args.narrow_run, args.diverse_run)


if __name__ == "__main__":
    main()
