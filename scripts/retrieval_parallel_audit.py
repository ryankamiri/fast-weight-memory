"""Matched untouched baselines and complete validation for the parallel pilots."""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.ttcd.qwen.causal_lm import TTCDQwen3ForCausalLM
from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from evaluation.models import load_model
from evaluation.shared.__main__ import load_rows, source_fingerprint, tokenizer_fingerprint
from evaluation.shared.backends import QwenBackend
from evaluation.shared.benchmarks import ControlledBenchmark
from evaluation.shared.retrieval_diagnostics import selection_counts
from evaluation.shared.runner import validate_predictions
from evaluation.storage import atomic_json, ensure_manifest, file_sha256, read_results
from evaluation.taal.prefix_store import checkpoint_fingerprint
from scripts.retrieval_acquisition import validate_checkpoint
from scripts.retrieval_exposure_audit import checked_rows, generation_audit, load_checkpoint, token_diagnostics
from training.config import load_config
from training.train import configure_trainable_parameters, verify_loading
from utils.precision import precision_context


ARMS = ("r3", "b3", "r3_fp32", "s3", "r3_eos")


def recipe_path(arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown parallel arm: {arm}")
    return Path(f"training/configs/taal/retrieval_parallel_{arm}.yaml")


def validation_rows():
    path = Path("data/shared_retrieval/val.jsonl")
    manifest = json.loads(path.with_name("manifest.json").read_text())
    if file_sha256(path) != manifest["files"]["val"]["sha256"]:
        raise ValueError("Full validation checksum changed")
    rows = load_rows(path)
    if len(rows) != 512 or len({r["metadata"]["family_id"] for r in rows}) != 64:
        raise ValueError("Expected 512 validation questions from 64 complete families")
    return rows


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def audit_model(model, tokenizer, provenance, output, allocation, reads_off=True):
    cohorts = {"fit": checked_rows("common-fit"), "val": validation_rows()}
    reference = [r for r in load_rows(Path("data/taal_workshop/evaluation.jsonl")) if r["condition"] == "reference"]
    if len(reference) != 16:
        raise ValueError("Expected the original 16-question regression slice")
    for name, rows in cohorts.items():
        if not token_diagnostics(model, rows, output / f"{name}-loss", "loss", allocation):
            return False
        if not generation_audit(model, tokenizer, provenance, rows, output / name, allocation):
            return False
    if not generation_audit(model, tokenizer, provenance, reference, output / "single-fact", allocation, workshop=True):
        return False
    if reads_off:
        # These are the original matched 128 validation rows, not a new sample.
        rows = load_rows(Path("data/retrieval_exposure/repeat/val.jsonl"))
        if not generation_audit(model, tokenizer, provenance, rows, output / "val-reads-off", allocation, 0.0):
            return False
    return True


def audit_trained(arm, checkpoint, output, allocation, device):
    metadata = validate_checkpoint(recipe_path(arm), checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    config = TaalQwen3Config.from_pretrained(checkpoint)
    model = load_model(str(checkpoint), config, dtype=torch.float32).to(device)
    provenance = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_fingerprint(checkpoint),
        "source_sha256": source_fingerprint(), "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
        "tokenizer": str(checkpoint), "precision": config.execution_precision,
        "progress": metadata["progress"],
    }
    parameters = configure_trainable_parameters(model, "taal_only")
    ensure_manifest(output / "checkpoint.json", provenance | {"trainable_parameters": sum(p.numel() for p in parameters)})
    try:
        complete = audit_model(model, tokenizer, provenance, output, allocation, config.adapter_type == "memory")
        if complete and arm == "r3_fp32":
            # Separate receipt/output for the secondary cross-precision slice.
            model.config.execution_precision = "bf16"
            complete = generation_audit(model, tokenizer, provenance | {"precision": "bf16"},
                                        checked_rows("common-fit")[:16], output / "bf16-cross-check", allocation)
        return complete
    finally:
        del parameters
        del model
        release()


def baseline_config(base, window):
    return TTCDQwen3Config.from_dict(base.to_dict() | {
        "fast_weight_layers": [], "teacher_window_size": window, "student_window_size": window,
        "max_persistent_kv_tokens": 0, "chunk_size": 128,
    })


def load_baseline(name, device, output, parity_row=None):
    settings = load_config(recipe_path("r3")).model
    base = Qwen3Config.from_pretrained(settings.model_id, revision=settings.revision)
    window = 4096 if name == "u_full" else 128
    config = baseline_config(base, window)
    model, loading = TTCDQwen3ForCausalLM.from_pretrained(
        settings.model_id, revision=settings.revision, config=config,
        dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True,
    )
    verify_loading(model, loading)
    if loading["missing_keys"] or loading["unexpected_keys"]:
        raise ValueError("Untouched baseline cannot initialize any new parameters")
    model.requires_grad_(False).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(settings.model_id, revision=settings.revision)
    provenance = {"checkpoint": settings.model_id, "revision": settings.revision,
                  "tokenizer": settings.model_id, "tokenizer_sha256": tokenizer_fingerprint(tokenizer),
                  "source_sha256": source_fingerprint(), "precision": "bf16", "attention_window": window,
                  "trainable_parameters": 0, "fast_weight_layers": []}
    if name == "u_full":
        # Compare actual pretrained weights through original HF Qwen and our wrapper.
        original = Qwen3ForCausalLM.from_pretrained(settings.model_id, revision=settings.revision,
                    dtype=torch.float32, attn_implementation="sdpa").to(device).eval()
        if parity_row is None:
            parity_row = checked_rows("common-fit")[0]
        ids = torch.tensor([parity_row["prompt_input_ids"]], device=device)
        with torch.inference_mode(), precision_context(device, "fp32"):
            expected = original(ids, logits_to_keep=1).logits
            actual = model(ids, logits_to_keep=1).logits
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        atomic_json(output / "original-host-parity.json", {"passed": True, "max_abs": float((actual-expected).abs().max()),
                    "sequence_length": ids.shape[1], "precision": "fp32", "loading": loading})
        del original, actual, expected
    ensure_manifest(output / "checkpoint.json", provenance)
    return model, tokenizer, provenance


def complete_old_validation(arm, train_job, output, allocation, device):
    model, tokenizer, provenance = load_checkpoint(arm, train_job, output, device)
    try:
        old = Path(f"output/retrieval-exposure-20261121-restart/audit-{arm}/{arm}/val")
        old_rows = load_rows(Path(f"data/retrieval_exposure/{arm}/val.jsonl"))
        predictions = read_results(old / "predictions.jsonl", "example_id")
        if len(predictions) != 128:
            raise ValueError("Historical validation must contain its complete 128 predictions")
        benchmark = ControlledBenchmark()
        examples = [benchmark.prepare(row, tokenizer) for row in old_rows]
        validate_predictions(old, predictions, examples, True, True)
        backend = QwenBackend(model, tokenizer, provenance,
                              {"do_sample": False, "max_new_tokens": 8, "eos_token_id": tokenizer.eos_token_id})
        backend.validate_saved(old, predictions)
        old_manifest = json.loads((old / "manifest.json").read_text())
        old_identity = old_manifest["backend"]
        for field in ("checkpoint_sha256", "tokenizer_sha256"):
            if old_identity[field] != provenance[field]:
                raise ValueError(f"Historical validation differs in {field}")
        if old_identity["autocast_dtype"] != "bfloat16":
            raise ValueError("Historical precision is not matched BF16")
        rows = validation_rows()
        missing = [r for r in rows if r["example_id"] not in predictions]
        if len(missing) != 384:
            raise ValueError("Historical IDs are not a subset of the full validation set")
        if not generation_audit(model, tokenizer, provenance, missing, output / "missing-val", allocation):
            return False
        predictions.update(read_results(output / "missing-val/predictions.jsonl", "example_id"))
        atomic_json(output / "full-val-selection.json", selection_counts(rows, predictions))
        ensure_manifest(output / "reuse.json", {"historical_directory": str(old), "historical_manifest": old_manifest,
                        "historical_questions": 128, "new_questions": 384, "complete_questions": 512})
        return True
    finally:
        del model
        release()


def run_baselines(output, allocation, device):
    for name in ("u_full", "u_sw128"):
        if allocation.stopped():
            return False
        model, tokenizer, provenance = load_baseline(name, device, output / name)
        try:
            if not audit_model(model, tokenizer, provenance, output / name, allocation, reads_off=False):
                return False
        finally:
            del model
            release()
    for arm, job in (("repeat", "12045460"), ("broad", "12045462")):
        if allocation.stopped() or not complete_old_validation(arm, job, output / f"old-{arm}", allocation, device):
            return False
    return True
