"""Replay an archived continuation with reads on/off and observe Qwen sublayers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoTokenizer

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.run import load_model
from evaluation.taal.decoder_diagnostic import DecoderRecorder, load_replay_source, replay_fixed_tokens
from evaluation.taal.longmemeval_numerics import write_report
from evaluation.taal.numerical_diagnostic import BlockMonitor, NumericalFailure
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder, make_trace_tokens
from utils.seed import seed_everything


def compare_reference(rows, reference):
    comparisons = []
    for row in rows:
        expected = reference.get((row["layer"], row["position"]))
        if expected is None:
            raise ValueError("No archived reference for a replayed token/layer")

        for component, metric in (("memory_input", "hidden_in"), ("memory_output", "hidden_out")):
            observed = row["components"][component]["norm_finite"]
            value = expected[metric]
            if value is None or observed is None:
                raise ValueError("Reference comparison requires finite captured norms")

            comparisons.append({
                "layer": row["layer"], "position": row["position"], "component": component,
                "reference": value, "observed": observed,
                "relative_error": abs(observed - value) / max(abs(value), 1e-30),
            })
    if not comparisons:
        raise ValueError("Reference comparison is empty")

    worst = max(comparisons, key=lambda row: row["relative_error"])
    return {"comparisons": len(comparisons), "worst": worst,
            "matched_within_1e-5": worst["relative_error"] <= 1e-5}


def run_replay_arm(model, tokenizer, source, manifest, arm, scale):
    arm.mkdir()
    prompt_length = source.example["prompt_length"]
    tokens = source.example["input_ids"] + source.continuation
    metadata = source.manifest["metadata"]
    checkpoint = manifest["checkpoint"]
    config = model.config
    exporter = TaalTraceExporter(arm / "memory_traces", run_metadata={
        **manifest, "evaluation": "longmemeval_fixed_token_diagnostic", "memory_read_scale": scale,
        "tokenizer": metadata["tokenizer"], "tokenizer_revision": metadata["tokenizer_revision"],
        "layers": list(range(config.num_hidden_layers)),
    })
    trace = TaalTraceRecorder(model, list(range(config.num_hidden_layers)), len(tokens))
    recorder = DecoderRecorder(model, arm / "decoder", manifest["layers"], manifest["trace_start"],
                               manifest["snapshot_positions"])
    monitor = BlockMonitor(model, prompt_length)
    report = {"status": "started", "memory_read_scale": scale, "forced_token_count": len(source.continuation)}
    started = time.perf_counter()
    write_report(arm / "report.json", report)
    print(f"DECODER_REPLAY scale={scale} prompt={prompt_length} forced={len(source.continuation)}", flush=True)
    device = next(model.parameters()).device
    prompt = torch.tensor([source.example["input_ids"]], dtype=torch.long, device=device)
    try:
        with recorder, monitor, trace:
            output, predictions = replay_fixed_tokens(model, prompt, source.continuation,
                                                      manifest["execution_block_size"], scale)
        expected = {(layer, position) for layer in manifest["layers"]
                    for position in range(manifest["trace_start"], len(tokens))}
        actual = {(row["layer"], row["position"]) for row in recorder.rows}
        if actual != expected or len(recorder.rows) != len(expected) or output.state.tokens_seen != len(tokens):
            raise ValueError("Incomplete or duplicate decoder replay coverage")

        report.update(status="finite_replay", tokens_seen=output.state.tokens_seen, predictions=predictions)
        report["decoder_rows"] = len(recorder.rows)
        if scale == 1.0:
            report["reference_comparison"] = compare_reference(recorder.rows, source.reference)
            if not report["reference_comparison"]["matched_within_1e-5"]:
                report["status"] = "reference_mismatch"
        outcome = {
            "forced_token_ids": source.continuation,
            "forced_text": tokenizer.decode(source.continuation),
            "interpretation": "These tokens were forced, not chosen by this arm. No answer score.",
        }
        try:
            exporter.export(TraceEpisode(
                run_id=arm.parent.name, example_id=source.example["question_id"],
                condition_id=f"forced_reads_{scale:g}", checkpoint=checkpoint,
                tokenizer_id=metadata["tokenizer"],
                tokens=make_trace_tokens(source.example["input_ids"], tokenizer)
                    + make_trace_tokens(source.continuation, tokenizer, offset=prompt_length, phase="generated"),
                writes=trace.writes, reads=trace.reads, internal_prefixes=trace.internal_prefixes,
                outcome=outcome, metadata={"forced_continuation": True, "memory_read_scale": scale,
                                           "read_scale_scope": "whole_episode", "source_run": manifest["source_run"]},
            ))
            report["trace_status"] = "exported"
        except ValueError as error:
            report.update(trace_status="invalid_export", trace_error=repr(error))
    except NumericalFailure as error:
        report.update(status="nonfinite_detected", first_bad_boundary=error.report)
    except Exception as error:
        report.update(status="diagnostic_error", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(arm / "report.json", report)
    print(f"DECODER_REPLAY_DONE scale={scale} status={report['status']}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=96)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--snapshot-offsets", type=int, nargs="+", default=[62, 63, 64])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-checkpoint decoder replay expects CUDA")
    if args.output_dir.exists():
        raise ValueError("Use a fresh output directory for the diagnostic")

    source = load_replay_source(args.source_run, args.decode_tokens, args.layers)
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    if fingerprint != source.manifest["fingerprint"]:
        raise ValueError("Replay checkpoint differs from the source trajectory")
    if source.manifest["settings"]["model"].get("retain_task_prefix", False):
        raise ValueError("This replay expects the original unpinned instruction prefix")
    if any(offset < 0 or offset >= args.decode_tokens for offset in args.snapshot_offsets):
        raise ValueError("Snapshot offsets must be inside the forced continuation")

    config = TaalQwen3Config.from_pretrained(args.checkpoint)
    settings = source.manifest["settings"]
    metadata = source.manifest["metadata"]
    prompt_length = source.example["prompt_length"]
    block_size = settings["generation"]["execution_block_size"]
    args.output_dir.mkdir(parents=True)
    tokens = source.example["input_ids"] + source.continuation
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()), "fingerprint": fingerprint,
        "source_run": str(args.source_run.resolve()), "source_commit": source.manifest["commit"],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "model_config": config.to_dict(),
        "prompt_length": prompt_length, "forced_token_ids": source.continuation,
        "token_digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "read_scales": [1.0, 0.0], "read_scale_scope": "whole_episode",
        "layers": args.layers, "trace_start": max(0, prompt_length - block_size),
        "snapshot_positions": [prompt_length + offset for offset in args.snapshot_offsets],
        "execution_block_size": block_size, "save_traces": True,
        "seed": settings["seed"], "tokenizer": metadata["tokenizer"],
        "tokenizer_revision": metadata["tokenizer_revision"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "interpretation": "Forced-token replay, not free generation or an accuracy evaluation. Fresh KV and memory per arm.",
    }
    write_report(args.output_dir / "manifest.json", manifest)
    write_report(args.output_dir / "example.json", source.example)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
    seed_everything(settings["seed"])
    model = load_model(str(args.checkpoint), config).to("cuda").eval()
    manifest["attention_implementation"] = model.config._attn_implementation
    write_report(args.output_dir / "manifest.json", manifest)
    reports = {}
    for scale in (1.0, 0.0):
        arm = args.output_dir / f"reads-{scale:g}"
        reports[f"reads-{scale:g}"] = run_replay_arm(model, tokenizer, source, manifest, arm, scale)
        torch.cuda.empty_cache()
    write_report(args.output_dir / "summary.json", reports)


if __name__ == "__main__":
    main()
