"""Post-training normalization interventions on an archived unstable D trajectory."""

import argparse
import copy
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
from evaluation.taal.longmemeval import create_trace_exporter, generate_taal_example
from evaluation.taal.longmemeval_decoder_replay import compare_reference
from evaluation.taal.longmemeval_numerics import write_report
from evaluation.taal.numerical_diagnostic import BlockMonitor, NumericalFailure
from evaluation.taal.prefix_store import checkpoint_fingerprint
from evaluation.taal.runaway_diagnostic import RunawayRecorder
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder, make_trace_tokens
from utils.seed import seed_everything


ARMS = {"baseline": (False, False), "input": (True, False),
        "mlp": (False, True), "both": (True, True)}


def ablation_config(config, arm):
    if arm not in ARMS:
        raise ValueError("Unknown normalization arm")
    if config.memory_pre_rmsnorm or config.memory_mlp_output_layernorm or config.memory_chunk_size != 1:
        raise ValueError("Expected the unnormalized C1 source checkpoint")

    inputs, mlp = ARMS[arm]
    return TaalQwen3Config.from_dict(config.to_dict() | {
        "memory_pre_rmsnorm": inputs, "memory_mlp_output_layernorm": mlp,
        "memory_normalization_eps": 1e-6,
    })


def run_fixed(model, tokenizer, source, manifest, directory):
    directory.mkdir()
    S = source.example["prompt_length"]
    tokens = source.example["input_ids"] + source.continuation
    layers = list(range(model.config.num_hidden_layers))
    monitor = BlockMonitor(model, S)
    updates = RunawayRecorder(monitor, directory / "updates", tail_size=1)
    decoder = DecoderRecorder(model, directory / "decoder", manifest["layers"], manifest["trace_start"])
    trace = TaalTraceRecorder(model, layers, len(tokens))
    exporter = TaalTraceExporter(directory / "memory_traces", run_metadata=manifest | {
        "layers": layers, "evaluation": "longmemeval_normalization_fixed",
    })
    report = {"status": "started", "trace_status": "incomplete", "phase": "fixed_tokens"}
    started = time.perf_counter()
    write_report(directory / "report.json", report)
    device = next(model.parameters()).device
    prompt = torch.tensor([source.example["input_ids"]], dtype=torch.long, device=device)
    try:
        with monitor, updates, decoder, trace:
            output, predictions = replay_fixed_tokens(model, prompt, source.continuation,
                                                      manifest["execution_block_size"], 1.0)
        expected = {(layer, position) for layer in manifest["layers"]
                    for position in range(manifest["trace_start"], len(tokens))}
        actual = {(row["layer"], row["position"]) for row in decoder.rows}
        if actual != expected or len(decoder.rows) != len(expected) or output.state.tokens_seen != len(tokens):
            raise ValueError("Incomplete or duplicate fixed-token coverage")

        report.update(status="finite_replay", predictions=predictions, tokens_seen=output.state.tokens_seen)
        try:
            exporter.export(TraceEpisode(
                run_id=directory.parent.name, example_id=source.example["question_id"],
                condition_id=manifest["arm"], checkpoint=manifest["checkpoint"],
                tokenizer_id=manifest["tokenizer"],
                tokens=make_trace_tokens(source.example["input_ids"], tokenizer)
                    + make_trace_tokens(source.continuation, tokenizer, offset=S, phase="generated"),
                writes=trace.writes, reads=trace.reads, internal_prefixes=trace.internal_prefixes,
                outcome={"forced_token_ids": source.continuation, "forced_text": tokenizer.decode(source.continuation)},
                metadata={"forced_continuation": True, "normalization": manifest["normalization"],
                          "expected_answer": source.example["answer"],
                          "interpretation": "Forced inputs, not this arm's generated answer."},
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
        report.update(seconds=time.perf_counter() - started, decoder_rows=len(decoder.rows))
        write_report(directory / "report.json", report)

    if manifest["arm"] == "baseline":
        rows = [row for row in decoder.rows if row["position"] < S + min(96, len(source.continuation))]
        expected_count = len(manifest["layers"]) * (S + min(96, len(source.continuation)) - manifest["trace_start"])
        if len(rows) != expected_count:
            report["reference_comparison"] = {"status": "incomplete", "rows": len(rows), "expected": expected_count}
        else:
            report["reference_comparison"] = compare_reference(rows, source.reference)
        write_report(directory / "report.json", report)
    return report


def run_generation(model, tokenizer, source, manifest, directory):
    directory.mkdir()
    settings = copy.deepcopy(source.manifest["settings"])
    settings["save_traces"] = True
    settings["model"]["memory_read_scale"] = 1.0
    settings["generation"]["max_new_tokens"] = manifest["max_new_tokens"]
    settings["generation"]["do_sample"] = False
    settings["generation"]["execution_block_size"] = manifest["execution_block_size"]
    exporter = create_trace_exporter(directory, manifest["checkpoint"], model.config,
                                     source.manifest["metadata"], settings, {})
    report = {"status": "started", "trace_status": "incomplete", "phase": "free_generation"}
    started = time.perf_counter()
    write_report(directory / "report.json", report)
    monitor = BlockMonitor(model, source.example["prompt_length"])
    try:
        with monitor:
            outcome = generate_taal_example(model, tokenizer, source.example, settings,
                                            next(model.parameters()).device, exporter)
        report.update(status="finite_generation", trace_status="exported", outcome=outcome)
    except NumericalFailure as error:
        report.update(status="nonfinite_detected", first_bad_boundary=error.report)
    except Exception as error:
        report.update(status="diagnostic_error", error=repr(error))
        raise
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(directory / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--decode-tokens", type=int, default=169)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-checkpoint normalization ablations expect CUDA")
    if args.output_dir.exists() or args.max_new_tokens < 1:
        raise ValueError("Use a fresh output directory and a positive generation budget")

    config = TaalQwen3Config.from_pretrained(args.checkpoint)
    layers = [layer for layer in (2, 3, 8, 12, 24, 27) if layer < config.num_hidden_layers]
    source = load_replay_source(args.source_run, args.decode_tokens, layers)
    fingerprint = checkpoint_fingerprint(args.checkpoint)
    if fingerprint != source.manifest["fingerprint"]:
        raise ValueError("Checkpoint differs from archived failing run")
    if source.manifest["settings"]["model"].get("retain_task_prefix", False):
        raise ValueError("Expected the original unpinned instruction prefix")

    config = ablation_config(config, args.arm)
    metadata = source.manifest["metadata"]
    settings = source.manifest["settings"]
    S = source.example["prompt_length"]
    tokens = source.example["input_ids"] + source.continuation
    args.output_dir.mkdir(parents=True)
    manifest = {
        "arm": args.arm, "checkpoint": str(args.checkpoint.resolve()), "fingerprint": fingerprint,
        "source_run": str(args.source_run.resolve()), "source_commit": source.manifest["commit"],
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "job_id": os.environ.get("SLURM_JOB_ID"), "model_config": config.to_dict(),
        "normalization": {"input_rmsnorm": config.memory_pre_rmsnorm,
                          "mlp_output_layernorm": config.memory_mlp_output_layernorm,
                          "eps": config.memory_normalization_eps, "affine": False, "extra_residual": False},
        "prompt_length": S, "forced_token_ids": source.continuation,
        "token_digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        "memory_read_scale": 1.0, "layers": layers,
        "trace_start": max(0, S - config.working_memory_size),
        "execution_block_size": settings["generation"]["execution_block_size"],
        "max_new_tokens": args.max_new_tokens, "seed": settings["seed"], "save_traces": True,
        "tokenizer": metadata["tokenizer"], "tokenizer_revision": metadata["tokenizer_revision"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "interpretation": "Post-training stability intervention, not matched retraining or a benchmark score. Fresh states per phase.",
    }
    write_report(args.output_dir / "manifest.json", manifest)
    write_report(args.output_dir / "example.json", source.example)
    tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"], revision=metadata["tokenizer_revision"])
    seed_everything(settings["seed"])
    model = load_model(str(args.checkpoint), config).to("cuda").eval()
    manifest["attention_implementation"] = model.config._attn_implementation
    write_report(args.output_dir / "manifest.json", manifest)
    reports = {}
    for phase, runner in (("fixed", run_fixed), ("generation", run_generation)):
        print(f"NORMALIZATION_START arm={args.arm} phase={phase}", flush=True)
        reports[phase] = runner(model, tokenizer, source, manifest, args.output_dir / phase)
        write_report(args.output_dir / "summary.json", reports)
        print(f"NORMALIZATION_DONE arm={args.arm} phase={phase} status={reports[phase]['status']}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
