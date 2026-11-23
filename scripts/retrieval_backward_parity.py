"""Read-only gradient and optimizer-step comparisons on fixed complete families."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from evaluation.storage import atomic_json, ensure_manifest, file_sha256
from evaluation.taal.execution_parity import tensor_difference
from scripts.retrieval_execution_parity import completed_case, diagnostic_rows, run_case
from scripts.retrieval_exposure_audit import checked_rows, generation_audit, load_checkpoint
from scripts.retrieval_parallel_audit import release
from utils.precision import execution_precision


def selected_families():
    rows = checked_rows("common-fit")
    families = list(dict.fromkeys(row["metadata"]["family_id"] for row in rows))[:2]
    selected = [row for row in rows if row["metadata"]["family_id"] in families]
    lengths = {len(row["input_ids"]) - len(row["prompt_input_ids"]) - 1 for row in selected}
    if len(selected) != 16 or lengths != {1, 2}:
        raise ValueError("Diagnostic needs two complete families covering one/two-token labels")
    return selected


def answer_forward(model, row, blocked):
    device = next(model.parameters()).device
    ids = torch.tensor([row["input_ids"]], device=device)
    P = len(row["prompt_input_ids"])
    targets = ids[0, P:]
    if not blocked:
        output = model(ids, logits_to_keep=len(targets) + 1, use_cache=False)
        return output.logits[0, :-1], targets
    # Same semantic segment, cached blocks, and teacher-forced answer tokens.
    # Autograd stays enabled here; public prefill intentionally uses inference_mode.
    block = model.config.working_memory_size
    spans = [(start, min(start + block, P)) for start in range(0, P, block)]
    spans += [(position, position + 1) for position in range(P, ids.shape[1])]
    state, logits = None, []
    for start, end in spans:
        output = model.model(ids[:, start:end], state=state, use_cache=True,
                             prepend_memory_tokens=start == 0)
        state = output.state
        left, right = max(start, P - 1), min(end, ids.shape[1] - 1)
        if left < right:
            logits.append(model.lm_head(output.last_hidden_state[0, left-start:right-start]))
    return torch.cat(logits), targets


def gradient_group(name):
    layer, branch = name.split(".taal.")
    if "memory_mlp" in branch:
        group = "initial_memory"
    elif "projection" in branch:
        group = branch.rsplit(".", 1)[0]
    else:
        group = branch
    return layer + "." + group


def optimizer_trial(parameters, gradients, optimizer_state=None):
    # Independent CPU copies: never step the checkpoint being diagnosed.
    copies = {name: torch.nn.Parameter(value.detach().cpu().clone()) for name, value in parameters.items()}
    for name, parameter in copies.items():
        parameter.grad = gradients[name].clone()
    norm = torch.nn.utils.clip_grad_norm_(list(copies.values()), 1.0, error_if_nonfinite=True)
    clipped = {name: value.grad.clone() for name, value in copies.items()}
    optimizer = torch.optim.AdamW(copies.values(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if optimizer_state is not None:
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    before = {name: value.detach().clone() for name, value in copies.items()}
    optimizer.step()
    delta = {name: value.detach() - before[name] for name, value in copies.items()}
    return clipped, delta, float(norm)


def compare_gradients(left, right):
    grouped = {}
    for name in left:
        grouped.setdefault(gradient_group(name), []).append(name)
    result = {}
    for group, names in grouped.items():
        a = torch.cat([left[name].flatten() for name in names])
        b = torch.cat([right[name].flatten() for name in names])
        norm_a, norm_b = float(a.norm()), float(b.norm())
        cosine = float(F.cosine_similarity(a, b, dim=0)) if norm_a and norm_b else None
        result[group] = tensor_difference(a, b) | {"reference_norm": norm_a, "candidate_norm": norm_b, "cosine": cosine}
    return result


@contextmanager
def capture_writes(model, row):
    originals, samples = [], []
    P = len(row["prompt_input_ids"])
    for index in sorted({0, len(model.model.layers) - 1}):
        memory = model.model.layers[index].taal.neural_memory
        original = memory._chunk_gradient
        count = [0]
        positions = {model.config.num_persistent_tokens, model.config.num_persistent_tokens + P - 1}

        def observe(weights, keys, values, strength, trace_tokens=False,
                    original=original, index=index, count=count, positions=positions):
            result = original(weights, keys, values, strength, trace_tokens=trace_tokens)
            if count[0] in positions:
                samples.append({"layer": index, "internal_position": count[0],
                    "text_position": count[0] - model.config.num_persistent_tokens,
                    "weights": {n: v.detach().cpu().clone() for n, v in weights.items()},
                    "keys": keys.detach().cpu().clone(), "values": values.detach().cpu().clone(),
                    "strength": strength.detach().cpu().clone(),
                    "manual": {n: v.detach().cpu().clone() for n, v in result.chunk_gradients.items()}})
            count[0] += keys.shape[1]
            return result

        originals.append((memory, original))
        memory._chunk_gradient = observe
    try:
        yield samples
    finally:
        for memory, original in originals:
            del memory._chunk_gradient


def derivative_reference(model, samples):
    results = []
    device = next(model.parameters()).device
    for sample in samples:
        memory = model.model.layers[sample["layer"]].taal.neural_memory
        weights = {name: value.to(device).requires_grad_() for name, value in sample["weights"].items()}
        keys, values, strength = (sample[name].to(device) for name in ("keys", "values", "strength"))
        with torch.enable_grad(), torch.autocast(device.type, enabled=False):
            prediction = memory.memory_mlp(keys, weights=weights).predicted_values
            loss = ((prediction-values).square().mean(-1) * strength).sum()
            gradients = torch.autograd.grad(loss, tuple(weights.values()))
        results.append({"layer": sample["layer"], "text_position": sample["text_position"],
            "shape": list(keys.shape), "parameters": {
                name: tensor_difference(sample["manual"][name], gradient.detach().cpu())
                for name, gradient in zip(weights, gradients)}})
    return results


def backward_case(model, row, precision, output, optimizer_state=None):
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    before = {name: value.detach().cpu().clone() for name, value in parameters.items()}
    device = next(model.parameters()).device
    model.train()
    paths, measurements = {}, {}
    for path in ("full", "blocked"):
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with execution_precision(device, precision) as settings, capture_writes(model, row) as samples:
            logits, labels = answer_forward(model, row, path == "blocked")
            losses = F.cross_entropy(logits.float(), labels, reduction="none")
            if not bool(torch.isfinite(losses).all()):
                raise ValueError("Non-finite backward diagnostic loss")
            terms = {"label": losses[:-1].sum() / len(labels), "eos": losses[-1] / len(labels), "combined": losses.mean()}
            gradients = {}
            for term, loss in terms.items():
                values = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=term != "combined", allow_unused=True)
                raw = {name: torch.zeros_like(parameter, device="cpu") if value is None else value.detach().float().cpu()
                       for (name, parameter), value in zip(parameters.items(), values)}
                clipped, delta, norm = optimizer_trial(parameters, raw, optimizer_state)
                gradients[term] = {"raw": raw, "clipped": clipped, "step_delta": delta, "global_norm": norm}
            paths[path] = gradients
            report = {"settings": settings, "loss": {name: float(value.detach()) for name, value in terms.items()},
                      "gradient_norms": {name: values["global_norm"] for name, values in gradients.items()},
                      "manual_derivative": derivative_reference(model, samples)}
            torch.save(samples, output / f"{path}-write-inputs.pt")
        if device.type == "cuda":
            torch.cuda.synchronize()
        report.update(seconds=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else None)
        measurements[path] = report
        del logits, losses, terms, values, samples
    comparisons = {term: {stage: compare_gradients(paths["full"][term][stage], paths["blocked"][term][stage])
                         for stage in ("raw", "clipped", "step_delta")} for term in ("label", "eos", "combined")}
    for name, parameter in parameters.items():
        if not torch.equal(parameter.detach().cpu(), before[name]) or parameter.grad is not None:
            raise ValueError("Read-only diagnostic changed the checkpoint")
    atomic_json(output / "backward.json", {"paths": measurements, "comparisons": comparisons,
                "objective": "unweighted answer+EOS; component gradients share total-token denominator",
                "optimizer_trial": "saved AdamW state, outer clip 1; independent CPU copies" if optimizer_state is not None else "fresh AdamW 3e-4, outer clip 1; independent CPU copies",
                "artifacts": {f"{name}-write-inputs.pt": file_sha256(output / f"{name}-write-inputs.pt") for name in ("full", "blocked")}})
    return paths


def run_diagnostics(output, allocation, device):
    rows = selected_families()
    ensure_manifest(output / "selected-families.json", rows)
    for arm, job in (("C", None), ("repeat", "12045460")):
        if allocation.stopped():
            return False
        model, tokenizer, provenance = load_checkpoint(arm, job, output / arm, device)
        try:
            state = torch.load(Path(provenance["checkpoint"]) / "trainer_state.pt", map_location="cpu", weights_only=False)
            parameter_names = [name for name, value in model.named_parameters() if value.requires_grad]
            if parameter_names != state["trainable_parameters"]:
                raise ValueError("Optimizer parameter order differs from checkpoint")
            optimizer_state = state["optimizer"]
            del state
            for row in rows + diagnostic_rows("C"):
                cross_path = output / arm / row["example_id"] / "cross-precision-backward.json"
                gradients_by_precision = {}
                needs_backward = row in rows and not cross_path.exists()
                for precision in ("bf16_auto", "fp32_auto"):
                    root = output / arm / row["example_id"] / precision
                    if allocation.stopped():
                        return False
                    root.mkdir(parents=True, exist_ok=True)
                    if not completed_case(root / "forward"):
                        run_case(model, tokenizer, row, precision, 1.0, root / "forward")
                    if needs_backward:
                        # Retain only this example's CPU gradients. No multi-GB
                        # gradient dump is needed; an interrupted pair recomputes.
                        gradients_by_precision[precision] = backward_case(model, row, precision, root, optimizer_state)
                    elif row in rows:
                        report = json.loads((root / "backward.json").read_text())
                        for name, checksum in report["artifacts"].items():
                            if file_sha256(root / name) != checksum:
                                raise ValueError("Backward diagnostic artifact changed")
                if needs_backward:
                    comparison = {
                        path: {term: {stage: compare_gradients(
                            gradients_by_precision["fp32_auto"][path][term][stage],
                            gradients_by_precision["bf16_auto"][path][term][stage])
                            for stage in ("raw", "clipped", "step_delta")}
                            for term in ("label", "eos", "combined")}
                        for path in ("full", "blocked")
                    }
                    atomic_json(cross_path, {"reference": "fp32_auto", "candidate": "bf16_auto", "comparisons": comparison})
                del gradients_by_precision
            if not generation_audit(model, tokenizer, provenance, rows, output / arm / "generation", allocation):
                return False
        finally:
            del model
            release()
    return True
