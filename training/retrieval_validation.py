"""Fixed family-complete train/val probes using the shared generation evaluator."""

import hashlib
import json
from pathlib import Path

import torch

from data.dataloader import CausalLMCollator
from data.retrieval import load_retrieval_splits
from evaluation.shared.backends import QwenBackend
from evaluation.shared.benchmarks import ControlledBenchmark
from evaluation.shared.runner import run
from evaluation.storage import file_sha256
from training.engine import precision_context, validate
from training.resume import restore_rng, rng_state


def select_probe_families(rows, count):
    families = []
    for row in rows:
        family = row["metadata"]["family_id"]
        if family not in families:
            families.append(family)
    if count > len(families):
        raise ValueError("Probe family budget exceeds available families")

    selected = set(families[:count])
    return [row for row in rows if row["metadata"]["family_id"] in selected]


def trainable_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class RetrievalValidation:
    def __init__(self, config, tokenizer, directory):
        self.config = config
        self.tokenizer = tokenizer
        self.directory = Path(directory)
        splits, manifest = load_retrieval_splits(config.data)
        if json.loads(tokenizer.backend_tokenizer.to_str()) != manifest["tokenizer_backend"]:
            raise ValueError("Training tokenizer differs from dataset tokenizer")

        if tokenizer.special_tokens_map != manifest["special_tokens"]:
            raise ValueError("Training special tokens differ from dataset tokenizer")

        if config.model.working_memory_size != manifest["config"]["kv_window"] or config.model.max_persistent_kv_tokens != 0:
            raise ValueError("Retrieval recipe must preserve the verified recent-KV budget without pinned prefix slots")

        self.rows = {}
        self.examples = {}
        self.benchmark = ControlledBenchmark()
        for split, rows in splits.items():
            for row in rows:
                ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                answer = tokenizer.encode(" " + row["answer"], add_special_tokens=False)
                if ids != row["prompt_input_ids"] or row["input_ids"] != ids + answer + [tokenizer.eos_token_id]:
                    raise ValueError("Dataset tokens differ from prompt/answer rendering")

            selected = select_probe_families(rows, config.validation.probe_families)
            self.rows[split] = selected
            self.examples[split] = [self.benchmark.prepare(row, tokenizer) for row in selected]
        self.identity = {"manifest_sha256": file_sha256(Path(config.data.manifest_path))}

    def __call__(self, model, progress, should_stop):
        before_rng = rng_state()
        was_training = model.training
        device = next(model.parameters()).device
        metrics = {}
        try:
            fingerprint = trainable_fingerprint(model)
            for split, rows in self.rows.items():
                batches = [CausalLMCollator()([row]) for row in rows]
                losses = validate(model, batches, device, should_stop, [1.0], self.config.loss)
                if losses is None:
                    return None

                metrics[f"{split}/probe_answer_nll"] = losses["val/loss"]
                metrics[f"{split}/probe_answer_perplexity"] = losses["val/perplexity"]
                if split == "val":
                    metrics["val/loss"] = losses["val/loss"]
                    metrics["val/records"] = len(rows)
                output = self.directory / "probes" / f"step-{progress.step}" / split
                backend = QwenBackend(
                    model, self.tokenizer,
                    {
                        "checkpoint": f"in-memory:{self.directory}:step-{progress.step}",
                        "tokenizer": self.config.model.model_id,
                        "tokenizer_revision": self.config.model.revision,
                        "trainable_sha256": fingerprint,
                        "precision": "FP32 parameters; BF16 CUDA autocast",
                    },
                    {
                        "do_sample": False,
                        "max_new_tokens": self.config.validation.probe_max_new_tokens,
                        "eos_token_id": self.tokenizer.eos_token_id,
                    },
                )
                with precision_context(device):
                    summary = run(
                        backend, self.benchmark, self.examples[split], output,
                        self.identity | {"split": split}, self.config.training.seed,
                        self.config.training.save_traces, should_stop=should_stop,
                    )
                if summary is None:
                    return None

                metrics[f"{split}/probe_greedy_accuracy"] = summary["metrics"]["exact_match"]
                metrics[f"{split}/probe_group_all_correct"] = summary["group_all_correct"]
                metrics[f"{split}/probe_questions"] = summary["examples"]
                metrics[f"{split}/probe_prefixes"] = summary["groups"]
            return metrics
        finally:
            model.train(was_training)
            restore_rng(before_rng)
