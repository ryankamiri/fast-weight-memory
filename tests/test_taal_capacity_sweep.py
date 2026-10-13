from collections import Counter
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import AutoTokenizer
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from data.dataloader import create_dataloader
from data.taal_conflict_control import MODEL_ID, MODEL_REVISION, build_examples
from data.taal_min_capacity import build_capacity_examples
from evaluation.taal import state_audit
from evaluation.taal.interventions import audit_conditions
from evaluation.taal.training_trace import export_training_traces
from tests import test_taal_state_audit
from training.config import load_config
from training.engine import build_scheduler, train
from training.train import configure_trainable_parameters


CASES = (
    ("b_distance", 1, 192),
    ("c_diversity", 8, 64),
    ("d_distance_diversity", 8, 192),
)


class TaalCapacitySweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(__file__).resolve().parents[1]
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        )
        cls.examples = {
            name: build_capacity_examples(cls.tokenizer, pairs=pairs, gap=gap)
            for name, pairs, gap in CASES
        }

    def test_distance_diversity_and_pairing_are_the_only_dataset_changes(self):
        reference = build_capacity_examples(self.tokenizer)
        for name, pairs, gap in CASES:
            with self.subTest(case=name):
                examples = self.examples[name]
                count = 2 * pairs
                self.assertEqual(len(examples), count)
                self.assertEqual(len({row["example_id"] for row in examples}), count)
                self.assertEqual(len({row["target_token_id"] for row in examples}), count)
                self.assertEqual({len(row["input_ids"]) for row in examples}, {gap + 14})
                donors = state_audit.swap_sources(examples, "record_name")
                for row in examples:
                    self.assertEqual(len(row["candidate_token_ids"]), count)
                    self.assertEqual(len(row["input_ids"]) - row["fact_position"], gap)
                    self.assertGreater(row["final_query_position"] - row["fact_position"], 32)
                    donor = donors[row["example_id"]]
                    self.assertEqual(row["record_name"], donor["record_name"])
                    self.assertNotEqual(row["target_token_id"], donor["target_token_id"])
                    differences = [i for i, (a, b) in enumerate(zip(row["input_ids"], donor["input_ids"])) if a != b]
                    self.assertEqual(differences, [row["fact_position"]])
                for before, after in zip(reference, examples[:2]):
                    self.assertEqual(before["answer"], after["answer"])
                    self.assertEqual(before["input_ids"][:21], after["input_ids"][:21])
                    self.assertEqual(before["input_ids"][before["final_query_position"]:],
                                     after["input_ids"][after["final_query_position"]:])
        self.assertEqual([row["input_ids"] for row in reference],
                         [row["input_ids"] for row in self.examples["c_diversity"][:2]])
        self.assertEqual([row["input_ids"] for row in self.examples["b_distance"]],
                         [row["input_ids"] for row in self.examples["d_distance_diversity"][:2]])
        original = build_examples(self.tokenizer, pairs=8, window=128, gap=192)
        self.assertEqual(self.examples["d_distance_diversity"], original)
        with self.assertRaisesRegex(ValueError, "positive"):
            build_capacity_examples(self.tokenizer, pairs=0)

    def test_recipes_preserve_successful_model_and_optimization(self):
        reference = load_config(str(self.root / "training/configs/taal/qwen3_0_6b_capacity_paired.yaml"))
        for name, pairs, gap in CASES:
            with self.subTest(case=name):
                config = load_config(str(self.root / f"training/configs/taal/qwen3_0_6b_capacity_{name}.yaml"))
                expected = reference.to_dict()
                actual = config.to_dict()
                for section in ("model", "optimizer", "scheduler", "loss", "training", "validation", "checkpoints"):
                    self.assertEqual(actual[section], expected[section])
                self.assertEqual(config.data.train.end, 2 * pairs)
                self.assertEqual(config.data.val.end, 2 * pairs)
                self.assertEqual(config.data.batch_size, 1)
                self.assertTrue(config.data.query_fresh_writes_enabled)
                self.assertEqual(config.data.seq_len, 96 if gap == 64 else 224)
                self.assertTrue(all(len(row["input_ids"]) + 1 <= config.data.seq_len
                                    for row in self.examples[name]))
                evaluation = yaml.safe_load((self.root / f"evaluation/configs/taal/capacity_{name}_audit.yaml").read_text())
                self.assertEqual(evaluation["dataset"]["path"], config.data.dataset_id)
                self.assertEqual(evaluation["dataset"]["end"], config.data.val.end)
                self.assertEqual(evaluation["execution_block_size"], 32)
                self.assertEqual(evaluation["swap_group_field"], "record_name")
                self.assertTrue(evaluation["save_traces"])
                self.assertEqual(len(audit_conditions(evaluation.get("audit_suite", "standard"))), 6)

    def test_cpu_training_eviction_all_selected_traces_and_all_six_audit_modes(self):
        # A real tiny TaaL model with the real pinned tokenizer exercises the
        # training/audit paths. Only the CLI's CUDA guard/device and model load
        # are replaced. This is not a Qwen/H200 convergence benchmark.
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp,
        )
        for name, pairs, gap in CASES:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                examples = self.examples[name]
                dataset = root / "data.jsonl"
                dataset.write_text("".join(json.dumps(row) + "\n" for row in examples))
                dataset.with_suffix(".metadata.json").write_text(json.dumps({
                    "tokenizer": MODEL_ID, "tokenizer_revision": MODEL_REVISION,
                }))
                config = load_config(str(self.root / f"training/configs/taal/qwen3_0_6b_capacity_{name}.yaml"))
                config.data.dataset_id = str(dataset)
                config.training.max_steps = 1
                config.training.eval_every_steps = 1
                torch.manual_seed(42)
                model_config = test_taal_state_audit.TaalStateAuditTests.config()
                model_config.vocab_size = len(self.tokenizer)
                model_config.working_memory_size = 32
                model = TaalQwen3ForCausalLM(model_config).train()
                with torch.no_grad():
                    model.model.layers[0].taal.residual_gate.fill_(0.05)
                parameters = configure_trainable_parameters(model, "taal_only")
                optimizer = torch.optim.AdamW(parameters, lr=config.optimizer.lr)
                scheduler = build_scheduler(optimizer, config)
                train_loader = create_dataloader(config.data, config.data.train, config.loss, shuffle=True, seed=42)
                val_loader = create_dataloader(config.data, config.data.val, config.loss, shuffle=False, seed=42)
                metrics = []
                progress = train(model, train_loader, val_loader, optimizer, scheduler, config,
                                 torch.device("cpu"), metrics.append, lambda: False)
                self.assertEqual(progress.step, 1)
                self.assertEqual(progress.skipped_updates, 0)
                self.assertEqual(metrics[-1]["val/records"], 2 * pairs)
                self.assertIn("val/loss_memory_read_scale_0", metrics[-1])
                export_training_traces(model, config, root / "training", progress)
                training_manifest = json.loads((root / "training/memory_traces/manifest.json").read_text())
                self.assertEqual(len(training_manifest["episodes"]), 2 * pairs)
                with torch.inference_mode():
                    model.eval()
                    for row in examples:
                        prefix, _ = state_audit.split_episode(row)
                        state = model.prefill(torch.tensor([prefix]), execution_block_size=32).state
                        # Check raw source-span eviction, not just gap metadata.
                        retained = state.past_key_values.layers[0].positions.tolist()
                        self.assertTrue(set(retained).isdisjoint(range(21)))
                evaluation = yaml.safe_load((self.root / f"evaluation/configs/taal/capacity_{name}_audit.yaml").read_text())
                evaluation["dataset"]["path"] = str(dataset)
                audit_config = root / "audit.yaml"
                audit_config.write_text(yaml.safe_dump(evaluation))
                output = root / "audit"
                argv = ["audit", "--config", str(audit_config), "--checkpoint", str(root), "--output-dir", str(output)]
                with (
                    patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                    patch.object(state_audit, "load_dataset", return_value=examples),
                    patch.object(state_audit, "checkpoint_fingerprint", return_value="CPU-fixture"),
                    patch.object(state_audit, "load_model", return_value=model),
                    patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=self.tokenizer),
                    redirect_stdout(io.StringIO()),
                ):
                    state_audit.main()
                    state_audit.main()  # Completed audit resumes without duplicate results.
                rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                self.assertEqual(len({row["evaluation_id"] for row in rows}), 12 * pairs)
                self.assertEqual(Counter(row["audit_condition"] for row in rows),
                                 {condition.name: 2 * pairs for condition in audit_conditions("standard")})
                manifest = json.loads((output / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), 12 * pairs)
                self.assertEqual(len(manifest["comparisons"]), 10 * pairs)
                for key, relative in manifest["episodes"].items():
                    with gzip.open(output / "memory_traces" / relative, "rt") as handle:
                        episode = json.load(handle)
                    self.assertEqual(len(episode["tokens"]), gap + 15)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                    self.assertEqual(episode["tokens"][-1]["token_id"], episode["outcome"]["vocabulary_top_token_id"])
                    self.assertEqual(len(episode["reads"]), len(episode["tokens"]))
                    self.assertEqual(len(episode["writes"]), len(episode["tokens"]) + model_config.num_persistent_tokens)
                    if key.endswith("/swapped"):
                        own_id = episode["example_id"]
                        donor_id = episode["outcome"]["swapped_from_example_id"]
                        self.assertEqual(state_audit.swap_sources(examples, "record_name")[own_id]["example_id"], donor_id)
                    json.dumps(episode, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
