from collections import Counter
from contextlib import redirect_stdout
import copy
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
from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
from data.taal_d_heldout import (
    D_CHECKPOINT_SHA256, EXPERIMENTS, build_heldout_examples,
    dataset_metadata, dataset_text,
)
from data.taal_min_capacity import build_capacity_examples
from evaluation.taal import state_audit
from evaluation.taal.heldout import summarize_heldout, verify_inputs
from evaluation.taal.interventions import audit_conditions
from tests import test_taal_state_audit


class TaalDHeldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.root = Path(__file__).resolve().parents[1]
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
        )
        cls.base = build_capacity_examples(cls.tokenizer, pairs=8, gap=192)
        cls.datasets = {name: build_heldout_examples(cls.tokenizer, name) for name in EXPERIMENTS}

    def test_selection_disjointness_balance_and_exact_donor_pairs(self):
        original_prompts = {tuple(row["input_ids"]) for row in self.base}
        for name, count in (("bindings", 128), ("context", 80), ("questions", 64)):
            with self.subTest(experiment=name):
                rows = self.datasets[name]
                self.assertEqual(rows, build_heldout_examples(self.tokenizer, name))
                self.assertEqual(len(rows), count)
                self.assertEqual([row["fact_id"] for row in rows], list(range(count)))
                self.assertEqual(len({tuple(row["input_ids"]) for row in rows}), count)
                self.assertEqual(Counter(row["condition"] for row in rows),
                                 {arm: 16 for arm in {row["condition"] for row in rows}})
                donors = state_audit.swap_sources(rows, "pair_group")
                config = yaml.safe_load((self.root / f"evaluation/configs/taal/d_heldout_{name}.yaml").read_text())
                self.assertEqual(len(state_audit.select_examples(rows, config["dataset"])), count)
                self.assertTrue(config["save_traces"])
                self.assertEqual(config["execution_block_size"], 32)
                self.assertEqual(config["swap_group_field"], "pair_group")
                for row in rows:
                    self.assertEqual(row["candidate_token_ids"], self.base[0]["candidate_token_ids"])
                    self.assertEqual(row["input_ids"][row["fact_position"]], row["target_token_id"])
                    self.assertEqual(len(row["input_ids"]) - row["fact_position"], 192)
                    self.assertEqual(len(row["input_ids"]), row["final_answer_position"])
                    self.assertEqual(len(row["input_ids"]) - row["final_query_position"], 20)
                    self.assertGreater(row["final_query_position"] - row["fact_position"] - 7, 32)
                    donor = donors[row["example_id"]]
                    self.assertEqual(row["condition"], donor["condition"])
                    self.assertEqual(row["record_name"], donor["record_name"])
                    self.assertEqual(len(row["input_ids"]), len(donor["input_ids"]))
                    differences = [i for i, (a, b) in enumerate(zip(row["input_ids"], donor["input_ids"])) if a != b]
                    self.assertEqual(differences, [row["fact_position"]])
                    if row["condition"] != "reference":
                        self.assertNotIn(tuple(row["input_ids"]), original_prompts)

    def test_bindings_only_replace_the_label_and_cover_every_unseen_binding(self):
        by_record = {}
        for original in self.base:
            by_record.setdefault(original["record_name"], set()).add(original["target_token_id"])
        heldout = self.datasets["bindings"][16:]
        self.assertEqual(len({(row["record_name"], row["target_token_id"]) for row in heldout}), 112)
        for row in heldout:
            original = next(item for item in self.base if item["example_id"] == row["source_example_id"])
            differences = [i for i, (a, b) in enumerate(zip(original["input_ids"], row["input_ids"])) if a != b]
            self.assertEqual(differences, [original["fact_position"]])
            self.assertNotIn(row["target_token_id"], by_record[row["record_name"]])

    def test_context_changes_never_combine_filler_and_position(self):
        for row in self.datasets["context"][16:]:
            original = next(item for item in self.base if item["example_id"] == row["source_example_id"])
            self.assertEqual(row["target_token_id"], original["target_token_id"])
            if row["condition"].startswith("filler_"):
                self.assertEqual(row["fact_position"], original["fact_position"])
                start, end = original["fact_position"] + 7, original["final_query_position"]
                self.assertEqual(row["input_ids"][:start], original["input_ids"][:start])
                self.assertEqual(row["input_ids"][end:], original["input_ids"][end:])
                self.assertNotEqual(row["input_ids"][start:end], original["input_ids"][start:end])
                self.assertTrue(set(row["input_ids"][start:end]).isdisjoint(row["candidate_token_ids"]))
            else:
                offset = row["fact_start_position"]
                self.assertIn(offset, (32, 64))
                self.assertEqual(row["input_ids"][offset:], original["input_ids"])
                self.assertEqual(row["final_query_position"], original["final_query_position"] + offset)
                self.assertTrue(set(row["input_ids"][:offset]).isdisjoint(row["candidate_token_ids"]))

    def test_paraphrases_preserve_prefix_length_positions_and_answer_prefix(self):
        for row in self.datasets["questions"][16:]:
            original = next(item for item in self.base if item["example_id"] == row["source_example_id"])
            boundary = original["final_query_position"]
            self.assertEqual(row["input_ids"][:boundary], original["input_ids"][:boundary])
            self.assertEqual(row["input_ids"][-6:], original["input_ids"][-6:])
            self.assertNotEqual(row["input_ids"][boundary:], original["input_ids"][boundary:])
            for field in ("target_token_id", "fact_position", "final_query_position", "final_answer_position"):
                self.assertEqual(row[field], original[field])

    def test_preflight_accepts_only_exact_data_checkpoint_and_full_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training_metadata.json").write_text(json.dumps({"progress": {"step": 449}, "reason": "signal"}))
            (root / "config.json").write_text(json.dumps({"num_hidden_layers": 28, "working_memory_size": 32, "memory_chunk_size": 1}))
            source = root / "reference.jsonl"
            source.write_text(dataset_text(self.base))
            for name, examples in self.datasets.items():
                with self.subTest(experiment=name):
                    data = root / f"{name}.jsonl"
                    data.write_text(dataset_text(examples))
                    data.with_suffix(".metadata.json").write_text(json.dumps(dataset_metadata(examples, name)))
                    config = yaml.safe_load((self.root / f"evaluation/configs/taal/d_heldout_{name}.yaml").read_text())
                    config["dataset"]["path"] = str(data)
                    path = root / "audit.yaml"
                    path.write_text(yaml.safe_dump(config))
                    with patch("evaluation.taal.heldout.checkpoint_fingerprint", return_value=D_CHECKPOINT_SHA256):
                        with redirect_stdout(io.StringIO()):
                            self.assertEqual(len(verify_inputs(name, path, root, self.tokenizer, source)), len(examples))
                        config["dataset"]["end"] -= 2
                        path.write_text(yaml.safe_dump(config))
                        with self.assertRaisesRegex(ValueError, "select every"):
                            verify_inputs(name, path, root, self.tokenizer, source)
                        config["dataset"]["end"] += 2
                        path.write_text(yaml.safe_dump(config))
                        data.write_text(dataset_text(examples[:-1]))
                        with self.assertRaisesRegex(ValueError, "isolated setup"):
                            verify_inputs(name, path, root, self.tokenizer, source)
                    data.write_text(dataset_text(examples))
                    with patch("evaluation.taal.heldout.checkpoint_fingerprint", return_value="wrong"):
                        with self.assertRaisesRegex(ValueError, "Checkpoint differs"):
                            verify_inputs(name, path, root, self.tokenizer, source)

    def test_cpu_audit_each_arm_all_controls_eviction_resume_and_generated_traces(self):
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "CPU fixture"),
            device=lambda *args: torch.device("cpu"), tensor=torch.tensor,
            long=torch.long, logsumexp=torch.logsumexp,
        )
        for name, dataset in self.datasets.items():
            examples = [row for row in dataset if row["record_name"] == "Record R0000"]
            with self.subTest(experiment=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / "data.jsonl"
                data.write_text(dataset_text(examples))
                data.with_suffix(".metadata.json").write_text(json.dumps(dataset_metadata(examples, name)))
                config = yaml.safe_load((self.root / f"evaluation/configs/taal/d_heldout_{name}.yaml").read_text())
                config["dataset"]["path"] = str(data)
                path = root / "audit.yaml"
                path.write_text(yaml.safe_dump(config))
                tiny = test_taal_state_audit.TaalStateAuditTests.config()
                tiny.vocab_size = len(self.tokenizer)
                tiny.working_memory_size = 32
                torch.manual_seed(42)
                model = TaalQwen3ForCausalLM(tiny).eval()
                with torch.no_grad():
                    model.model.layers[0].taal.residual_gate.fill_(0.05)
                with torch.inference_mode():
                    for row in examples:
                        prefix, query = state_audit.split_episode(row)
                        output = model.prefill(torch.tensor([prefix]), execution_block_size=32)
                        positions = output.state.past_key_values.layers[0].positions.tolist()
                        self.assertTrue(set(positions).isdisjoint(range(row["fact_start_position"], row["fact_position"] + 7)))
                        self.assertEqual(len(query), 20)
                output_dir = root / "audit"
                argv = ["audit", "--config", str(path), "--checkpoint", str(root), "--output-dir", str(output_dir)]
                with (
                    patch("sys.argv", argv), patch.object(state_audit, "torch", cpu_torch),
                    patch.object(state_audit, "load_dataset", return_value=examples),
                    patch.object(state_audit, "checkpoint_fingerprint", return_value="CPU-fixture"),
                    patch.object(state_audit, "load_model", return_value=model),
                    patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=self.tokenizer),
                    redirect_stdout(io.StringIO()),
                ):
                    state_audit.main()
                    state_audit.main()
                rows = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), len(examples) * 6)
                summary = summarize_heldout(rows, examples)
                self.assertEqual(set(summary["arms"]), {row["condition"] for row in examples})
                for arm in summary["arms"].values():
                    self.assertEqual(arm["record_pairs"], 1)
                    self.assertEqual(arm["metrics"]["correct_full"]["examples"], 2)
                with self.assertRaisesRegex(ValueError, "every unique"):
                    summarize_heldout(rows[:-1], examples)
                invalid = copy.deepcopy(rows)
                invalid[0]["target_log_probability"] = float("nan")
                with self.assertRaises(ValueError):
                    summarize_heldout(invalid, examples)
                manifest = json.loads((output_dir / "memory_traces/manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), len(rows))
                self.assertEqual(len(manifest["comparisons"]), len(examples) * 5)
                donors = state_audit.swap_sources(examples, "pair_group")
                for key, filename in manifest["episodes"].items():
                    with gzip.open(output_dir / "memory_traces" / filename, "rt") as handle:
                        trace = json.load(handle)
                    self.assertEqual(trace["tokens"][-1]["phase"], "generated")
                    self.assertEqual(trace["tokens"][-1]["token_id"], trace["outcome"]["vocabulary_top_token_id"])
                    self.assertEqual(len(trace["reads"]), len(trace["tokens"]))
                    self.assertEqual(len(trace["writes"]), len(trace["tokens"]) + tiny.num_persistent_tokens)
                    if key.endswith("/swapped"):
                        self.assertEqual(trace["outcome"]["swapped_from_example_id"], donors[trace["example_id"]]["example_id"])
                    json.dumps(trace, allow_nan=False)

    def test_launcher_preserves_h200_preflight_and_commit_guard(self):
        script = (self.root / "evaluation/sbatch/taal/fs_qwen_d_heldout_short.sbatch").read_text()
        for value in ("--partition=gpu-short", "--gres=gpu:h200:1", "--time=02:00:00", "git rev-parse HEAD", "http_proxy", "HF_HOME", "-m evaluation.taal.heldout"):
            self.assertIn(value, script)

    def test_summary_keeps_reference_separate_and_requires_both_pair_directions(self):
        examples = self.datasets["bindings"][:2] + self.datasets["bindings"][16:18]
        rows = []
        for example in examples:
            for condition in audit_conditions("standard"):
                correct = condition.name == "correct_full"
                rows.append({
                    "evaluation_id": f"{example['example_id']}/{condition.name}",
                    "example_id": example["example_id"],
                    "dataset_condition": example["condition"],
                    "record_group": example["pair_group"],
                    "audit_condition": condition.name,
                    "candidate_correct": correct, "vocabulary_top_1_correct": correct,
                    "target_log_probability": -0.5 if correct else -1.5,
                    "target_reciprocal_rank": 1.0 if correct else 0.5,
                    "target_vs_best_decoy_margin": 1.0 if correct else -1.0,
                    "donor_target_vocabulary_top_1": condition.name == "swapped",
                    "own_minus_donor_label_log_probability": 1.0 if correct else -1.0,
                })
        summary = summarize_heldout(rows, examples)
        self.assertEqual(set(summary["arms"]), {"reference", "binding_shift_1"})
        for arm in summary["arms"].values():
            self.assertEqual(arm["metrics"]["correct_full"]["examples"], 2)
            self.assertEqual(arm["metrics"]["correct_full"]["answer_loss"], 0.5)
            self.assertEqual(arm["pairs_both_directions_own_and_donor_correct"], 1)
            self.assertEqual(arm["paired_vs_correct_full"]["reads_disabled"]["correct_full_minus_condition_target_log_probability"], 1.0)
        self.assertTrue(all(not row["same_target"] for row in summary["reference_contrasts"]))
        rows[-1]["donor_target_vocabulary_top_1"] = False
        summary = summarize_heldout(rows, examples)
        self.assertEqual(summary["arms"]["binding_shift_1"]["pairs_both_directions_own_and_donor_correct"], 0)
        self.assertEqual(summary["arms"]["reference"]["pairs_both_directions_own_and_donor_correct"], 1)


if __name__ == "__main__":
    unittest.main()
