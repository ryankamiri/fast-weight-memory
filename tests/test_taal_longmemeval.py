import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from datasets import Dataset
import torch
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from evaluation.data import PROMPT_VERSION
from evaluation.run import configure_model, generate_example, load_model, main
from evaluation.taal.longmemeval import create_trace_exporter, resolve_generation_settings, select_dataset


class Tokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(str(token) for token in ids)


class TaalLongMemEvalTests(unittest.TestCase):
    def test_shipped_recipes_with_actual_d_checkpoint_config(self):
        root = Path(__file__).resolve().parents[1]
        config = TaalQwen3Config.from_json_file(root / "tests/fixtures/taal_d_final_config.json")
        original = config.to_dict()
        self.assertEqual(config.working_memory_size, 32)
        self.assertEqual(config.memory_chunk_size, 1)
        self.assertEqual(config.num_hidden_layers, 28)
        for suffix in ("", "_half_reads", "_no_reads", "_smoke"):
            with self.subTest(suffix=suffix):
                recipe = yaml.safe_load((root / f"evaluation/configs/taal/longmemeval{suffix}.yaml").read_text())
                configure_model(config, recipe["mode"], recipe["model"])
                generation = resolve_generation_settings(recipe["generation"], config.working_memory_size)
                self.assertEqual(generation["execution_block_size"], 32)
                self.assertIsNone(recipe["generation"]["execution_block_size"])
                self.assertTrue(recipe["save_traces"])
        self.assertEqual(config.to_dict(), original)
        for block in (256, 0, -1, 1.5, True):
            with self.subTest(block=block), self.assertRaisesRegex(ValueError, "execution_block_size"):
                resolve_generation_settings({"execution_block_size": block}, config.working_memory_size)
        self.assertEqual(resolve_generation_settings({}, 32)["execution_block_size"], 32)
        self.assertEqual(resolve_generation_settings({"execution_block_size": 16}, 32)["execution_block_size"], 16)

    def test_automatic_block_size_generates_and_exports_without_changing_kv(self):
        self.settings["generation"]["execution_block_size"] = None
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            with patch.object(self.model, "prefill", wraps=self.model.prefill) as prefill:
                actual = self.generate(exporter)
            self.assertEqual(prefill.call_args.kwargs["execution_block_size"], 4)
            self.assertGreater(actual["generated_tokens"], 0)
            self.assertEqual(len(exporter.manifest["episodes"]), 1)
            self.assertEqual(self.model.config.working_memory_size, 4)

    def setUp(self):
        torch.manual_seed(123)
        self.config = TaalQwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, working_memory_size=4, max_persistent_kv_tokens=0,
            memory_dim=8, memory_depth=2, memory_conv_kernel_size=2,
            memory_chunk_size=1, num_persistent_tokens=2,
        )
        self.config._attn_implementation = "sdpa"
        self.model = TaalQwen3ForCausalLM(self.config).eval()
        self.settings = {
            "seed": 42, "mode": "taal", "model": {},
            "generation": {"execution_block_size": 3, "max_new_tokens": 3,
                           "do_sample": False, "eos_token_id": []},
        }
        self.example = {
            "question_id": "question-1", "input_ids": [1, 2, 3, 4, 5, 6, 7],
            "prompt_length": 7,
            "persistent_prefix_length": 2, "answer": "reference only",
            "question": "What is it?", "question_type": "single-session-user",
        }
        self.metadata = {"tokenizer": "test-tokenizer", "tokenizer_revision": "test"}

    def exporter(self, directory, predictions=None):
        return create_trace_exporter(
            Path(directory), "test-checkpoint", self.config, self.metadata,
            self.settings, predictions or {},
        )

    def generate(self, exporter=None):
        return generate_example(self.model, Tokenizer(), self.example, self.settings,
                                torch.device("cpu"), exporter)

    def test_trace_default_and_generation_parity_and_coverage(self):
        baseline = self.generate()
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            with patch.object(self.model, "prefill", wraps=self.model.prefill) as prefill:
                actual = self.generate(exporter)
            self.assertEqual(actual, baseline)
            self.assertIsNone(prefill.call_args.kwargs["state"])
            self.assertIsNone(prefill.call_args.kwargs["persistent_mask"])
            relative = exporter.manifest["episodes"]["question-1/native"]
            with gzip.open(exporter.root / relative, "rt") as stream:
                trace = json.load(stream)
            self.assertEqual(len(trace["tokens"]), 10)
            generated = [token for token in trace["tokens"] if token["phase"] == "generated"]
            self.assertEqual([token["token_id"] for token in generated], actual["generated_token_ids"])
            for layer in (0, 1):
                self.assertEqual([event["position"] for event in trace["reads"] if event["layer"] == layer], list(range(10)))
                self.assertEqual(len([event for event in trace["writes"] if event["layer"] == layer]), 12)
            self.assertEqual(len(trace["internal_prefixes"]), 2)
            self.assertEqual(trace["metadata"]["expected_answer"], "reference only")
            self.assertEqual(trace["outcome"], actual)
            self.exporter(directory, {"question-1": actual})
            (exporter.root / relative).unlink()
            with self.assertRaisesRegex(ValueError, "no trace"):
                self.exporter(directory, {"question-1": actual})

    def test_eos_feedback_is_traced(self):
        first = self.generate()["generated_token_ids"][0]
        self.settings["generation"]["eos_token_id"] = [first]
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            result = self.generate(exporter)
            self.assertEqual(result["stop_reason"], "eos")
            self.assertEqual(result["generated_tokens"], 1)
            path = exporter.root / exporter.manifest["episodes"]["question-1/native"]
            with gzip.open(path, "rt") as stream:
                trace = json.load(stream)
            self.assertEqual(len(trace["reads"]), 16)
            self.assertEqual(trace["tokens"][-1]["phase"], "generated")

    def test_sampling_and_each_question_start_fresh(self):
        self.settings["generation"].update(do_sample=True, top_k=5)
        baseline = self.generate()
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            self.assertEqual(self.generate(exporter), baseline)
            self.assertEqual(self.generate(exporter), baseline)

    def test_chunked_memory_trace_parity_across_execution_blocks(self):
        self.config.memory_chunk_size = 4
        self.model = TaalQwen3ForCausalLM(self.config).eval()
        baseline = self.generate()
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            self.assertEqual(self.generate(exporter), baseline)
            path = exporter.root / exporter.manifest["episodes"]["question-1/native"]
            with gzip.open(path, "rt") as stream:
                trace = json.load(stream)
            self.assertTrue(all(event["chunk_size"] == 4 for event in trace["writes"]))
            self.assertTrue(any(not event["chunk_boundary"] for event in trace["writes"]))

    def test_generation_failure_detaches_observers(self):
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            with patch.object(self.model, "generate", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    self.generate(exporter)
            self.assertFalse(exporter.manifest["episodes"])
            for layer in self.model.model.layers:
                self.assertIsNone(layer.taal.trace_observer)
                self.assertIsNone(layer.taal.neural_memory.trace_observer)

    def test_reads_off_keeps_writes_and_all_generation_events(self):
        self.settings["model"]["memory_read_scale"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            self.generate(exporter)
            path = exporter.root / exporter.manifest["episodes"]["question-1/native"]
            with gzip.open(path, "rt") as stream:
                trace = json.load(stream)
            self.assertTrue(all(not event["enabled"] for event in trace["reads"]))
            self.assertTrue(all(event["write_enabled"] for event in trace["writes"]))
            self.assertEqual(trace["metadata"]["read_scale_scope"], "whole_episode")

    def test_prefix_retention_is_explicit_and_budget_checked(self):
        self.settings["model"]["retain_task_prefix"] = True
        with self.assertRaisesRegex(ValueError, "budget"):
            self.generate()
        self.config.max_persistent_kv_tokens = 2
        self.model = TaalQwen3ForCausalLM(self.config).eval()
        with patch.object(self.model, "prefill", wraps=self.model.prefill) as prefill:
            self.generate()
        self.assertEqual(prefill.call_args.kwargs["persistent_mask"].tolist(), [True, True] + [False] * 5)

    def test_checkpoint_loading_and_geometry_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            config = configure_model(TaalQwen3Config.from_pretrained(directory), "taal", {})
            loaded = load_model(directory, config).eval()
            self.assertIsInstance(loaded, TaalQwen3ForCausalLM)
            self.assertEqual(loaded.config.working_memory_size, 4)
            self.assertEqual(loaded.config.memory_chunk_size, 1)
            for name, value in loaded.state_dict().items():
                torch.testing.assert_close(value, self.model.state_dict()[name].to(value.dtype), rtol=0, atol=0)

    def test_invalid_architecture_and_options_rejected(self):
        with self.assertRaises(ValueError):
            configure_model(TTCDQwen3Config(), "taal")
        with self.assertRaises(ValueError):
            configure_model(self.config, "fw_swa")
        for settings in ({"chunk_size": 8}, {"memory_read_scale": float("nan")},
                         {"memory_read_scale": -1}, {"retain_task_prefix": "false"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                configure_model(self.config, "taal", settings)

    def test_trace_toggle_and_resume_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            exporter = self.exporter(directory)
            result = self.generate(exporter)
            with self.assertRaisesRegex(ValueError, "disagree"):
                self.exporter(directory, {"question-1": {**result, "hypothesis": "different"}})
        self.settings["save_traces"] = False
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(self.exporter(directory))
            self.assertFalse((Path(directory) / "memory_traces").exists())
        self.settings["save_traces"] = "false"
        with self.assertRaises(ValueError):
            self.exporter("unused")

    def test_bounds_preserve_selected_source_rows(self):
        dataset = Dataset.from_list([{"question_id": str(i)} for i in range(5)])
        self.assertEqual(select_dataset(dataset, {"start": 1, "end": 3})["question_id"], ["1", "2"])
        self.assertEqual(len(select_dataset(dataset, {})), 5)
        for start, end in ((-1, 3), (3, 3), (0, 6), (True, 3)):
            with self.assertRaises(ValueError):
                select_dataset(dataset, {"start": start, "end": end})

    def test_config_and_launcher(self):
        root = Path(__file__).resolve().parents[1]
        settings = yaml.safe_load((root / "evaluation/configs/taal/longmemeval.yaml").read_text())
        self.assertEqual(settings["mode"], "taal")
        self.assertTrue(settings["save_traces"])
        self.assertFalse(settings["model"]["retain_task_prefix"])
        self.assertEqual(settings["dataset"]["split"], "test")
        self.assertIsNone(settings["dataset"]["end"])
        launcher = (root / "evaluation/sbatch/taal/fs_qwen_longmemeval_short.sbatch").read_text()
        self.assertIn("--partition=gpu-short", launcher)
        self.assertIn("--gres=gpu:h200:1", launcher)
        self.assertIn("utils.explorer_preflight", launcher)
        self.assertIn("evaluation/configs/taal/longmemeval.yaml", launcher)
        self.assertIn('config="${3:-', launcher)
        self.assertIn('"${@:4}"', launcher)
        for name, scale in (("half_reads", 0.5), ("no_reads", 0.0)):
            variant = yaml.safe_load((root / f"evaluation/configs/taal/longmemeval_{name}.yaml").read_text())
            self.assertEqual(variant["model"]["memory_read_scale"], scale)
            variant["model"]["memory_read_scale"] = 1.0
            self.assertEqual(variant, settings)

    def test_runner_generation_and_completed_resume_offline(self):
        dataset = Dataset.from_list([{
            "question_id": "question-1", "question_type": "single-session-user",
            "question": "What is it?", "answer": "reference only", "abstention": False,
        }])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self.model.save_pretrained(checkpoint)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(json.dumps({
                **self.metadata, "records": 1, "prompt_version": PROMPT_VERSION,
            }))
            config_path = root / "eval.yaml"
            config_path.write_text(yaml.safe_dump({
                **self.settings,
                "dataset": {"repo_id": "offline", "variant": "oracle", "revision": "test"},
                "judge": {"model": "gpt-5.6-terra", "concurrency": 1},
            }))
            arguments = ["evaluation.run", "--config", str(config_path), "--checkpoint", str(checkpoint),
                         "--output-dir", str(root / "output"), "--generate-only"]

            def cpu_generate(model, tokenizer, example, settings, device, exporter):
                return generate_example(model, tokenizer, example, settings, torch.device("cpu"), exporter)

            with (
                patch("sys.argv", arguments),
                patch("evaluation.run.load_dotenv"),
                patch("evaluation.run.hf_hub_download", return_value=str(metadata_path)),
                patch("evaluation.run.load_dataset", return_value=dataset),
                patch("evaluation.run.prepare_example", return_value=self.example),
                patch("evaluation.run.AutoTokenizer.from_pretrained", return_value=Tokenizer()),
                patch("evaluation.run.torch.cuda.is_available", return_value=True),
                patch("evaluation.run.seed_everything"),
                patch("evaluation.run.load_model", return_value=self.model) as loader,
                patch.object(self.model, "to", return_value=self.model),
                patch("evaluation.run.generate_example", side_effect=cpu_generate),
            ):
                main()
                main()
            self.assertEqual(loader.call_count, 1)
            results = (root / "output/predictions.jsonl").read_text().splitlines()
            self.assertEqual(len(results), 1)
            manifest = json.loads((root / "output/memory_traces/manifest.json").read_text())
            self.assertEqual(set(manifest["episodes"]), {"question-1/native"})
            self.assertEqual(len(manifest["run_metadata"]["settings"]["checkpoint_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main()
