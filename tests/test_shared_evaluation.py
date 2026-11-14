"""CPU checks for benchmark semantics, episode isolation and durable evaluation."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from tokenizers import Tokenizer as FastTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.ttcd.qwen.causal_lm import TTCDQwen3ForCausalLM
from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from evaluation.data import prepare_example
from evaluation.run import generate_example
from evaluation.shared.__main__ import main
from evaluation.shared.backends import QwenBackend
from evaluation.shared.benchmarks import (
    BabilongBenchmark,
    ControlledBenchmark,
    LongMemEvalBenchmark,
    PreparedControlledBenchmark,
    RulerBenchmark,
)
from evaluation.shared.contracts import Example, Prompt
from evaluation.shared.runner import rescore, run
from evaluation.shared.vendor.babilong import prompts as babilong_prompts
from evaluation.shared.vendor.ruler import metrics as official
from evaluation.storage import atomic_json, file_sha256


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return " ".join(str(token) for token in ids)


class SharedEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        config = TaalQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=2,
            memory_chunk_size=1,
            num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        self.model = TaalQwen3ForCausalLM(config).eval()
        self.tokenizer = Tokenizer()
        self.generation = {
            "execution_block_size": 3,
            "max_new_tokens": 3,
            "do_sample": False,
            "eos_token_id": [],
        }
        self.backend = QwenBackend(
            self.model,
            self.tokenizer,
            provenance={"checkpoint": "tiny", "tokenizer": "tiny"},
            generation=self.generation,
        )
        self.prompt = Prompt((1, 2, 3, 4, 5, 6, 7))
        self.examples = [
            Example("a", self.prompt, ("amber",), {"gap": 4}, "pair"),
            Example(
                "b", Prompt((1, 2, 3, 4, 5, 8, 9)), ("anchor",), {"gap": 4}, "pair"
            ),
        ]

    def test_taal_matches_existing_longmemeval_with_and_without_traces(self):
        expected = generate_example(
            self.model,
            self.tokenizer,
            {
                "question_id": "a",
                "input_ids": list(self.prompt.token_ids),
                "persistent_prefix_length": 0,
                "answer": "amber",
            },
            {"seed": 42, "generation": self.generation},
            torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for tracing in (False, True):
                actual = self.backend.generate(
                    "a", self.prompt, Path(directory), seed=42, save_traces=tracing
                )
                self.assertEqual(actual.hypothesis, expected["hypothesis"])
                self.assertEqual(
                    actual.generated_token_ids, expected["generated_token_ids"]
                )
                self.assertEqual(actual.stop_reason, expected["stop_reason"])
                self.assertEqual(
                    {a.kind for a in actual.artifacts},
                    {"token_trace", "memory_trace"} if tracing else set(),
                )

    def test_independent_questions_replay_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = self.backend.generate(
                "a", self.prompt, output, seed=42, save_traces=False
            )
            self.backend.generate(
                "b", self.examples[1].prompt, output, seed=42, save_traces=False
            )
            repeated = self.backend.generate(
                "a", self.prompt, output, seed=42, save_traces=False
            )
            self.assertEqual(first.generated_token_ids, repeated.generated_token_ids)
            settings = {"dataset_identity": {"revision": "fixture", "split": "test"}}
            summary = run(
                self.backend, ControlledBenchmark(), self.examples, output, **settings
            )
            self.assertEqual(summary["examples"], 2)
            self.assertEqual(summary["groups"], 1)
            with patch.object(
                self.backend,
                "generate",
                side_effect=AssertionError("resume regenerated"),
            ):
                self.assertEqual(
                    run(
                        self.backend,
                        ControlledBenchmark(),
                        self.examples,
                        output,
                        **settings,
                    ),
                    summary,
                )
            self.assertEqual(
                rescore(ControlledBenchmark(), self.examples, output), summary
            )
            trace = next((output / "token_traces").glob("*.json"))
            trace.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                run(
                    self.backend,
                    ControlledBenchmark(),
                    self.examples,
                    output,
                    **settings,
                )

    def test_partial_resume_with_new_backend_and_separate_trace_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_generate = self.backend.generate

            def interrupt_second(example_id, *args, **kwargs):
                if example_id == "b":
                    raise RuntimeError("interrupted")
                return original_generate(example_id, *args, **kwargs)

            with patch.object(self.backend, "generate", side_effect=interrupt_second):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run(
                        self.backend,
                        ControlledBenchmark(),
                        self.examples,
                        root / "first",
                        dataset_identity={},
                    )
            resumed = QwenBackend(
                self.model,
                self.tokenizer,
                provenance={"checkpoint": "tiny", "tokenizer": "tiny"},
                generation=self.generation,
            )
            with patch.object(resumed, "generate", wraps=resumed.generate) as generate:
                summary = run(
                    resumed,
                    ControlledBenchmark(),
                    self.examples,
                    root / "first",
                    dataset_identity={},
                )
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(generate.call_args.args[0], "b")
                self.assertEqual(summary["examples"], 2)
            run(
                resumed,
                ControlledBenchmark(),
                self.examples,
                root / "second",
                dataset_identity={},
            )
            self.assertTrue((root / "second/memory_traces/manifest.json").is_file())

    def test_changed_prompt_or_seed_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(
                self.backend,
                ControlledBenchmark(),
                self.examples,
                output,
                dataset_identity={},
                save_traces=False,
            )
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                run(
                    self.backend,
                    ControlledBenchmark(),
                    self.examples,
                    output,
                    dataset_identity={},
                    seed=7,
                    save_traces=False,
                )
            changed = [Example("a", Prompt((9,)), ("amber",)), self.examples[1]]
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                run(
                    self.backend,
                    ControlledBenchmark(),
                    changed,
                    output,
                    dataset_identity={},
                    save_traces=False,
                )

    def test_answers_never_enter_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                self.backend, "generate", wraps=self.backend.generate
            ) as generate:
                run(
                    self.backend,
                    ControlledBenchmark(),
                    self.examples[:1],
                    Path(directory),
                    dataset_identity={},
                    save_traces=False,
                )
            args = generate.call_args.args
            self.assertIsInstance(args[1], Prompt)
            self.assertFalse(hasattr(args[1], "references"))
            self.assertNotIn("amber", str(generate.call_args))

    def test_failure_is_logged_and_not_a_completed_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(
                self.backend, "generate", side_effect=RuntimeError("non-finite")
            ):
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    run(
                        self.backend,
                        ControlledBenchmark(),
                        self.examples,
                        output,
                        dataset_identity={},
                    )
            self.assertFalse((output / "predictions.jsonl").exists())
            self.assertEqual(
                json.loads((output / "failures.jsonl").read_text())["example_id"], "a"
            )

    def test_unsupported_controls_fail_and_no_state_can_leak(self):
        for setting in ({"state": object()}, {"wrong_option": True}):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                QwenBackend(
                    self.model, self.tokenizer, provenance={}, generation=setting
                )
        self.assertFalse(self.backend.capabilities.snapshot)
        self.assertFalse(self.backend.capabilities.continuation_score)
        with self.assertRaises(ValueError):
            Prompt(())

    def test_ttcd_attention_baseline_parity(self):
        config = TTCDQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            teacher_window_size=4,
            student_window_size=4,
            chunk_size=3,
            max_persistent_kv_tokens=0,
            fast_weight_layers=[],
        )
        config._attn_implementation = "sdpa"
        model = TTCDQwen3ForCausalLM(config).eval()
        backend = QwenBackend(
            model,
            self.tokenizer,
            provenance={"checkpoint": "tiny", "tokenizer": "tiny"},
            generation=self.generation,
        )
        expected = generate_example(
            model,
            self.tokenizer,
            {
                "question_id": "a",
                "input_ids": list(self.prompt.token_ids),
                "persistent_prefix_length": 0,
            },
            {"seed": 42, "generation": self.generation},
            torch.device("cpu"),
        )
        with tempfile.TemporaryDirectory() as directory:
            actual = backend.generate(
                "a", self.prompt, Path(directory), seed=42, save_traces=True
            )
            self.assertEqual(actual.hypothesis, expected["hypothesis"])
            self.assertEqual([a.kind for a in actual.artifacts], ["token_trace"])
        with self.assertRaisesRegex(ValueError, "only supported"):
            QwenBackend(
                model, self.tokenizer, provenance={}, generation={}, memory_read_scale=0
            )

    def test_eos_is_captured_and_invalid_options_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.backend.generate(
                "a", self.prompt, Path(directory), seed=42, save_traces=False
            )
            eos_backend = QwenBackend(
                self.model,
                self.tokenizer,
                provenance={"checkpoint": "tiny", "tokenizer": "tiny"},
                generation={
                    **self.generation,
                    "eos_token_id": first.generated_token_ids[0],
                },
            )
            result = eos_backend.generate(
                "eos", self.prompt, Path(directory), seed=42, save_traces=True
            )
            self.assertEqual(result.stop_reason, "eos")
            self.assertEqual(result.generated_tokens, 1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            QwenBackend(
                self.model,
                self.tokenizer,
                provenance={},
                generation={"max_new_tokens": 0},
            )
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.backend.require(["snapshot"])

    def test_corrupt_prediction_and_missing_memory_index_fail_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run(
                self.backend,
                ControlledBenchmark(),
                self.examples[:1],
                output,
                dataset_identity={},
            )
            prediction_path = output / "predictions.jsonl"
            row = json.loads(prediction_path.read_text())
            original = row["hypothesis"]
            row["hypothesis"] = "tampered"
            prediction_path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "disagree"):
                run(
                    self.backend,
                    ControlledBenchmark(),
                    self.examples[:1],
                    output,
                    dataset_identity={},
                )
            row["hypothesis"] = original
            prediction_path.write_text(json.dumps(row) + "\n")
            (output / "memory_traces/manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "trace index"):
                run(
                    self.backend,
                    ControlledBenchmark(),
                    self.examples[:1],
                    output,
                    dataset_identity={},
                )

    def test_cpu_cli_loads_checkpoint_generates_resumes_and_rescores(self):

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            self.model.save_pretrained(checkpoint)
            raw = FastTokenizer(
                WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]")
            )
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
            tokenizer.save_pretrained(checkpoint)
            data = root / "data.jsonl"
            data.write_text(
                json.dumps(
                    {"example_id": "1", "prompt": "hello world", "answer": "hello"}
                )
                + "\n"
            )
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "benchmark": {"name": "controlled"},
                        "dataset": {
                            "path": str(data),
                            "revision": "fixture-v1",
                            "split": "test",
                        },
                        "model": {
                            "mode": "taal",
                            "checkpoint": str(checkpoint),
                            "device": "cpu",
                            "dtype": "float32",
                        },
                        "generation": self.generation,
                    }
                )
            )
            argv = [
                "shared",
                "--config",
                str(config),
                "--output-dir",
                str(root / "results"),
            ]
            with redirect_stdout(io.StringIO()), patch("sys.argv", argv):
                main()
                main()
            with (
                redirect_stdout(io.StringIO()),
                patch("sys.argv", argv + ["--score-only"]),
                patch(
                    "evaluation.shared.__main__.load_model",
                    side_effect=AssertionError("Scoring loaded a model"),
                ),
            ):
                main()
            self.assertTrue((root / "results/memory_traces/manifest.json").is_file())
            self.assertEqual(
                len((root / "results/predictions.jsonl").read_text().splitlines()), 1
            )

    def test_benchmark_helpers_and_configs_are_self_contained(self):

        root = Path(babilong_prompts.__file__).parents[1]
        manifest = json.loads((root / "sources.json").read_text())
        for name, source in manifest.items():
            self.assertTrue((root / name / "LICENSE").is_file())
            for filename, provenance in source["files"].items():
                self.assertEqual(
                    hashlib.sha256((root / name / filename).read_bytes()).hexdigest(),
                    provenance["sha256"],
                )
        config_root = Path(__file__).resolve().parents[1] / "evaluation/configs/shared"
        for path in config_root.glob("*.yaml"):
            config = yaml.safe_load(path.read_text())
            self.assertNotIn("repo", config["benchmark"])
            self.assertNotIn("..", Path(config["dataset"]["path"]).parts)

    def test_failed_atomic_write_preserves_previous_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            atomic_json(path, {"status": "valid"})
            original_hash = file_sha256(path)
            with self.assertRaises(ValueError):
                atomic_json(path, {"invalid": float("nan")})
            self.assertEqual(file_sha256(path), original_hash)
            self.assertEqual(list(root.iterdir()), [path])
            self.assertEqual(
                original_hash, hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_multitoken_controlled_scoring(self):
        adapter = ControlledBenchmark()
        item = adapter.prepare(
            {"example_id": "1", "prompt": "What?", "answer": "amber anchor"},
            self.tokenizer,
        )
        self.assertEqual(adapter.score(item, " amber anchor ")["exact_match"], 1)
        self.assertEqual(adapter.score(item, "amber")["exact_match"], 0)
        self.assertEqual(adapter.score(item, "amber anchor blah")["exact_match"], 0)

    def test_ruler_partial_and_joint_are_different(self):
        adapter = RulerBenchmark("niah_multivalue")
        item = adapter.prepare(
            {
                "index": 0,
                "input": "Context",
                "answer_prefix": " Answer:",
                "outputs": ["amber", "anchor"],
            },
            self.tokenizer,
        )
        self.assertEqual(item.prompt.token_ids, tuple(b"Context Answer:"))
        metrics = adapter.score(item, "AMBER")
        self.assertEqual(metrics["official_match"], 0.5)
        self.assertEqual(metrics["all_references_present"], 0)

        self.assertEqual(
            official.string_match_all(["AMBER"], [["amber", "anchor"]]),
            metrics["official_match"] * 100,
        )

    def test_babilong_uses_official_prompt_and_rejects_extra_labels(self):
        adapter = BabilongBenchmark("qa1")
        item = adapter.prepare(
            {
                "example_id": "0",
                "input": "Mary went to the kitchen.",
                "question": "Where is Mary?",
                "target": "kitchen",
            },
            self.tokenizer,
        )
        self.assertEqual(
            adapter.score(item, "Mary is in the kitchen.")["official_accuracy"], 1
        )
        self.assertEqual(
            adapter.score(item, "kitchen or garden")["official_accuracy"], 0
        )
        expected = adapter.prompts.get_formatted_input(
            "Mary went to the kitchen.",
            "Where is Mary?",
            **adapter.prompts.DEFAULT_PROMPTS["qa1"],
        )
        self.assertEqual(item.prompt.token_ids, tuple(expected.encode()))

    def test_longmemeval_exact_prompt_parity(self):
        row = {
            "question_id": "q",
            "question_type": "single-session-user",
            "answer": "amber",
            "question": "What?",
            "question_date": "2026/01/02",
            "haystack_dates": ["2026/01/01 (Thu) 10:00"],
            "haystack_sessions": [[{"role": "user", "content": "amber"}]],
        }
        prepared = prepare_example(row, self.tokenizer)
        item = LongMemEvalBenchmark().prepare(row, self.tokenizer)
        self.assertEqual(list(item.prompt.token_ids), prepared["input_ids"])
        self.assertEqual(
            item.prompt.persistent_prefix_length, prepared["persistent_prefix_length"]
        )
        self.assertEqual(LongMemEvalBenchmark().score(item, "amber"), {})

    def test_prepared_controlled_refuses_answer_in_input(self):
        with self.assertRaisesRegex(ValueError, "before the answer"):
            PreparedControlledBenchmark().prepare(
                {"input_ids": [1, 2], "final_answer_position": 1}, self.tokenizer
            )


if __name__ == "__main__":
    unittest.main()
