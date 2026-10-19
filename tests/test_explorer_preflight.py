import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from utils import explorer_preflight


class PreflightTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class ExplorerPreflightTests(unittest.TestCase):
    def test_longmemeval_checkpoint_preflight_end_to_end(self):
        root = Path(__file__).resolve().parents[1]
        config = root / "evaluation/configs/taal/longmemeval.yaml"
        row = {
            "question_id": "first", "question": "Label?", "answer": "amber",
            "question_date": "2026/01/02 (Fri) 12:00",
            "haystack_dates": ["2026/01/01 (Thu) 12:00"],
            "haystack_sessions": [[{"role": "user", "content": "amber"}]],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            for name in ("config.json", "model.safetensors", "training_metadata.json"):
                (checkpoint / name).touch()
            (checkpoint / "config.json").write_text(json.dumps({
                "model_type": "taal_qwen3", "working_memory_size": 32,
            }))
            metadata = checkpoint / "metadata.json"
            metadata.write_text(json.dumps({
                "source": explorer_preflight.SOURCE, "prompt_version": explorer_preflight.PROMPT_VERSION,
                "tokenizer": "test", "tokenizer_revision": "pinned",
            }))
            output = io.StringIO()
            with (
                patch.dict(os.environ, {name: directory for name in (
                    "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "http_proxy", "https_proxy",
                )}),
                patch("sys.argv", ["preflight", "--config", str(config), "--checkpoint", directory]),
                patch.object(explorer_preflight.torch.cuda, "is_available", return_value=True),
                patch.object(explorer_preflight.torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(explorer_preflight.torch.cuda, "get_device_name", return_value="test GPU"),
                patch.object(explorer_preflight, "hf_hub_download", return_value=str(metadata)),
                patch.object(explorer_preflight, "load_dataset", return_value=[row]) as dataset,
                patch.object(explorer_preflight.AutoTokenizer, "from_pretrained", return_value=PreflightTokenizer()),
                patch.object(explorer_preflight.requests, "get"),
                patch.object(explorer_preflight.subprocess, "check_output", return_value="commit\n"),
                contextlib.redirect_stdout(output),
            ):
                explorer_preflight.main()
            result = json.loads(output.getvalue().removeprefix("EXPLORER_PREFLIGHT "))
            self.assertEqual(result["status"], "pass")
            self.assertGreater(result["first_record_tokens"], 0)

            settings = yaml.safe_load(config.read_text())
            settings["generation"]["execution_block_size"] = 256
            bad_config = checkpoint / "bad.yaml"
            bad_config.write_text(yaml.safe_dump(settings))
            with (
                patch.dict(os.environ, {name: directory for name in (
                    "HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "http_proxy", "https_proxy",
                )}),
                patch("sys.argv", ["preflight", "--config", str(bad_config), "--checkpoint", directory]),
                patch.object(explorer_preflight.torch.cuda, "is_available", return_value=True),
                patch.object(explorer_preflight.torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(explorer_preflight, "load_dataset") as dataset,
                patch.object(explorer_preflight.requests, "get") as request,
            ):
                with self.assertRaisesRegex(ValueError, "execution_block_size"):
                    explorer_preflight.main()
                dataset.assert_not_called()
                request.assert_not_called()

    def test_longmemeval_source_records_are_prepared_before_validation(self):
        row = {
            "question_id": "test", "question": "What is the label?", "answer": "amber",
            "question_date": "2026/01/02 (Fri) 12:00",
            "haystack_dates": ["2026/01/01 (Thu) 12:00"],
            "haystack_sessions": [[{"role": "user", "content": "The label is amber."}]],
            "history_ids": [1], "question_ids": [2],
        }
        metadata = {
            "source": explorer_preflight.SOURCE,
            "prompt_version": explorer_preflight.PROMPT_VERSION,
            "tokenizer": "test-tokenizer", "tokenizer_revision": "pinned",
        }
        with patch.object(explorer_preflight.AutoTokenizer, "from_pretrained", return_value=PreflightTokenizer()) as loader:
            prepared = explorer_preflight.prepare_preflight_record(row, metadata, {}, Path("checkpoint"))
            self.assertGreater(len(prepared["input_ids"]), 2)
            self.assertNotIn("input_ids", row)
            loader.assert_called_once_with("test-tokenizer", revision="pinned")
            with self.assertRaises(ValueError):
                explorer_preflight.prepare_preflight_record(
                    {**row, "haystack_dates": []}, metadata, {}, Path("checkpoint"),
                )
            with self.assertRaisesRegex(ValueError, "prompt version"):
                explorer_preflight.prepare_preflight_record(
                    row, {**metadata, "prompt_version": "wrong"}, {}, Path("checkpoint"),
                )

    def test_pretokenized_records_keep_existing_validation(self):
        with patch.object(explorer_preflight.AutoTokenizer, "from_pretrained") as loader:
            row = {"input_ids": [1, 2]}
            self.assertIs(explorer_preflight.prepare_preflight_record(row, {}, {}, None), row)
            self.assertIs(explorer_preflight.prepare_preflight_record(row, {}, {}, Path("checkpoint")), row)
            for row in ({}, {"input_ids": []}):
                with self.assertRaisesRegex(ValueError, "no input_ids"):
                    explorer_preflight.prepare_preflight_record(row, {}, {}, None)
            loader.assert_not_called()

    def test_training_uses_transformers_cache_and_pinned_data(self):
        root = Path(__file__).resolve().parents[1]
        config = root / "training/configs/taal/qwen3_0_6b_delayed_recall_overfit.yaml"
        settings = yaml.safe_load(config.read_text())
        with tempfile.TemporaryDirectory() as directory:
            # Cache contents are irrelevant here; the preflight verifies JSON readability.
            cached = Path(directory) / "config.json"
            cached.write_text("{}")
            env = {
                "HF_HOME": directory, "TRANSFORMERS_CACHE": directory + "/transformers",
                "HF_DATASETS_CACHE": directory + "/datasets", "http_proxy": "http://proxy",
                "https_proxy": "http://proxy",
            }
            output = io.StringIO()
            with (
                patch.dict(os.environ, env),
                patch("sys.argv", ["preflight", "--config", str(config)]),
                patch.object(explorer_preflight.torch.cuda, "is_available", return_value=True),
                patch.object(explorer_preflight.torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(explorer_preflight.torch.cuda, "get_device_name", return_value="test GPU"),
                patch.object(explorer_preflight, "hf_hub_download", return_value=str(cached)) as hub,
                patch.object(explorer_preflight, "load_dataset", return_value=[{"input_ids": [1, 2]}]) as dataset,
                patch.object(explorer_preflight.requests, "get") as request,
                patch.object(explorer_preflight.subprocess, "check_output", return_value="commit\n"),
                contextlib.redirect_stdout(output),
            ):
                explorer_preflight.main()
            self.assertEqual(hub.call_args_list[0].kwargs["cache_dir"], env["TRANSFORMERS_CACHE"])
            self.assertTrue(all(call.kwargs["local_files_only"] for call in hub.call_args_list))
            self.assertEqual(dataset.call_args.kwargs["revision"], settings["data"]["revision"])
            request.return_value.raise_for_status.assert_called_once()
            record = json.loads(output.getvalue().removeprefix("EXPLORER_PREFLIGHT "))
            self.assertEqual(record["status"], "pass")
            self.assertEqual(record["first_record_tokens"], 2)

    def test_no_gpu_fails_before_network_or_model_loading(self):
        root = Path(__file__).resolve().parents[1]
        config = root / "training/configs/taal/qwen3_0_6b_delayed_recall_overfit.yaml"
        with (
            patch("sys.argv", ["preflight", "--config", str(config)]),
            patch.object(explorer_preflight.torch.cuda, "is_available", return_value=False),
            patch.object(explorer_preflight.requests, "get") as request,
        ):
            with self.assertRaisesRegex(RuntimeError, "BF16-capable CUDA GPU"):
                explorer_preflight.main()
            request.assert_not_called()

    def test_diagnostic_launchers_request_short_h200_and_preflight(self):
        root = Path(__file__).resolve().parents[1]
        for domain in ("training", "evaluation"):
            launcher = (root / domain / "sbatch/taal/fs_qwen_delayed_recall_timing_short.sbatch").read_text()
            self.assertIn("#SBATCH --partition=gpu-short", launcher)
            self.assertIn("#SBATCH --gres=gpu:h200:1", launcher)
            self.assertIn("#SBATCH --time=02:00:00", launcher)
            self.assertIn("python -m utils.explorer_preflight", launcher)
            self.assertIn("export TAAL_TIMING_FIRST_BATCH=1", launcher)
            self.assertIn("export TAAL_KERNEL_SAMPLE=1", launcher)
            for name in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "http_proxy", "https_proxy"):
                self.assertIn(f"export {name}=", launcher)

    def test_cadence_launchers_keep_full_budget_and_explicit_config(self):
        root = Path(__file__).resolve().parents[1]
        for domain, filename in (
            ("training", "fs_qwen_delayed_recall_overfit.sbatch"),
            ("evaluation", "fs_qwen_delayed_recall_scores.sbatch"),
        ):
            launcher = (root / domain / "sbatch/taal" / filename).read_text()
            for setting in (
                "#SBATCH --partition=gpu", "#SBATCH --gres=gpu:h200:1",
                "#SBATCH --time=08:00:00", "python -m utils.explorer_preflight",
            ):
                self.assertIn(setting, launcher)
            for name in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "http_proxy", "https_proxy"):
                self.assertIn(f"export {name}=", launcher)
        training = (root / "training/sbatch/taal/fs_qwen_delayed_recall_overfit.sbatch").read_text()
        self.assertIn('TRAIN_CONFIG="${1:-training/configs/taal/qwen3_0_6b_delayed_recall_overfit.yaml}"', training)
        self.assertEqual(training.count('--config "${TRAIN_CONFIG}"'), 2)
        timing = (root / "training/sbatch/taal/fs_qwen_delayed_recall_timing_short.sbatch").read_text()
        self.assertEqual(timing.count("--config training/configs/taal/qwen3_0_6b_delayed_recall_overfit_chunk1.yaml"), 2)


if __name__ == "__main__":
    unittest.main()
