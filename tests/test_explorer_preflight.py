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


class ExplorerPreflightTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
