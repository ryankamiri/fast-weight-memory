from contextlib import redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.taal import stability_observed_audit, state_audit
from tests.test_taal_longmemeval import Tokenizer
from tests import test_taal_state_audit


class StabilityAuditTests(unittest.TestCase):
    def test_episode_intervention_survives_prefix_query_and_generated_token(self):
        torch.set_num_threads(1)
        cpu_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True, get_device_name=lambda _: "CPU fixture"
            ),
            device=lambda *args: torch.device("cpu"),
            tensor=torch.tensor,
            long=torch.long,
            logsumexp=torch.logsumexp,
        )
        examples = []
        for index, answer in enumerate(("amber", "anchor")):
            examples.append(
                {
                    "example_id": f"pair-{index}",
                    "fact_id": index,
                    "record_name": "same-record",
                    "answer": answer,
                    "condition": "micro_conflict",
                    "query_variant": "exact",
                    "input_ids": [1, 5 + index, 2, 3, 4, 7, 8, 9, 10],
                    "target_token_id": 5 + index,
                    "candidate_token_ids": [5, 6],
                    "fact_position": 1,
                    "final_query_position": 6,
                    "final_answer_position": 9,
                    "prompt_length": 9,
                }
            )
        for arm in ("baseline", "no_momentum"):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = TaalQwen3ForCausalLM(
                    test_taal_state_audit.TaalStateAuditTests.config()
                ).eval()
                model.save_pretrained(root / "checkpoint")
                memory = model.model.layers[0].taal.neural_memory
                original_update = memory._update
                momentum_flags = []

                def update(*args, **kwargs):
                    momentum_flags.append(memory.momentum_enabled)
                    self.assertEqual(memory.momentum_enabled, arm != "no_momentum")
                    return original_update(*args, **kwargs)

                dataset = root / "data.jsonl"
                dataset.write_text("".join(json.dumps(row) + "\n" for row in examples))
                dataset.with_suffix(".metadata.json").write_text(
                    json.dumps(
                        {
                            "tokenizer": "fixture",
                            "tokenizer_revision": "fixture",
                        }
                    )
                )
                settings = {
                    "dataset": {"path": str(dataset), "start": 0, "end": 2},
                    "execution_block_size": 4,
                    "seed": 42,
                    "save_traces": True,
                    "swap_group_field": "record_name",
                    "stability_intervention": arm,
                }
                config = root / "audit.yaml"
                config.write_text(yaml.safe_dump(settings))
                output = root / "audit"
                argv = [
                    "audit",
                    "--config",
                    str(config),
                    "--checkpoint",
                    str(root / "checkpoint"),
                    "--output-dir",
                    str(output),
                ]
                with (
                    patch("sys.argv", argv),
                    patch.dict(os.environ, {"TAAL_STABILITY_INTERVENTION": arm}),
                    patch.object(state_audit, "torch", cpu_torch),
                    patch.object(state_audit, "load_dataset", return_value=examples),
                    patch.object(state_audit, "load_model", return_value=model),
                    patch.object(
                        state_audit.AutoTokenizer,
                        "from_pretrained",
                        return_value=Tokenizer(),
                    ),
                    patch.object(memory, "_update", side_effect=update),
                    redirect_stdout(io.StringIO()),
                ):
                    stability_observed_audit.main()
                self.assertTrue(momentum_flags)
                self.assertTrue(memory.momentum_enabled)
                rows = [
                    json.loads(line)
                    for line in (output / "results.jsonl").read_text().splitlines()
                ]
                self.assertEqual(len(rows), 12)
                self.assertTrue(
                    all(
                        row["query_momentum_enabled"] == (arm != "no_momentum")
                        for row in rows
                    )
                )
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertTrue(
                    all(
                        condition["momentum_enabled"] == (arm != "no_momentum")
                        for condition in manifest["conditions"]
                    )
                )
                traces = json.loads(
                    (output / "memory_traces/manifest.json").read_text()
                )
                self.assertEqual(len(traces["episodes"]), 12)
                self.assertEqual(len(traces["comparisons"]), 10)
                for relative in traces["episodes"].values():
                    with gzip.open(output / "memory_traces" / relative, "rt") as stream:
                        episode = json.load(stream)
                    self.assertEqual(episode["tokens"][-1]["phase"], "generated")
                self.assertTrue((output / "read-rounding.jsonl").stat().st_size)


if __name__ == "__main__":
    unittest.main()
