import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.taal.address_diagnostic import (
    AddressCapture, audit_example, collect_prefix, mapping_rows,
    reconstruction, select_heldout,
)
from evaluation.taal.prefix_store import PrefixStore
from evaluation.taal.state_audit import clone_memory_states, fork_session_state
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder
from evaluation.storage import read_results
from tests import test_taal_state_audit


class AddressTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(73)
        config = test_taal_state_audit.TaalStateAuditTests.config()
        config.num_hidden_layers = 2
        config.layer_types = ["sliding_attention"] * 2
        model = TaalQwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for layer in model.model.layers:
                layer.taal.residual_gate.fill_(0.25)
        return model

    @torch.inference_mode()
    def test_capture_matches_native_and_normalized_inputs(self):
        model = self.model()
        ids = torch.tensor([[1, 2, 3]])
        expected = model.prefill(ids)
        capture = AddressCapture(model, start=0, persistent=True)
        with capture:
            actual = model.prefill(ids)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        payload = capture.payload()
        for layer, events in capture.events.items():
            self.assertEqual(len(events), 5)
            self.assertEqual([event["position"] for event in events], [None, None, 0, 1, 2])
            for event in events:
                self.assertAlmostEqual(event["key"].norm().item(), 1.0, places=5)
                self.assertAlmostEqual(event["query"].norm().item(), 1.0, places=5)
        json.dumps(payload, allow_nan=False)

    @torch.inference_mode()
    def test_oracle_replays_identical_writes_and_restores_hooks(self):
        model = self.model()
        state = model.prefill(torch.tensor([[1, 2, 3]])).state
        def fork():
            return fork_session_state(state, clone_memory_states(state.memory_states))
        native = AddressCapture(model, start=3, capture_writes=True)
        with native:
            baseline = model.prefill(torch.tensor([[4]]), state=fork(), prepend_memory_tokens=False)
        identity = {layer: write.query for layer, write in native.writes.items()}
        same = AddressCapture(model, start=3, reference=native.writes, addresses=identity, capture_writes=True)
        with same:
            replayed = model.prefill(torch.tensor([[4]]), state=fork(), prepend_memory_tokens=False)
        torch.testing.assert_close(replayed.logits, baseline.logits, rtol=0, atol=0)
        alternate = {layer: -write.query for layer, write in native.writes.items()}
        original = model.model.layers[0].taal.neural_memory._process_chunk
        oracle = AddressCapture(model, start=3, reference=native.writes, addresses=alternate, capture_writes=True)
        recorder = TaalTraceRecorder(model, oracle.layers, 1, position_offset=3)
        with recorder, oracle:
            changed = model.prefill(torch.tensor([[4]]), state=fork(), prepend_memory_tokens=False)
        self.assertFalse(torch.equal(changed.logits, baseline.logits))
        self.assertEqual(len(recorder.writes), 2)
        for layer, write in oracle.writes.items():
            torch.testing.assert_close(write.keys, native.writes[layer].keys, rtol=0, atol=0)
            torch.testing.assert_close(write.values, native.writes[layer].values, rtol=0, atol=0)
            for name, value in changed.state.memory_states[layer].weights.items():
                torch.testing.assert_close(value, baseline.state.memory_states[layer].weights[name], rtol=0, atol=0)
                torch.testing.assert_close(changed.state.memory_states[layer].momentum[name], baseline.state.memory_states[layer].momentum[name], rtol=0, atol=0)
        self.assertEqual(model.model.layers[0].taal.neural_memory._process_chunk, original)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with AddressCapture(model, start=3):
                raise RuntimeError("sentinel")
        self.assertEqual(model.model.layers[0].taal.neural_memory._process_chunk, original)

    def test_selection_and_reconstruction_controls(self):
        rows = [{"example_id": "a", "probe_split": "test", "pair_group": "p", "target_token_id": 1},
                {"example_id": "b", "probe_split": "test", "pair_group": "p", "target_token_id": 2},
                {"example_id": "c", "probe_split": "train", "pair_group": "q", "target_token_id": 3}]
        selected, donors = select_heldout(rows, "test")
        self.assertEqual(len(selected), 2)
        self.assertEqual(donors["a"]["example_id"], "b")
        with self.assertRaises(ValueError):
            select_heldout(rows[:1], "test")
        self.assertEqual(reconstruction(torch.ones(3), torch.ones(3))["mse"], 0.0)
        self.assertGreater(reconstruction(-torch.ones(3), torch.ones(3))["relative_error"], 0)
        with self.assertRaises(ValueError):
            reconstruction(torch.tensor([float("nan")]), torch.ones(1))

    def test_end_to_end_snapshots_oracles_and_export_resume(self):
        from transformers import AutoTokenizer
        from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
        from data.taal_state_probe import build_probe_examples
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        selected, donors = select_heldout(build_probe_examples(tokenizer), "test")
        examples = selected[:2]
        self.assertEqual(donors[examples[0]["example_id"]], examples[1])
        config = test_taal_state_audit.TaalStateAuditTests.config()
        config.vocab_size = len(tokenizer)
        config.working_memory_size = 128
        model = TaalQwen3ForCausalLM(config).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter = TaalTraceExporter(root / "memory_traces", run_metadata={"checkpoint": "test", "tokenizer": MODEL_ID})
            store = PrefixStore(root / "prefixes", {"test": True})
            settings = {"mode": "reconstruction", "execution_block_size": 128}
            for example in examples:
                collect_prefix(model, tokenizer, example, root, exporter, store, settings)
            metrics, vectors = mapping_rows(model, examples[0], examples[1], root)
            self.assertEqual(len(metrics), 6)
            self.assertEqual(len(vectors), 6)
            self.assertEqual({row["stage"] for row in metrics}, {"fact", "middle", "final"})
            settings["mode"] = "oracle"
            for example in examples:
                audit_example(model, tokenizer, example, donors[example["example_id"]], root, exporter, store, settings)
            audit_example(model, tokenizer, examples[0], examples[1], root, exporter, store, settings)
            self.assertEqual(len(read_results(root / "results.jsonl", id_field="evaluation_id")), 8)
            self.assertEqual(len(exporter.manifest["episodes"]), 8)
            for path in (root / "memory_traces" / "vectors").glob("*.json.gz"):
                with gzip.open(path, "rt") as handle:
                    json.loads(handle.read())

    def test_short_launcher_and_trace_defaults(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "evaluation/sbatch/taal/fs_qwen_address_short.sbatch").read_text()
        self.assertIn("#SBATCH --partition=gpu-short", launcher)
        self.assertIn("#SBATCH --gres=gpu:h200:1", launcher)
        self.assertIn("utils.explorer_preflight", launcher)
        for mode in ("reconstruction", "oracle"):
            settings = yaml.safe_load((root / f"evaluation/configs/taal/address_{mode}.yaml").read_text())
            self.assertTrue(settings["save_traces"])
            self.assertEqual(settings["mode"], mode)


if __name__ == "__main__":
    unittest.main()
