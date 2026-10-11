import unittest
from pathlib import Path
import tempfile

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from evaluation.taal.retention_readers import fit_reader, predict
from evaluation.taal.retention_replay import FixedInputReplay, collect
from evaluation.taal.prefix_store import PrefixStore
from evaluation.taal.state_readout_probe import StateProjector
from evaluation.taal.trace_export import TaalTraceExporter
from evaluation.taal.state_audit import clone_memory_states
from tests import test_taal_state_audit as audit_tests


class RetentionTests(unittest.TestCase):
    def test_readers_and_dual_ridge_use_train_statistics(self):
        torch.manual_seed(11)
        x = torch.randn(64, 200)
        y = (x[:, 0] > 0).long()
        for kind in ("ridge", "mlp"):
            _, reader = fit_reader(x[:48], y[:48], x[48:], y[48:], classes=2,
                kind=kind, strengths=[0.01], seed=3, epochs=5)
            torch.testing.assert_close(reader["mean"], x[:48].mean(0))
            self.assertEqual(predict(reader, x[48:]).shape, (16,))
            self.assertTrue(torch.isfinite(reader["std"]).all())

    def test_fixed_inputs_replay_normal_parity_and_frozen_state(self):
        torch.manual_seed(19)
        model = TaalQwen3ForCausalLM(audit_tests.TaalStateAuditTests.config()).eval()
        with torch.inference_mode():
            state = model.prefill(torch.tensor([[1, 2, 3]])).state
            before = clone_memory_states(state.memory_states)
            original = model.model.layers[0].taal.neural_memory._update
            replay = FixedInputReplay(model, state.memory_states, start=3)
            with replay:
                output = model.prefill(torch.tensor([[4, 5, 6]]), state=state, prepend_memory_tokens=False)
            self.assertEqual(model.model.layers[0].taal.neural_memory._update, original)
            for name, value in output.state.memory_states[0].weights.items():
                torch.testing.assert_close(replay.states["normal"][0].weights[name], value, rtol=0, atol=0)
                torch.testing.assert_close(replay.states["frozen"][0].weights[name], before[0].weights[name], rtol=0, atol=0)
                torch.testing.assert_close(replay.states["frozen"][0].momentum[name], before[0].momentum[name], rtol=0, atol=0)
            self.assertEqual(replay.positions[0], 6)
            self.assertEqual(len(replay.writes("no_fresh")), 3)
            self.assertTrue(all(not event.write_enabled and event.proposed_write_norm == 0
                                for event in replay.writes("no_fresh")))
            self.assertTrue(all(event.net_weight_change_norm == 0 and not event.updates_enabled
                                for event in replay.writes("frozen")))

    def test_replay_restores_methods_after_exception(self):
        model = TaalQwen3ForCausalLM(audit_tests.TaalStateAuditTests.config()).eval()
        states = model.model.initial_memory_states(batch_size=1)
        memory = model.model.layers[0].taal.neural_memory
        original = memory._update
        with self.assertRaisesRegex(RuntimeError, "test"):
            with FixedInputReplay(model, states, start=0):
                raise RuntimeError("test")
        self.assertEqual(memory._update, original)

    def test_competing_data_and_end_to_end_trace_export(self):
        from transformers import AutoTokenizer
        from data.taal_conflict_control import MODEL_ID, MODEL_REVISION
        from data.taal_interference import build_interference_examples
        from evaluation.taal.lifecycle import lifecycle_boundaries
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
        rows = build_interference_examples(tokenizer)
        self.assertEqual(len(rows), 384)
        for neutral, competing in zip(rows[::2], rows[1::2]):
            self.assertEqual(len(neutral["input_ids"]), len(competing["input_ids"]))
            boundary = lifecycle_boundaries(competing, tokenizer)
            self.assertLessEqual(competing["distractor_end"], boundary["final"] - 128)
            self.assertEqual(neutral["input_ids"][boundary["final"]:], competing["input_ids"][boundary["final"]:])
            self.assertNotEqual(competing["target_token_id"], competing["distractor_token_id"])
        config = audit_tests.TaalStateAuditTests.config()
        config.vocab_size = len(tokenizer)
        config.working_memory_size = 128
        model = TaalQwen3ForCausalLM(config).eval()
        projector = StateProjector(model.model.initial_memory_states(batch_size=1), dimension=4, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exporter = TaalTraceExporter(root / "memory_traces", run_metadata={"checkpoint": "test", "tokenizer": MODEL_ID})
            store = PrefixStore(root / "prefixes", {"test": True})
            settings = {"execution_block_size": 128, "policies": ["normal", "no_fresh", "frozen"]}
            features = collect(model, tokenizer, rows[1], root, exporter, projector, settings, store)
            self.assertEqual(len(exporter.manifest["episodes"]), 6)
            self.assertIn("frozen/final", features)
            # An interrupted collection can replay safely without duplicate scores.
            collect(model, tokenizer, rows[1], root, exporter, projector, settings, store)
            self.assertEqual(len(exporter.manifest["episodes"]), 6)


if __name__ == "__main__":
    unittest.main()
