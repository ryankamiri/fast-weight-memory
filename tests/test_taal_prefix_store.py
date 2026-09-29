import copy
from dataclasses import fields
import gzip
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.taal import state_audit
from evaluation.taal.prefix_store import PrefixStore, PrefixTrace, checkpoint_fingerprint
from evaluation.taal.state_audit import AUDIT_CONDITIONS, fork_session_state
from evaluation.taal.trace_export import TaalTraceRecorder


class TaalPrefixStoreTests(unittest.TestCase):
    @staticmethod
    def model(chunk_size=1):
        config = TaalQwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, head_dim=4, attention_dropout=0.0,
            working_memory_size=4, max_persistent_kv_tokens=2,
            memory_dim=8, memory_depth=2, memory_conv_kernel_size=2,
            memory_chunk_size=chunk_size, num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).eval()
        with torch.no_grad():
            for layer in model.model.layers:
                layer.taal.residual_gate.fill_(0.7)
        return model

    @staticmethod
    def example(index=0):
        return {
            "example_id": f"episode/{index}", "fact_id": index,
            "condition": "no_bridge", "query_variant": "exact",
            "input_ids": [1 + index, 2, 3, 4, 5, 6, 7],
            "final_query_position": 5, "target_token_id": 8 + index,
            "candidate_token_ids": [8, 9, 10],
        }

    def assert_memory_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for layer in expected:
            for field in fields(expected[layer]):
                left = getattr(expected[layer], field.name)
                right = getattr(actual[layer], field.name)
                if isinstance(left, dict):
                    self.assertEqual(set(left), set(right))
                    for name in left:
                        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
                elif isinstance(left, torch.Tensor):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                else:
                    self.assertEqual(left, right)

    @torch.inference_mode()
    def test_restore_preserves_all_six_condition_scores_and_prefix_trace(self):
        torch.manual_seed(40)
        model = self.model()
        example = self.example()
        recorder = TaalTraceRecorder(model, layers=[0, 1], token_count=5)
        with recorder:
            base = model.prefill(
                torch.tensor([example["input_ids"][:5]]), execution_block_size=3,
                persistent_mask=torch.tensor([True, False, False, False, False]),
            ).state
        other = model.prefill(torch.tensor([[9, 8, 7, 6, 5]]), execution_block_size=3).state
        trace = PrefixTrace(recorder.writes, recorder.reads, recorder.internal_prefixes)
        with TemporaryDirectory() as directory:
            store = PrefixStore(Path(directory), {"checkpoint": "test"})
            store.save(example, base, trace)
            restored = store.load(example, "cpu")
            self.assertEqual(restored.trace, trace)
            self.assertEqual(restored.state.tokens_seen, base.tokens_seen)
            self.assert_memory_equal(base.memory_states, restored.state.memory_states)
            torch.testing.assert_close(
                restored.state.past_key_values.persistent_positions,
                base.past_key_values.persistent_positions, atol=0, rtol=0,
            )
            for old, new in zip(base.past_key_values.layers, restored.state.past_key_values.layers):
                self.assertEqual(old.cumulative_length, new.cumulative_length)
                self.assertEqual(old.is_initialized, new.is_initialized)
                self.assertEqual(old.device, new.device)
                self.assertEqual(old.dtype, new.dtype)
                for name in ("keys", "values", "positions", "is_persistent"):
                    torch.testing.assert_close(getattr(old, name), getattr(new, name), atol=0, rtol=0)
            sources = {
                "correct": base.memory_states,
                "reset": model.model.initial_memory_states(batch_size=1),
                "zeroed": model.model.zero_memory_states(batch_size=1),
                "swapped": other.memory_states,
            }
            for condition in AUDIT_CONDITIONS:
                with self.subTest(condition=condition.name):
                    expected = model.prefill(
                        torch.tensor([[6, 7]]), execution_block_size=2,
                        state=fork_session_state(base, sources[condition.memory_source]),
                        memory_read_scale=condition.read_scale, prepend_memory_tokens=False,
                    )
                    restored_source = (
                        restored.state.memory_states if condition.memory_source == "correct"
                        else sources[condition.memory_source]
                    )
                    actual = model.prefill(
                        torch.tensor([[6, 7]]), execution_block_size=2,
                        state=fork_session_state(restored.state, restored_source),
                        memory_read_scale=condition.read_scale, prepend_memory_tokens=False,
                    )
                    torch.testing.assert_close(expected.logits, actual.logits, atol=0, rtol=0)
                    self.assert_memory_equal(expected.state.memory_states, actual.state.memory_states)
            # Loads are independent: an intervention cannot mutate the disk snapshot.
            restored.state.memory_states[0].weights["layers.0.weight"].zero_()
            untouched = store.load_memory(example, "cpu")
            self.assert_memory_equal(base.memory_states, untouched)

    @torch.inference_mode()
    def test_batched_prefixes_restore_individual_sessions(self):
        model = self.model()
        base = model.prefill(
            torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), execution_block_size=3,
        ).state
        with TemporaryDirectory() as directory:
            store = PrefixStore(Path(directory), {"checkpoint": "test"})
            for index in (0, 1):
                store.save(self.example(index), base, None, batch_index=index)
            expected = model.prefill(
                torch.tensor([[6, 7], [7, 6]]), execution_block_size=2,
                state=copy.deepcopy(base), prepend_memory_tokens=False,
            )
            for index, query in enumerate(([6, 7], [7, 6])):
                restored = store.load(self.example(index), "cpu")
                self.assertIsNone(restored.trace)
                for old, new in zip(base.past_key_values.layers, restored.state.past_key_values.layers):
                    torch.testing.assert_close(old.keys[index:index + 1], new.keys, atol=0, rtol=0)
                actual = model.prefill(
                    torch.tensor([query]), execution_block_size=2,
                    state=restored.state, prepend_memory_tokens=False,
                )
                torch.testing.assert_close(expected.logits[index:index + 1], actual.logits)

    @torch.inference_mode()
    def test_pending_writes_and_histories_survive_round_trip(self):
        model = self.model(chunk_size=3)
        base = model.prefill(torch.tensor([[1, 2]]), execution_block_size=2).state
        self.assertEqual(base.memory_states[0].pending_count, 1)
        with TemporaryDirectory() as directory:
            store = PrefixStore(Path(directory), {"checkpoint": "test"})
            store.save(self.example(), base, None)
            restored = store.load(self.example(), "cpu").state
            self.assert_memory_equal(base.memory_states, restored.memory_states)
            expected = model.prefill(
                torch.tensor([[3, 4]]), execution_block_size=2,
                state=copy.deepcopy(base), prepend_memory_tokens=False,
            )
            actual = model.prefill(
                torch.tensor([[3, 4]]), execution_block_size=2,
                state=restored, prepend_memory_tokens=False,
            )
            torch.testing.assert_close(expected.logits, actual.logits, atol=0, rtol=0)

    def test_snapshot_identity_rejects_changed_checkpoint_or_settings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            PrefixStore(path, {"checkpoint_sha256": "old", "save_traces": True})
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                PrefixStore(path, {"checkpoint_sha256": "new", "save_traces": True})
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                PrefixStore(path, {"checkpoint_sha256": "old", "save_traces": False})
            store = PrefixStore(path, {"checkpoint_sha256": "old", "save_traces": True})
            changed = self.example()
            changed["input_ids"] = [9, 8, 7]
            self.assertNotEqual(store.path(changed), store.path(self.example()))
            self.assertFalse(store.contains(changed))

    @torch.inference_mode()
    def test_interrupted_save_does_not_publish_an_incomplete_snapshot(self):
        model = self.model()
        base = model.prefill(torch.tensor([[1, 2]]), execution_block_size=2).state
        with TemporaryDirectory() as directory:
            store = PrefixStore(Path(directory), {"checkpoint": "test"})
            with patch("evaluation.taal.prefix_store.torch.save", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    store.save(self.example(), base, None)
            self.assertFalse(store.contains(self.example()))
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
            store.save(self.example(), base, None)
            with patch("evaluation.taal.prefix_store.torch.save", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    store.save(self.example(), base, None)
            self.assert_memory_equal(base.memory_states, store.load_memory(self.example(), "cpu"))

    def test_fingerprint_detects_replaced_weights_at_same_path(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            self.model().save_pretrained(path)
            before = checkpoint_fingerprint(path)
            weights = path / "model.safetensors"
            with weights.open("ab") as file:
                file.write(b"changed")
            self.assertNotEqual(before, checkpoint_fingerprint(path))

    def test_evaluator_resume_reuses_prefixes_and_exports_saved_traces(self):
        model = self.model()
        examples = [self.example(0), self.example(1), self.example(2)]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            model.save_pretrained(checkpoint)
            config = root / "audit.yaml"
            config.write_text(yaml.safe_dump({
                "dataset": {"repo_id": "test", "variant": "test", "revision": "pinned"},
                "seed": 42, "save_traces": True, "execution_block_size": 3,
            }))
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"tokenizer": "test", "tokenizer_revision": "pinned"}))
            output = root / "output"
            tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids[0]))
            # Exercise the real evaluator on a tiny CPU model; only external
            # data/model loading and the launch-time CUDA requirement are mocked.
            cpu_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test CPU"),
                device=lambda *_: torch.device("cpu"), tensor=torch.tensor,
                long=torch.long, logsumexp=torch.logsumexp,
            )
            arguments = ["state_audit", "--config", str(config), "--checkpoint", str(checkpoint), "--output-dir", str(output)]
            with (
                patch("sys.argv", arguments),
                patch.object(state_audit, "torch", cpu_torch),
                patch.object(state_audit, "hf_hub_download", return_value=str(metadata)),
                patch.object(state_audit, "load_dataset", return_value=examples),
                patch.object(state_audit, "load_model", return_value=model),
                patch.object(state_audit.AutoTokenizer, "from_pretrained", return_value=tokenizer),
                patch.dict("os.environ", {"TAAL_TIMING_FIRST_BATCH": "0"}),
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                original_save = PrefixStore.save

                def interrupted_save(store, *args, **kwargs):
                    original_save(store, *args, **kwargs)
                    raise RuntimeError("stop after first complete prefix")

                with patch.object(PrefixStore, "save", interrupted_save):
                    with self.assertRaisesRegex(RuntimeError, "stop after first"):
                        state_audit.main()
                self.assertEqual(len(list((output / "prefix_states").glob("*.pt"))), 1)
                original_prefill = model.prefill

                def observe_prefill(*args, **kwargs):
                    if kwargs.get("state") is None and args[0][0, 0].item() == 3:
                        # Episode 0 must already be scored before computing
                        # episode 2's prefix. Do not front-load the whole bank.
                        saved_rows = (output / "results.jsonl").read_text().splitlines()
                        self.assertEqual(len(saved_rows), 6)
                    return original_prefill(*args, **kwargs)

                with patch.object(model, "prefill", side_effect=observe_prefill) as prefill:
                    state_audit.main()
                prefix_calls = [call for call in prefill.call_args_list if call.kwargs.get("state") is None]
                query_calls = [call for call in prefill.call_args_list if call.kwargs.get("state") is not None]
                self.assertEqual(len(prefix_calls), 2)  # Only the previously unfinished episodes.
                self.assertEqual(len(query_calls), 18)  # Six short suffixes per episode.
                rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 18)
                manifest = json.loads((output / "memory_traces" / "manifest.json").read_text())
                self.assertEqual(len(manifest["episodes"]), 18)
                trace_files = list((output / "memory_traces").rglob("*.json.gz"))
                episode_traces = []
                for path in trace_files:
                    with gzip.open(path, "rt") as file:
                        trace = json.load(file)
                    if "tokens" in trace:
                        episode_traces.append(trace)
                self.assertEqual(len(episode_traces), 18)
                for trace in episode_traces:
                    self.assertEqual(len(trace["reads"]), 14)
                    self.assertEqual(len(trace["internal_prefixes"]), 2)
                    self.assertEqual({event["position"] for event in trace["reads"]}, set(range(7)))
                with patch.object(model, "prefill", side_effect=AssertionError("already finished")):
                    state_audit.main()
                # An interrupted query phase also resumes from disk without a
                # long prefix pass. Leave a non-swapped condition unfinished.
                missing_id = "episode/1/reads_disabled"
                missing = next(row for row in rows if row["evaluation_id"] == missing_id)
                (output / "results.jsonl").write_text("".join(
                    json.dumps(row) + "\n" for row in rows
                    if row["evaluation_id"] != missing_id
                ))
                with patch.object(model, "prefill", wraps=model.prefill) as resumed:
                    state_audit.main()
                self.assertEqual(resumed.call_count, 1)
                self.assertIsNotNone(resumed.call_args.kwargs.get("state"))
                final_rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
                actual = next(row for row in final_rows if row["evaluation_id"] == missing_id)
                self.assertEqual(actual["target_log_probability"], missing["target_log_probability"])
                self.assertEqual(len(final_rows), 18)


if __name__ == "__main__":
    unittest.main()
