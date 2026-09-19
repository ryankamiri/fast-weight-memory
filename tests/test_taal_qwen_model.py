import copy
import tempfile
import unittest

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.taal.qwen.model import TaalQwen3Model


class TaalQwen3ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.input_ids = torch.randint(0, 40, (2, 9))

    def config(self, **kwargs):
        options = dict(
            vocab_size=40,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_dropout=0.0,
            working_memory_size=4,
            max_persistent_kv_tokens=4,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=1,
            memory_chunk_size=1,
            num_persistent_tokens=2,
        )
        options.update(kwargs)
        config = TaalQwen3Config(**options)
        config._attn_implementation = "sdpa"
        return config

    def test_zero_gate_matches_standard_sliding_qwen(self):
        config = self.config()
        model = TaalQwen3Model(config).eval()
        base_config = Qwen3Config.from_dict(config.to_dict())
        base_config.use_sliding_window = True
        base_config.sliding_window = config.working_memory_size
        base_config.layer_types = ["sliding_attention"] * config.num_hidden_layers
        base_config._attn_implementation = "sdpa"
        base = Qwen3Model(base_config).eval()
        base.load_state_dict(
            {
                name: value
                for name, value in model.state_dict().items()
                if ".taal." not in name
            },
            strict=True,
        )

        with torch.no_grad():
            actual = model(self.input_ids, output_hidden_states=True)
            expected = base(
                self.input_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        torch.testing.assert_close(
            actual.last_hidden_state,
            expected.last_hidden_state,
            atol=0,
            rtol=0,
        )
        self.assertEqual(set(actual.state.memory_states), {0, 1})
        self.assertEqual(actual.state.tokens_seen, self.input_ids.shape[1])
        self.assertEqual(len(actual.hidden_states), 3)
        for found, wanted in zip(actual.hidden_states, expected.hidden_states):
            torch.testing.assert_close(found, wanted, atol=0, rtol=0)

    def test_cached_calls_preserve_qwen_outputs_and_session_state(self):
        model = TaalQwen3Model(self.config()).eval()
        with torch.no_grad():
            expected = model(self.input_ids)
            state = None
            outputs = []
            for start, end in ((0, 3), (3, 5), (5, 9)):
                result = model(
                    self.input_ids[:, start:end],
                    state=state,
                    use_cache=True,
                )
                state = result.state
                outputs.append(result.last_hidden_state)
                self.assertEqual(state.tokens_seen, end)
                self.assertEqual(set(state.memory_states), {0, 1})
                self.assertIs(result.past_key_values, state.past_key_values)
            torch.testing.assert_close(
                torch.cat(outputs, dim=1),
                expected.last_hidden_state,
                atol=2e-6,
                rtol=2e-5,
            )
            for layer in state.past_key_values.layers:
                self.assertEqual(layer.get_seq_length(), self.input_ids.shape[1])
                self.assertEqual(layer.keys.shape[-2], 3)

    def test_write_mask_changes_memory_without_masking_host_tokens(self):
        model = TaalQwen3Model(self.config()).eval()
        blocked = torch.zeros_like(self.input_ids, dtype=torch.bool)
        enabled = torch.ones_like(self.input_ids, dtype=torch.bool)
        with torch.no_grad():
            blocked_output = model(self.input_ids, write_mask=blocked)
            enabled_output = model(self.input_ids, write_mask=enabled)

        torch.testing.assert_close(
            blocked_output.last_hidden_state,
            enabled_output.last_hidden_state,
            atol=0,
            rtol=0,
        )
        differences = []
        for layer_idx in blocked_output.state.memory_states:
            blocked_state = blocked_output.state.memory_states[layer_idx]
            enabled_state = enabled_output.state.memory_states[layer_idx]
            differences.extend(
                not torch.equal(blocked_state.weights[name], enabled_state.weights[name])
                for name in blocked_state.weights
            )
        self.assertTrue(any(differences))

    def test_configuration_roundtrip_and_save_reload(self):
        config = self.config(memory_chunk_size=2)
        restored_config = TaalQwen3Config.from_dict(config.to_dict())
        self.assertEqual(restored_config.working_memory_size, 4)
        self.assertEqual(restored_config.max_persistent_kv_tokens, 4)
        self.assertEqual(restored_config.taal_layer_config().memory.chunk_size, 2)

        model = TaalQwen3Model(config).eval()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored = TaalQwen3Model.from_pretrained(folder).eval()
        self.assertIsInstance(restored.config, TaalQwen3Config)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value)

    def test_loads_native_qwen_backbone_weights(self):
        config = self.config()
        native_config = Qwen3Config.from_dict(config.to_dict())
        native = Qwen3Model(native_config)
        with tempfile.TemporaryDirectory() as folder:
            native.save_pretrained(folder)
            loaded, info = TaalQwen3Model.from_pretrained(
                folder,
                config=config,
                output_loading_info=True,
            )

        self.assertEqual(info["unexpected_keys"], [])
        self.assertTrue(info["missing_keys"])
        self.assertTrue(all(".taal." in name for name in info["missing_keys"]))
        for name, value in native.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value)

    def test_rejects_incomplete_continuation_state(self):
        model = TaalQwen3Model(self.config()).eval()
        with torch.no_grad():
            first = model(self.input_ids[:, :3])
        first.state.memory_states.pop(1)
        with self.assertRaisesRegex(ValueError, "every layer memory state"):
            model(self.input_ids[:, 3:4], state=first.state)

    def test_checkpointing_matches_outputs_state_and_gradients(self):
        base = TaalQwen3Model(self.config()).train()
        checked = copy.deepcopy(base)
        checked.gradient_checkpointing_enable()
        self.assertTrue(checked.is_gradient_checkpointing)
        persistent_mask = torch.arange(self.input_ids.shape[1]) < 2

        results = []
        for model in (base, checked):
            output = model(
                self.input_ids,
                persistent_mask=persistent_mask,
            )
            loss = output.last_hidden_state[..., 0].sum()
            for memory_state in output.state.memory_states.values():
                loss = loss + sum(
                    weight.square().sum()
                    for weight in memory_state.weights.values()
                )
            loss.backward()
            results.append(output)

        torch.testing.assert_close(
            results[0].last_hidden_state,
            results[1].last_hidden_state,
        )
        for layer_idx in results[0].state.memory_states:
            actual = results[0].state.memory_states[layer_idx]
            expected = results[1].state.memory_states[layer_idx]
            for name in actual.weights:
                torch.testing.assert_close(actual.weights[name], expected.weights[name])
        for (name, actual), (_, expected) in zip(
            base.named_parameters(),
            checked.named_parameters(),
        ):
            self.assertIsNotNone(actual.grad, name)
            self.assertIsNotNone(expected.grad, name)
            torch.testing.assert_close(
                actual.grad,
                expected.grad,
                atol=2e-6,
                rtol=2e-5,
                msg=name,
            )

    @torch.inference_mode()
    def test_persistent_system_kv_survives_blocked_prefill(self):
        persistent_mask = torch.zeros(9, dtype=torch.bool)
        persistent_mask[:2] = True
        for backend in ("eager", "sdpa"):
            config = self.config()
            config._attn_implementation = backend
            model = TaalQwen3Model(config).eval()
            reference = model(
                self.input_ids,
                persistent_mask=persistent_mask,
            )

            state = None
            block_outputs = []
            for start, end in ((0, 3), (3, 5), (5, 9)):
                output = model(
                    self.input_ids[:, start:end],
                    state=state,
                    use_cache=True,
                    persistent_mask=persistent_mask[start:end],
                )
                state = output.state
                block_outputs.append(output.last_hidden_state)

            torch.testing.assert_close(
                torch.cat(block_outputs, dim=1),
                reference.last_hidden_state,
                atol=2e-6,
                rtol=2e-5,
            )
            for layer in state.past_key_values.layers:
                torch.testing.assert_close(
                    layer.positions,
                    torch.tensor([0, 1, 6, 7, 8]),
                )
                torch.testing.assert_close(
                    layer.is_persistent,
                    torch.tensor([True, True, False, False, False]),
                )
                # TaaL's internal learned prefix never enters Qwen KV.
                self.assertEqual(layer.keys.shape[-2], 5)

    def test_rejects_invalid_or_exceeded_persistent_kv_budget(self):
        for invalid in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                self.config(max_persistent_kv_tokens=invalid)

        model = TaalQwen3Model(
            self.config(max_persistent_kv_tokens=1)
        ).eval()
        persistent_mask = torch.tensor([True, True] + [False] * 7)
        with self.assertRaisesRegex(
            ValueError,
            "max_persistent_kv_tokens=1",
        ):
            model(self.input_ids, persistent_mask=persistent_mask)


if __name__ == "__main__":
    unittest.main()
