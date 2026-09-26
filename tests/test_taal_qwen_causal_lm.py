import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from architectures.taal.qwen.causal_lm import (
    TaalQwen3BridgeMemoryOutput,
    TaalQwen3CausalLMOutput,
    TaalQwen3ForCausalLM,
)
from architectures.taal.qwen.configuration import TaalQwen3Config


class TaalQwen3CausalLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.ids = torch.randint(0, 40, (1, 11))

    def config(self, **kwargs):
        options = dict(
            vocab_size=40,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_dropout=0.0,
            working_memory_size=4,
            max_persistent_kv_tokens=2,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=2,
            memory_chunk_size=1,
            memory_initial_forget=0.01,
            memory_initial_momentum=0.9,
            memory_initial_write_strength=0.1,
            num_persistent_tokens=2,
            eos_token_id=39,
        )
        options.update(kwargs)
        config = TaalQwen3Config(**options)
        config._attn_implementation = "sdpa"
        return config

    @staticmethod
    def assert_memory_state_equal(actual, expected):
        for collection in ("weights", "momentum"):
            actual_values = getattr(actual, collection)
            expected_values = getattr(expected, collection)
            for name in actual_values:
                torch.testing.assert_close(actual_values[name], expected_values[name])
        for name in (
            "query_conv_history",
            "key_conv_history",
            "value_conv_history",
        ):
            actual_history = getattr(actual, name)
            expected_history = getattr(expected, name)
            if actual_history is None or expected_history is None:
                if actual_history is not expected_history:
                    raise AssertionError(f"{name} differs")
            else:
                torch.testing.assert_close(actual_history, expected_history)
        self_pending = actual.pending_gradient
        other_pending = expected.pending_gradient
        if self_pending is None or other_pending is None:
            if self_pending is not other_pending:
                raise AssertionError("pending gradients differ")
        else:
            for name in self_pending:
                torch.testing.assert_close(self_pending[name], other_pending[name])
        if actual.pending_input_sum is None or expected.pending_input_sum is None:
            if actual.pending_input_sum is not expected.pending_input_sum:
                raise AssertionError("pending input sums differ")
        else:
            torch.testing.assert_close(
                actual.pending_input_sum,
                expected.pending_input_sum,
            )
        if actual.pending_count != expected.pending_count:
            raise AssertionError("pending counts differ")

    def assert_session_state_equal(self, actual, expected):
        self.assertEqual(actual.tokens_seen, expected.tokens_seen)
        self.assertEqual(set(actual.memory_states), set(expected.memory_states))
        for layer_index in actual.memory_states:
            self.assert_memory_state_equal(
                actual.memory_states[layer_index],
                expected.memory_states[layer_index],
            )
        for actual_layer, expected_layer in zip(
            actual.past_key_values.layers,
            expected.past_key_values.layers,
        ):
            torch.testing.assert_close(actual_layer.keys, expected_layer.keys)
            torch.testing.assert_close(actual_layer.values, expected_layer.values)
            torch.testing.assert_close(actual_layer.positions, expected_layer.positions)
            torch.testing.assert_close(
                actual_layer.is_persistent,
                expected_layer.is_persistent,
            )

    @staticmethod
    def memory_state_shapes(state):
        shapes = {}
        for layer_index, memory_state in state.memory_states.items():
            layer_shapes = {}
            for collection_name in ("weights", "momentum", "pending_gradient"):
                collection = getattr(memory_state, collection_name)
                layer_shapes[collection_name] = (
                    None
                    if collection is None
                    else {
                        name: tuple(value.shape)
                        for name, value in collection.items()
                    }
                )
            for tensor_name in (
                "pending_input_sum",
                "query_conv_history",
                "key_conv_history",
                "value_conv_history",
            ):
                value = getattr(memory_state, tensor_name)
                layer_shapes[tensor_name] = (
                    None if value is None else tuple(value.shape)
                )
            shapes[layer_index] = layer_shapes
        return shapes

    def test_logits_shifted_and_bridge_losses(self):
        model = TaalQwen3ForCausalLM(self.config()).train()
        labels = self.ids.clone()
        labels[:, 3] = -100
        result = model(
            self.ids,
            labels=labels,
            logits_to_keep=2,
            output_hidden_states=True,
        )
        self.assertIsInstance(result, TaalQwen3CausalLMOutput)
        logits = model.lm_head(result.hidden_states[-1]).float()
        expected = F.cross_entropy(
            logits[:, :-1].reshape(-1, 40),
            labels[:, 1:].reshape(-1),
        )
        torch.testing.assert_close(result.loss, expected)
        self.assertEqual(result.logits.shape, (1, 2, 40))

        delayed = torch.full_like(labels, -100)
        delayed[:, -1] = labels[:, -1]
        bridge = model.forward_bridge_memory(
            self.ids,
            labels=labels,
            delayed_labels=delayed,
            all_token_loss_weight=1.0,
            delayed_answer_loss_weight=2.0,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        self.assertIsInstance(bridge, TaalQwen3BridgeMemoryOutput)
        bridge_logits = model.lm_head(bridge.hidden_states[-1]).float()
        all_expected = F.cross_entropy(
            bridge_logits[:, :-1].reshape(-1, 40),
            labels[:, 1:].reshape(-1),
        )
        delayed_expected = F.cross_entropy(
            bridge_logits[:, :-1].reshape(-1, 40),
            delayed[:, 1:].reshape(-1),
        )
        torch.testing.assert_close(bridge.all_token_loss, all_expected)
        torch.testing.assert_close(bridge.delayed_answer_loss, delayed_expected)
        torch.testing.assert_close(
            bridge.loss,
            all_expected + 2 * delayed_expected,
        )

    def test_read_ablation_preserves_writes_but_changes_logits(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        with torch.no_grad():
            model.model.layers[0].taal.residual_gate.fill_(1.0)
        with torch.inference_mode():
            reads_off = model(self.ids, memory_read_scale=0.0)
            reads_on = model(self.ids, memory_read_scale=1.0)

        self.assert_memory_state_equal(
            reads_off.state.memory_states[0],
            reads_on.state.memory_states[0],
        )
        self.assertFalse(torch.equal(reads_off.logits, reads_on.logits))

    def test_delayed_loss_reaches_frozen_taal_path(self):
        model = TaalQwen3ForCausalLM(self.config()).train()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(".taal." in name)
        with torch.no_grad():
            model.model.layers[0].taal.residual_gate.fill_(0.25)
        delayed = torch.full_like(self.ids, -100)
        delayed[:, -1] = self.ids[:, -1]

        output = model.forward_bridge_memory(
            self.ids,
            delayed_labels=delayed,
            all_token_loss_weight=0.0,
            delayed_answer_loss_weight=1.0,
        )
        output.loss.backward()

        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(".taal." in name for name in trainable))
        self.assertTrue(all(parameter.grad is not None for parameter in trainable.values()))
        self.assertGreater(
            trainable["model.layers.0.taal.memory_projection_out.weight"]
            .grad.abs()
            .sum(),
            0,
        )
        self.assertGreater(
            trainable["model.layers.0.taal.neural_memory.memory_mlp.layers.0.weight"]
            .grad.abs()
            .sum(),
            0,
        )

    @torch.inference_mode()
    def test_blocked_prefill_inserts_learned_prefix_once(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        model.model.layers[0].taal.residual_gate.fill_(0.75)
        persistent_mask = torch.arange(self.ids.shape[1]) < 2
        expected = model(
            self.ids,
            use_cache=True,
            logits_to_keep=1,
            persistent_mask=persistent_mask,
            prepend_memory_tokens=True,
        )
        actual = model.prefill(
            self.ids,
            execution_block_size=3,
            persistent_mask=persistent_mask,
            prepend_memory_tokens=True,
        )

        torch.testing.assert_close(actual.logits, expected.logits)
        self.assert_memory_state_equal(
            actual.state.memory_states[0],
            expected.state.memory_states[0],
        )
        layer = actual.state.past_key_values.layers[0]
        torch.testing.assert_close(layer.positions, torch.tensor([0, 1, 8, 9, 10]))
        torch.testing.assert_close(
            layer.is_persistent,
            torch.tensor([True, True, False, False, False]),
        )

    @torch.inference_mode()
    def test_long_session_keeps_kv_and_memory_state_bounded(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        state = None
        memory_shapes = None
        for turn in range(20):
            segment = torch.randint(0, 40, (1, 7))
            persistent_mask = (
                torch.arange(7) < 2 if turn == 0 else None
            )
            output = model.prefill(
                segment,
                execution_block_size=3,
                state=state,
                persistent_mask=persistent_mask,
            )
            state = output.state
            for layer in state.past_key_values.layers:
                self.assertLessEqual(
                    layer.keys.shape[-2],
                    model.config.working_memory_size - 1 + 2,
                )
                self.assertEqual(layer.keys.shape, layer.values.shape)

            current_shapes = self.memory_state_shapes(state)
            if memory_shapes is None:
                memory_shapes = current_shapes
            else:
                self.assertEqual(current_shapes, memory_shapes)
        self.assertEqual(state.tokens_seen, 140)

    @torch.inference_mode()
    def test_irregular_prefill_splits_preserve_partial_memory_chunks(self):
        model = TaalQwen3ForCausalLM(
            self.config(
                memory_chunk_size=4,
                memory_conv_kernel_size=3,
            )
        ).eval()
        model.model.layers[0].taal.residual_gate.fill_(0.75)
        expected = model(
            self.ids,
            use_cache=True,
            logits_to_keep=1,
            prepend_memory_tokens=True,
        )
        actual = model.prefill(
            self.ids,
            execution_block_size=3,
            prepend_memory_tokens=True,
        )

        torch.testing.assert_close(actual.logits, expected.logits)
        self.assert_session_state_equal(actual.state, expected.state)
        self.assertEqual(actual.state.memory_states[0].pending_count, 1)

    @torch.inference_mode()
    def test_every_layer_prepends_once_per_utterance_not_per_execution_block(self):
        model = TaalQwen3ForCausalLM(
            self.config(num_hidden_layers=2)
        ).eval()
        input_lengths = {layer_idx: [] for layer_idx in range(2)}
        handles = []
        for layer_idx, decoder_layer in enumerate(model.model.layers):
            def record_length(module, inputs, layer_idx=layer_idx):
                _, sequence_length, _ = inputs[0].shape
                input_lengths[layer_idx].append(sequence_length)

            handles.append(
                decoder_layer.taal.neural_memory.register_forward_pre_hook(
                    record_length
                )
            )
        try:
            first = model.prefill(
                self.ids[:, :5],
                execution_block_size=3,
            )
            second = model.prefill(
                self.ids[:, 5:8],
                execution_block_size=2,
                state=first.state,
            )
            model(
                self.ids[:, 8:9],
                state=second.state,
                use_cache=True,
                prepend_memory_tokens=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        # N_persistent=2. Each utterance's first execution block receives the
        # learned prefix at every transformer layer; later blocks and decode do not.
        for layer_lengths in input_lengths.values():
            self.assertEqual(layer_lengths, [5, 2, 4, 1, 1])

    @torch.inference_mode()
    def test_generation_matches_manual_stateful_decode(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        model.model.layers[0].taal.residual_gate.fill_(0.5)
        prompt = self.ids[:, :7]
        generated = model.generate(
            prompt,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=[],
            execution_block_size=3,
        )

        manual = model.prefill(prompt, execution_block_size=3)
        expected = []
        for _ in range(3):
            token = manual.logits[:, -1].argmax(-1, keepdim=True)
            expected.append(token)
            manual = model(
                token,
                state=manual.state,
                use_cache=True,
                logits_to_keep=1,
                prepend_memory_tokens=False,
            )
        torch.testing.assert_close(generated.token_ids, torch.cat(expected, dim=1))
        self.assertEqual(generated.state.tokens_seen, 10)
        self.assert_memory_state_equal(
            generated.state.memory_states[0],
            manual.state.memory_states[0],
        )

    def test_configured_update_controls_survive_model_initialization(self):
        model = TaalQwen3ForCausalLM(self.config())
        memory = model.model.layers[0].taal.neural_memory
        self.assertAlmostEqual(memory.forget_projection.bias.sigmoid().item(), 0.01)
        self.assertAlmostEqual(memory.momentum_projection.bias.sigmoid().item(), 0.9)
        self.assertAlmostEqual(
            memory.write_strength_projection.bias.sigmoid().item(),
            0.1,
        )

    def test_loads_native_qwen_causal_lm_weights(self):
        config = self.config()
        native = Qwen3ForCausalLM(Qwen3Config.from_dict(config.to_dict())).eval()
        with tempfile.TemporaryDirectory() as folder:
            native.save_pretrained(folder)
            loaded, info = TaalQwen3ForCausalLM.from_pretrained(
                folder,
                config=config,
                output_loading_info=True,
            )
        self.assertEqual(info["unexpected_keys"], [])
        self.assertTrue(info["missing_keys"])
        self.assertTrue(all(".taal." in name for name in info["missing_keys"]))
        for name, value in native.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], value)
        persistent = loaded.model.layers[0].taal.persistent_tokens
        self.assertTrue(torch.isfinite(persistent).all())
        self.assertGreater(torch.count_nonzero(persistent).item(), 0)
        self.assertLess(persistent.abs().max().item(), 0.2)

        # The host loader must initialize this standalone parameter, even if
        # the constructor's nn.init call was suppressed during from_pretrained.
        with torch.no_grad():
            persistent.fill_(1e30)
        loaded._init_weights(loaded.model.layers[0].taal)
        self.assertLess(persistent.abs().max().item(), 0.2)


if __name__ == "__main__":
    unittest.main()
