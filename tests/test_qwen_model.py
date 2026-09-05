import copy
import tempfile
import unittest

import torch
from jaxtyping import TypeCheckError
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

from architectures.qwen.configuration import FWQwen3Config
from architectures.qwen.model import FWQwen3Model, FWQwen3ModelOutput
from architectures.states.model_state import FWModelState
from architectures.cache.sliding_window import SlidingWindowKVCache


class ModelTests(unittest.TestCase):
    def test_invalid_windows_are_validated_by_config(self):
        for teacher, student in ((None, 2), (2, 3), (0, 1), (3, -1)):
            with self.assertRaises(ValueError):
                FWQwen3Config(teacher_window_size=teacher, student_window_size=student)

    def setUp(self):
        torch.manual_seed(11)
        self.ids = torch.randint(0, 40, (2, 11))

    def config(self, fast_weight_layers=(0, 2), **kwargs):
        return FWQwen3Config(
            vocab_size=40, hidden_size=24, intermediate_size=32,
            num_hidden_layers=3, num_attention_heads=4,
            num_key_value_heads=2, head_dim=6,
            fast_weight_layers=list(fast_weight_layers),
            teacher_window_size=5, student_window_size=2, chunk_size=4,
            conv_kernel_size=3, attention_dropout=0.0, **kwargs,
        )

    def test_checkpointing_outputs_and_gradients(self):
        for fast_layers in ([], [0, 2]):
            base = FWQwen3Model(self.config(fast_weight_layers=fast_layers)).train()
            base.config.attention_dropout = 0.2
            for layer in base.layers:
                layer.self_attn.attention_dropout = 0.2
            checked = copy.deepcopy(base)
            checked.gradient_checkpointing_enable()
            self.assertTrue(checked.is_gradient_checkpointing)
            results = []
            for model in (base, checked):
                torch.manual_seed(123)
                result = model(self.ids, use_cache=False)
                loss = result.last_hidden_state[..., 0].sum()
                # Exercise tensors returned inside the dataclass, not only hidden states.
                for state in result.state.mlp_states.values():
                    loss = loss + state.W_fast.square().sum() + state.pending_r.square().sum()
                loss.backward()
                results.append(result)
            torch.testing.assert_close(results[0].last_hidden_state, results[1].last_hidden_state)
            self.assert_states_close(results[0].state, results[1].state)
            for (name, a), (_, b) in zip(base.named_parameters(), checked.named_parameters()):
                self.assertIsNotNone(a.grad, name)
                self.assertIsNotNone(b.grad, name)
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_checkpointing_gradients_through_carried_state(self):
        base = FWQwen3Model(self.config()).train()
        checked = copy.deepcopy(base)
        checked.gradient_checkpointing_enable()
        results = []
        for model in (base, checked):
            first = model(self.ids[:, :7], use_cache=False)
            for state in first.state.mlp_states.values():
                for tensor in vars(state).values():
                    if tensor is not None:
                        tensor.retain_grad()
            second = model(self.ids[:, 7:], state=first.state, use_cache=False)
            # Only the second call contributes directly to the loss.
            second.last_hidden_state[..., 0].sum().backward()
            results.append((first, second))
        self.assert_states_close(results[0][1].state, results[1][1].state)
        for i in base.config.fast_weight_layers:
            for name, tensor in vars(results[0][0].state.mlp_states[i]).items():
                other = getattr(results[1][0].state.mlp_states[i], name)
                self.assertIsNotNone(tensor.grad, name)
                self.assertIsNotNone(other.grad, name)
                torch.testing.assert_close(tensor.grad, other.grad, atol=2e-6, rtol=2e-5)
        for (name, a), (_, b) in zip(base.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(a.grad, name)
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_checkpointing_cache_handling_and_toggle(self):
        model = FWQwen3Model(self.config()).eval()
        prefill = model(self.ids[:, :3], use_cache=True)
        model.gradient_checkpointing_enable()
        # Enabled checkpointing does not change inference caching.
        with torch.no_grad():
            decoded = model(self.ids[:, 3:4], state=prefill.state, use_cache=True)
        self.assertEqual(decoded.past_key_values.get_seq_length(), 4)
        model.train()
        with self.assertRaisesRegex(ValueError, "checkpointed training"):
            model(self.ids[:, 4:5], state=decoded.state)
        self.assertEqual(decoded.past_key_values.get_seq_length(), 4)
        output = model(self.ids, use_cache=True)  # Checkpointed training overrides the request.
        self.assertIsNone(output.past_key_values)
        self.assertIsNone(output.state.past_key_values)
        model.gradient_checkpointing_disable()
        self.assertFalse(model.is_gradient_checkpointing)
        self.assertIsNotNone(model(self.ids, use_cache=True).past_key_values)

    def test_checkpointing_options_and_frozen_inputs(self):
        model = FWQwen3Model(self.config()).train()
        with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
            model.gradient_checkpointing_enable({"use_reentrant": True})
        self.assertFalse(model.is_gradient_checkpointing)
        options = {"preserve_rng_state": True}
        model.gradient_checkpointing_enable(options)
        self.assertEqual(options, {"preserve_rng_state": True})
        for param in model.parameters():
            param.requires_grad_(False)
        model.layers[0].mlp.W_proj.requires_grad_(True)
        output = model(self.ids, use_cache=False)
        output.last_hidden_state[..., 0].sum().backward()
        grad = model.layers[0].mlp.W_proj.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)

    def assert_states_close(self, actual, expected):
        self.assertEqual(actual.tokens_seen, expected.tokens_seen)
        self.assertEqual(set(actual.mlp_states), set(expected.mlp_states))
        for i in actual.mlp_states:
            for name in vars(actual.mlp_states[i]):
                torch.testing.assert_close(
                    getattr(actual.mlp_states[i], name), getattr(expected.mlp_states[i], name),
                    atol=2e-6, rtol=2e-5,
                )

    def test_ttcd_defaults_and_layer_selection(self):
        config = FWQwen3Config(num_hidden_layers=28)
        self.assertEqual(config.fast_weight_layers, [0, 7, 14, 21])
        layer_flags = [layer.is_fast_weight_layer for layer in FWQwen3Model(self.config()).layers]
        self.assertEqual(layer_flags, [True, False, True])
        for invalid in ([3], [-1], [0, 0], [True]):
            with self.assertRaises(ValueError):
                self.config(fast_weight_layers=invalid)

    def test_ordinary_layers_match_qwen_with_teacher_window_and_padding(self):
        for backend in ("eager", "sdpa"):
            config = self.config(
                fast_weight_layers=[], use_sliding_window=True, sliding_window=3,
                layer_types=["full_attention", "sliding_attention", "full_attention"],
            )
            config._attn_implementation = backend
            model = FWQwen3Model(config).eval()
            base_config = Qwen3Config.from_dict(config.to_dict())
            base_config.layer_types = ["sliding_attention"] * config.num_hidden_layers
            base_config.sliding_window = config.teacher_window_size
            base_config.use_sliding_window = True
            base_config._attn_implementation = backend
            base = Qwen3Model(base_config).eval()
            base.load_state_dict(model.state_dict(), strict=True)
            padding = torch.ones_like(self.ids, dtype=torch.bool)
            padding[0, :2] = 0
            kwargs = dict(attention_mask=padding, use_cache=False, output_hidden_states=True)
            actual = model(self.ids, **kwargs)
            expected = base(self.ids, **kwargs)
            valid = padding.bool()
            torch.testing.assert_close(actual.last_hidden_state[valid], expected.last_hidden_state[valid])
            self.assertEqual(len(actual.hidden_states), 4)
            self.assertEqual(actual.state.mlp_states, {})
            for a, b in zip(actual.hidden_states, expected.hidden_states):
                torch.testing.assert_close(a[valid], b[valid])

    def test_mask_oracle_query_length_independent_of_windows(self):
        model = FWQwen3Model(self.config())
        for start, S in ((0, 11), (7, 3), (10, 1)):
            x = torch.zeros(2, S, 24)
            q = torch.arange(start, start + S)
            k = torch.arange(start + S)
            masks = model._prepare_masks(None, x, q, k)
            for name, window in (("teacher", 5), ("student", 2)):
                expected = torch.full((1, 1, S, start + S), float("-inf"))
                for row, position in enumerate(q.tolist()):
                    left = 0 if window is None else max(0, position - window + 1)
                    expected[..., row, left:position + 1] = 0
                expected = expected.clamp_min(torch.finfo(x.dtype).min)
                torch.testing.assert_close(masks[name], expected)

    def test_all_layers_receive_the_teacher_window(self):
        for fast_layers in ([], [0, 2]):
            model = FWQwen3Model(self.config(fast_weight_layers=fast_layers)).eval()
            captured = []

            def capture(module, args, kwargs):
                captured.append((module.is_fast_weight_layer, kwargs["teacher_attention_mask"], kwargs["student_attention_mask"]))

            handles = [layer.register_forward_pre_hook(capture, with_kwargs=True) for layer in model.layers]
            try:
                model(self.ids)
            finally:
                for handle in handles:
                    handle.remove()
            q = torch.arange(11)[:, None]
            k = torch.arange(11)[None, :]
            expected_teacher = ((k <= q) & (k > q - 5))[None, None]
            expected_student = ((k <= q) & (k > q - 2))[None, None]
            self.assertEqual(len(captured), 3)
            for is_fast, teacher, student in captured:
                self.assertIs(teacher, captured[0][1])
                torch.testing.assert_close(teacher == 0, expected_teacher)
                if is_fast:
                    torch.testing.assert_close(student == 0, expected_student)
                else:
                    self.assertIsNone(student)

    def test_only_boolean_padding_masks_are_supported(self):
        model = FWQwen3Model(self.config())
        x = torch.zeros(2, 11, 24)
        positions = torch.arange(11)
        valid = torch.ones(2, 11, dtype=torch.bool)
        torch.testing.assert_close(
            model(self.ids, attention_mask=valid).last_hidden_state,
            model(self.ids).last_hidden_state,
        )
        for invalid in (valid.long(), valid.float(), valid[:, None, None, :]):
            with self.assertRaises(TypeCheckError):
                model(self.ids, attention_mask=invalid)
            with self.assertRaisesRegex(ValueError, "boolean"):
                model._prepare_masks(invalid, x, positions, positions)

    def test_full_training_vs_chunked_prefill_and_decode(self):
        model = FWQwen3Model(self.config()).eval()
        with torch.no_grad():
            for i in model.config.fast_weight_layers:
                model.layers[i].mlp.teacher_conv.weight.normal_(std=0.2)
                model.layers[i].mlp.student_conv.weight.normal_(std=0.2)
            expected = model(self.ids, use_cache=False)
            state = None
            parts = []
            for start, end in ((0, 3), (3, 7), (7, 10), (10, 11)):
                output = model(self.ids[:, start:end], state=state, use_cache=True)
                state = output.state
                parts.append(output.last_hidden_state)
                self.assertIs(output.past_key_values, state.past_key_values)
                for layer_idx in range(3):
                    self.assertEqual(state.past_key_values.get_seq_length(layer_idx), end)
            torch.testing.assert_close(torch.cat(parts, dim=1), expected.last_hidden_state, atol=2e-6, rtol=2e-5)
            self.assert_states_close(state, expected.state)

    def test_eviction_prefill_decode_masks_and_all_layer_states(self):
        for window in (1, 2, 5):
            config = self.config()
            config.teacher_window_size = window
            config.student_window_size = min(window, 2)
            model = FWQwen3Model(config).eval()
            ids = torch.randint(0, 40, (2, 23))
            with torch.no_grad():
                for i in config.fast_weight_layers:
                    model.layers[i].mlp.teacher_conv.weight.normal_(std=0.2)
                    model.layers[i].mlp.student_conv.weight.normal_(std=0.2)
                expected = model(ids, use_cache=False)
                full_cached = model(ids, use_cache=True)
                for ends in (list(range(1, 24)), [3, 4, 5, 6, 7, 8, 23], [9, 10, 11, 17, 18, 23]):
                    state = None
                    start = 0
                    parts = []
                    for end in ends:
                        offset = max(0, start - (window - 1))
                        captured = []

                        def capture(module, args, kwargs):
                            captured.append((kwargs["teacher_attention_mask"], kwargs["student_attention_mask"]))

                        handle = model.layers[0].register_forward_pre_hook(capture, with_kwargs=True)
                        try:
                            result = model(
                                ids[:, start:end], state=state, use_cache=True,
                                attention_mask=torch.ones(2, end - offset, dtype=torch.bool),
                            )
                        finally:
                            handle.remove()
                        q = torch.arange(start, end)[:, None]
                        k = torch.arange(offset, end)[None, :]
                        for mask, size in zip(captured[0], (window, config.student_window_size)):
                            allowed = ((q >= k) & (k > q - size))[None, None].expand(2, -1, -1, -1)
                            torch.testing.assert_close(mask == 0, allowed)
                        state = result.state
                        parts.append(result.last_hidden_state)
                        self.assertEqual(state.tokens_seen, end)
                        for i, layer in enumerate(state.past_key_values.layers):
                            self.assertEqual(layer.get_seq_length(), end)
                            self.assertEqual(layer.keys.shape[-2], min(end, window - 1))
                        for mlp_state in state.mlp_states.values():
                            self.assertEqual(mlp_state.pending_count, end % config.chunk_size)
                        start = end
                    torch.testing.assert_close(torch.cat(parts, dim=1), expected.last_hidden_state, atol=2e-6, rtol=2e-5)
                    self.assert_states_close(state, expected.state)
                    for actual_layer, expected_layer in zip(state.past_key_values.layers, full_cached.past_key_values.layers):
                        torch.testing.assert_close(actual_layer.keys, expected_layer.keys, atol=2e-6, rtol=2e-5)
                        torch.testing.assert_close(actual_layer.values, expected_layer.values, atol=2e-6, rtol=2e-5)

    def test_ordinary_padding_mask_tracks_retained_keys(self):
        model = FWQwen3Model(self.config(fast_weight_layers=[])).eval()
        padding = torch.ones_like(self.ids, dtype=torch.bool)
        padding[0, :2] = False
        with torch.no_grad():
            expected = model(self.ids, attention_mask=padding)
            first = model(self.ids[:, :7], attention_mask=padding[:, :7], use_cache=True)
            # Teacher window 5 retains positions 3-6 before reading positions 7-10.
            tail = model(self.ids[:, 7:], state=first.state, use_cache=True, attention_mask=padding[:, 3:])
            torch.testing.assert_close(tail.last_hidden_state, expected.last_hidden_state[:, 7:])

    def test_eviction_training_gradients_without_checkpointing(self):
        full_model = FWQwen3Model(self.config()).train()
        split_model = copy.deepcopy(full_model)
        full = full_model(self.ids)
        first = split_model(self.ids[:, :7], use_cache=True)
        tail = split_model(self.ids[:, 7:], state=first.state, use_cache=True)
        torch.testing.assert_close(tail.last_hidden_state, full.last_hidden_state[:, 7:])
        full.last_hidden_state[:, 7:, 0].sum().backward()
        tail.last_hidden_state[..., 0].sum().backward()
        for (name, a), (_, b) in zip(full_model.named_parameters(), split_model.named_parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5, msg=name)

    def test_reject_wrong_cache_geometry_and_old_full_history_mask(self):
        model = FWQwen3Model(self.config()).eval()
        for cache in (SlidingWindowKVCache(3, 2), SlidingWindowKVCache(1, 5)):
            with self.assertRaisesRegex(ValueError, "window and layer count"):
                model(self.ids, state=FWModelState(past_key_values=cache), use_cache=True)
        with torch.no_grad():
            first = model(self.ids[:, :7], use_cache=True)
            with self.assertRaisesRegex(ValueError, "retained keys"):
                model(self.ids[:, 7:8], state=first.state, use_cache=True,
                      attention_mask=torch.ones(2, 8, dtype=torch.bool))
            self.assertEqual(first.past_key_values.get_seq_length(), 7)

    def test_training_gradients_causality_and_state_lifetime(self):
        model = FWQwen3Model(self.config()).train()
        output = model(self.ids, use_cache=False)
        self.assertIsNone(output.state.past_key_values)
        self.assertEqual(output.state.tokens_seen, 11)
        output.last_hidden_state.square().mean().backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        changed = self.ids.clone()
        changed[:, 7:] = (changed[:, 7:] + 1) % 40
        altered = model(changed, use_cache=False)
        torch.testing.assert_close(output.last_hidden_state[:, :7], altered.last_hidden_state[:, :7])
        repeated = model(self.ids, use_cache=False)
        torch.testing.assert_close(repeated.last_hidden_state, output.last_hidden_state)
        self.assertFalse(any("W_fast" in key or "pending_" in key for key in model.state_dict()))

    def test_two_sessions_are_independent(self):
        model = FWQwen3Model(self.config()).eval()
        other_ids = (self.ids + 3) % 40
        with torch.no_grad():
            expected_a = model(self.ids, use_cache=False)
            expected_b = model(other_ids, use_cache=False)
            a = model(self.ids[:, :3], use_cache=True)
            b = model(other_ids[:, :6], use_cache=True)
            self.assertIsNot(a.past_key_values, b.past_key_values)
            tail_a = model(self.ids[:, 3:], state=a.state, use_cache=True)
            tail_b = model(other_ids[:, 6:], state=b.state, use_cache=True)
            torch.testing.assert_close(tail_a.last_hidden_state, expected_a.last_hidden_state[:, 3:])
            torch.testing.assert_close(tail_b.last_hidden_state, expected_b.last_hidden_state[:, 6:])

    def test_save_reload_and_base_checkpoint_initialization(self):
        config = self.config()
        model = FWQwen3Model(config).eval()
        # Ensure loading preserves trained FW parameters, not just their defaults.
        with torch.no_grad():
            model.layers[0].mlp.W_proj.normal_()
            model.layers[0].mlp.student_conv.weight.normal_()
        with tempfile.TemporaryDirectory() as folder:
            model.save_pretrained(folder)
            restored = FWQwen3Model.from_pretrained(folder)
            self.assertEqual(restored.config.fast_weight_layers, [0, 2])
            torch.testing.assert_close(
                model(self.ids, use_cache=False).last_hidden_state,
                restored(self.ids, use_cache=False).last_hidden_state,
            )
        base_config = Qwen3Config.from_dict(config.to_dict())
        base = Qwen3Model(base_config)
        with tempfile.TemporaryDirectory() as folder:
            base.save_pretrained(folder)
            loaded, info = FWQwen3Model.from_pretrained(folder, config=config, output_loading_info=True)
            self.assertEqual(info["unexpected_keys"], [])
            self.assertTrue(info["missing_keys"])
            for key, value in base.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)
            mlp = loaded.layers[0].mlp
            torch.testing.assert_close(mlp.W_proj, torch.eye(24))
            torch.testing.assert_close(mlp.beta_proj, torch.zeros(24))
            for conv in (mlp.teacher_conv, mlp.student_conv):
                torch.testing.assert_close(conv.weight[..., -1], torch.ones(32, 1))
                torch.testing.assert_close(conv.weight[..., :-1], torch.zeros(32, 1, 2))

    def test_simple_forward_api_and_rejected_states(self):
        model = FWQwen3Model(self.config()).eval()
        model.config._attn_implementation = "eager"
        model.config.output_hidden_states = True
        model.config.output_attentions = True
        model.config.return_dict = False
        actual = model(self.ids)
        self.assertIsInstance(actual, FWQwen3ModelOutput)
        self.assertIsInstance(actual.state, FWModelState)
        self.assertIsNone(actual.past_key_values)
        self.assertIsNone(actual.hidden_states)
        self.assertEqual(len(model(self.ids, output_hidden_states=True).hidden_states), 4)
        for removed in ("inputs_embeds", "position_ids", "cache_position", "output_attentions", "return_dict"):
            with self.assertRaises(TypeError):
                model(self.ids, **{removed: None})
        padding = torch.ones_like(self.ids, dtype=torch.bool)
        padding[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "Padded"):
            model(self.ids, attention_mask=padding)
        prefill = model(self.ids[:, :3], use_cache=True)
        with self.assertRaisesRegex(ValueError, "all its layer MLP states"):
            model(self.ids[:, 3:4], state=FWModelState(
                past_key_values=prefill.past_key_values, tokens_seen=3,
            ), use_cache=True)
        with self.assertRaisesRegex(ValueError, "use_cache=True"):
            model(self.ids[:, 3:4], state=prefill.state, use_cache=False)
        with self.assertRaises(TypeError):
            model(self.ids[:, 3:4], past_key_values=prefill.past_key_values)
        model(self.ids[:, 3:4], state=prefill.state, use_cache=True)
        with self.assertRaisesRegex(ValueError, "old states"):
            model(self.ids[:, 3:4], state=prefill.state, use_cache=True)

    def test_loading_backbone_from_causal_lm_checkpoint(self):
        config = self.config()
        base = Qwen3ForCausalLM(Qwen3Config.from_dict(config.to_dict()))
        with tempfile.TemporaryDirectory() as folder:
            base.save_pretrained(folder)
            loaded = FWQwen3Model.from_pretrained(folder, config=config)
            for key, value in base.model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[key], value)

    def test_low_precision_activations_keep_fp32_memory(self):
        model = FWQwen3Model(self.config()).to(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            output = model(self.ids[:, :3], use_cache=True)
            output = model(self.ids[:, 3:], state=output.state, use_cache=True)
            self.assertEqual(output.last_hidden_state.dtype, torch.bfloat16)
            for state in output.state.mlp_states.values():
                self.assertEqual(state.W_fast.dtype, torch.float32)
                self.assertEqual(state.pending_r.dtype, torch.float32)
                self.assertEqual(state.pending_k.dtype, torch.float32)
                self.assertEqual(state.teacher_conv_state.dtype, torch.bfloat16)

    def test_teacher_window_can_exceed_native_sliding_window(self):
        model = FWQwen3Model(self.config(
            use_sliding_window=True, sliding_window=3,
            layer_types=["sliding_attention"] * 3,
        )).eval()
        with torch.no_grad():
            expected = model(self.ids, use_cache=False)
            first = model(self.ids[:, :7], use_cache=True)
            tail = model(self.ids[:, 7:], state=first.state, use_cache=True)
            torch.testing.assert_close(tail.last_hidden_state, expected.last_hidden_state[:, 7:])
            self.assert_states_close(tail.state, expected.state)
            self.assertTrue(all(tail.past_key_values.is_sliding))

    def test_mlp_only_continuation_without_kv_cache(self):
        model = FWQwen3Model(self.config())
        first = model(self.ids[:, :3], use_cache=False)
        before = first.state.mlp_states[0].W_fast.clone()
        second = model(self.ids[:, 3:7], state=first.state, use_cache=False)
        self.assertEqual(second.state.tokens_seen, 7)
        self.assertIsNone(second.state.past_key_values)
        self.assertEqual(second.state.mlp_states[0].pending_count, 3)
        torch.testing.assert_close(first.state.mlp_states[0].W_fast, before)
        second.last_hidden_state.square().mean().backward()
        self.assertIsNotNone(model.layers[0].mlp.W_proj.grad)
        with self.assertRaisesRegex(ValueError, "reconstruct earlier KV"):
            model(self.ids[:, 7:8], state=second.state, use_cache=True)


if __name__ == "__main__":
    unittest.main()
