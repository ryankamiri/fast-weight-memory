import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from architectures.taal.qwen.model import TaalQwen3Model
from training.train import verify_loading


class TaalInitializationTests(unittest.TestCase):
    def config(self):
        config = TaalQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            working_memory_size=8,
            max_persistent_kv_tokens=2,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=1,
            memory_chunk_size=1,
            num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        return config

    def taal_layers(self, model):
        backbone = model.model if isinstance(model, TaalQwen3ForCausalLM) else model
        return [layer.taal for layer in backbone.layers]

    def test_native_checkpoint_initializes_gates_after_meta_allocation(self):
        # HF replaces missing meta parameters with empty CPU storage before
        # calling _init_weights. Poison scalar storage so allocator luck cannot
        # hide a missing gate initializer.
        empty_like = torch.empty_like
        for native_type, taal_type in (
            (Qwen3Model, TaalQwen3Model),
            (Qwen3ForCausalLM, TaalQwen3ForCausalLM),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(model=taal_type.__name__, dtype=dtype):
                    config = self.config()
                    native = native_type(Qwen3Config.from_dict(config.to_dict()))
                    poisoned = []

                    def poisoned_empty_like(value, *args, **kwargs):
                        allocated = empty_like(value, *args, **kwargs)
                        if value.is_meta and allocated.ndim == 0:
                            allocated.fill_(12345)
                            poisoned.append(allocated)
                        return allocated

                    with tempfile.TemporaryDirectory() as folder:
                        native.save_pretrained(folder)
                        with patch("torch.empty_like", side_effect=poisoned_empty_like):
                            loaded, info = taal_type.from_pretrained(
                                folder, config=config, dtype=dtype,
                                attn_implementation="sdpa", output_loading_info=True,
                            )
                    self.assertEqual(len(poisoned), config.num_hidden_layers)
                    self.assertEqual(info["unexpected_keys"], [])
                    for layer in self.taal_layers(loaded):
                        self.assertEqual(layer.residual_gate.item(), 0.0)
                        self.assertEqual(layer.residual_gate.tanh().item(), 0.0)
                    for name, value in native.state_dict().items():
                        torch.testing.assert_close(
                            loaded.state_dict()[name], value.to(dtype=dtype),
                            atol=0, rtol=0,
                        )

    def test_complete_taal_checkpoint_preserves_learned_gates(self):
        for model_type in (TaalQwen3Model, TaalQwen3ForCausalLM):
            with self.subTest(model=model_type.__name__):
                model = model_type(self.config())
                with torch.no_grad():
                    for index, layer in enumerate(self.taal_layers(model)):
                        layer.residual_gate.fill_(0.25 * (index + 1))
                with tempfile.TemporaryDirectory() as folder:
                    model.save_pretrained(folder)
                    restored, info = model_type.from_pretrained(
                        folder, output_loading_info=True,
                    )
                self.assertEqual(info["missing_keys"], [])
                for original, loaded in zip(
                    self.taal_layers(model), self.taal_layers(restored), strict=True,
                ):
                    torch.testing.assert_close(
                        original.residual_gate, loaded.residual_gate, atol=0, rtol=0,
                    )

    def test_training_verifies_only_newly_added_gates(self):
        model = TaalQwen3ForCausalLM(self.config())
        name = "model.layers.0.taal.residual_gate"
        info = {"missing_keys": [name]}
        verify_loading(model, info)
        gate = model.model.layers[0].taal.residual_gate
        for value in (12345.0, float("nan")):
            with self.subTest(value=value):
                with torch.no_grad():
                    gate.fill_(value)
                with self.assertRaisesRegex(ValueError, "residual gates must initialize to zero"):
                    verify_loading(model, info)
        # A legitimate learned gate in a complete checkpoint is not reset or
        # required to be zero by the fresh-initialization guard.
        with torch.no_grad():
            gate.fill_(0.25)
        verify_loading(model, {"missing_keys": []})
        self.assertEqual(gate.item(), 0.25)


if __name__ == "__main__":
    unittest.main()
