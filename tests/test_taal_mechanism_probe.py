import unittest

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.taal.mechanism_probe import inspect_episode


class Tokenizer:
    def decode(self, tokens):
        return str(tokens[0])


class TaalMechanismProbeTests(unittest.TestCase):
    @torch.inference_mode()
    def test_probe_records_controls_and_restores_forget_projection(self):
        config = TaalQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_dropout=0.0,
            working_memory_size=4,
            max_persistent_kv_tokens=0,
            memory_dim=8,
            memory_depth=2,
            memory_conv_kernel_size=2,
            memory_chunk_size=1,
            num_persistent_tokens=2,
        )
        config._attn_implementation = "sdpa"
        model = TaalQwen3ForCausalLM(config).eval()
        projection = model.model.layers[0].taal.neural_memory.forget_projection
        weight = projection.weight.clone()
        bias = projection.bias.clone()
        example = {
            "example_id": "probe",
            "input_ids": [1, 2, 3, 4, 5, 6],
            "fact_position": 1,
            "final_query_position": 4,
            "target_token_id": 7,
            "candidate_token_ids": [7, 8],
        }

        baseline = inspect_episode(model, example, Tokenizer(), forget=None)
        low_forget = inspect_episode(model, example, Tokenizer(), forget=0.0001)

        self.assertEqual(len(baseline["events"]), len(baseline["positions"]))
        by_position = {event["position"]: event for event in baseline["events"]}
        self.assertIn("value_prediction_error_norm", by_position[1])
        self.assertIn("forget_coefficient", by_position[1])
        self.assertIn("pre_add_injection_norm", by_position[1])
        self.assertIn("post_add_delta_norm", by_position[1])
        self.assertIn("proposed_write_norm", by_position[1])
        self.assertAlmostEqual(by_position[1]["forget_coefficient"], 0.01, places=4)
        self.assertAlmostEqual(
            next(event for event in low_forget["events"] if event["position"] == 1)[
                "forget_coefficient"
            ],
            0.0001,
            places=6,
        )
        torch.testing.assert_close(projection.weight, weight)
        torch.testing.assert_close(projection.bias, bias)
