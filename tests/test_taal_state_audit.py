import unittest
from pathlib import Path

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.taal.state_audit import (
    example_batches_by_prefix_length,
    fork_session_state,
    select_examples,
    select_memory_session,
    split_episode,
    summarize,
)


class TaalStateAuditTests(unittest.TestCase):
    @staticmethod
    def config():
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
        return config

    def test_slurm_launcher_runs_the_state_audit(self):
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root
            / "evaluation/sbatch/taal/fs_qwen_delayed_recall_state_audit.sbatch"
        ).read_text()

        self.assertIn("#SBATCH --gres=gpu:a100:1", launcher)
        self.assertIn("-m evaluation.taal.state_audit", launcher)
        self.assertIn(
            "--config evaluation/configs/taal/delayed_recall_state_audit.yaml",
            launcher,
        )
        self.assertIn('--checkpoint "$1"', launcher)
        self.assertIn("output/taal/pilot-1b-${SLURM_JOB_ID}", launcher)

    @torch.inference_mode()
    def test_fork_replaces_only_memory_and_clones_kv(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        source = model.prefill(
            torch.tensor([[1, 2, 3, 4, 5]]),
            execution_block_size=3,
        ).state
        reset = model.model.initial_memory_states(batch_size=1)
        fork = fork_session_state(source, reset)

        self.assertEqual(fork.tokens_seen, source.tokens_seen)
        self.assertIsNot(fork.past_key_values, source.past_key_values)
        source_layer = source.past_key_values.layers[0]
        fork_layer = fork.past_key_values.layers[0]
        torch.testing.assert_close(fork_layer.keys, source_layer.keys)

        for name, expected in reset[0].weights.items():
            torch.testing.assert_close(fork.memory_states[0].weights[name], expected)
            self.assertIsNot(fork.memory_states[0].weights[name], expected)

        source_length = source_layer.cumulative_length
        resumed = model.prefill(
            torch.tensor([[6, 7]]),
            state=fork,
            execution_block_size=2,
            prepend_memory_tokens=False,
        )
        self.assertEqual(resumed.logits.shape, (1, 1, 32))
        self.assertEqual(resumed.state.tokens_seen, source.tokens_seen + 2)
        # Resuming the fork must not mutate the pre-query state used by the
        # other interventions.
        self.assertEqual(source_layer.cumulative_length, source_length)
        self.assertGreater(source_layer.keys.abs().sum(), 0)

    def test_reset_and_zero_are_distinct_controls(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        reset = model.model.initial_memory_states(batch_size=1)
        zeroed = model.model.zero_memory_states(batch_size=1)

        self.assertTrue(any(
            value.count_nonzero() > 0
            for value in reset[0].weights.values()
        ))
        self.assertTrue(all(
            value.count_nonzero() == 0
            for value in zeroed[0].weights.values()
        ))
        self.assertTrue(all(
            value.count_nonzero() == 0
            for value in zeroed[0].momentum.values()
        ))

    @torch.inference_mode()
    def test_state_bank_can_extract_independent_batched_sessions(self):
        model = TaalQwen3ForCausalLM(self.config()).eval()
        output = model.prefill(
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
            execution_block_size=2,
        )
        first = select_memory_session(output.state.memory_states, 0)
        second = select_memory_session(output.state.memory_states, 1)

        for name in first[0].weights:
            torch.testing.assert_close(
                first[0].weights[name],
                output.state.memory_states[0].weights[name][0:1],
            )
            torch.testing.assert_close(
                second[0].weights[name],
                output.state.memory_states[0].weights[name][1:2],
            )
            self.assertIsNot(first[0].weights[name], second[0].weights[name])

    def test_split_and_selection_use_recorded_query_boundary(self):
        examples = [
            {
                "example_id": "second",
                "fact_id": 2,
                "condition": "no_bridge",
                "query_variant": "exact",
                "input_ids": [1, 2, 3, 4],
                "final_query_position": 2,
            },
            {
                "example_id": "first",
                "fact_id": 1,
                "condition": "no_bridge",
                "query_variant": "exact",
                "input_ids": [5, 6, 7],
                "final_query_position": 1,
            },
            {
                "example_id": "excluded",
                "fact_id": 3,
                "condition": "bridge",
                "query_variant": "exact",
                "input_ids": [8, 9],
                "final_query_position": 1,
            },
        ]
        selected = select_examples(examples, {
            "start": 1,
            "end": 3,
            "conditions": ["no_bridge"],
            "query_variants": ["exact"],
        })

        self.assertEqual([row["example_id"] for row in selected], ["first", "second"])
        self.assertEqual(split_episode(selected[0]), ([5], [6, 7]))
        batches = list(example_batches_by_prefix_length(selected, batch_size=2))
        self.assertEqual([[row["example_id"] for row, _ in batch] for batch in batches], [
            ["first"],
            ["second"],
        ])
        with self.assertRaisesRegex(ValueError, "must split"):
            split_episode({"input_ids": [1], "final_query_position": 1})
        with self.assertRaisesRegex(ValueError, "Expected 2 audit examples"):
            select_examples(selected[:1], {
                "start": 1,
                "end": 3,
                "conditions": ["no_bridge"],
                "query_variants": ["exact"],
            })

    def test_summary_reports_paired_condition_effects(self):
        def row(example, condition, correct, log_probability, choice=7):
            return {
                "example_id": example,
                "audit_condition": condition,
                "candidate_correct": correct,
                "vocabulary_top_1_correct": correct,
                "target_log_probability": log_probability,
                "target_reciprocal_rank": 1.0 if correct else 0.5,
                "target_vs_best_decoy_margin": 1.0 if correct else -1.0,
                "candidate_choice_token_id": choice,
                "swapped_target_token_id": 8 if condition == "swapped" else None,
                "swapped_target_in_candidates": condition == "swapped",
                "swapped_target_log_probability": (
                    -0.2 if condition == "swapped" else None
                ),
                "swapped_target_vocabulary_top_1": condition == "swapped",
            }

        result = summarize([
            row("a", "correct_full", True, -0.1),
            row("a", "reads_disabled", False, -2.1),
            row("a", "swapped", False, -1.1, choice=8),
        ])

        self.assertEqual(result["metrics"]["correct_full"]["candidate_accuracy"], 1.0)
        self.assertEqual(
            result["paired_vs_correct_full"]["reads_disabled"]
            ["correct_full_minus_condition_candidate_accuracy"],
            1.0,
        )
        self.assertEqual(
            result["swapped_source"]["candidate_choice_rate_when_eligible"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
