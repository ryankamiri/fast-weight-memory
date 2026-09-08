import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import torch
from openai import APIConnectionError
import httpx
import yaml

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from evaluation.data import PREFIX, prepare_example
from evaluation.judge import Verdict, judge_all, request_verdict, rubric, summarize
from evaluation.run import configure_model, generate_example
from evaluation.storage import append_result, ensure_manifest, read_results


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode(errors="replace")


def example():
    return {
        "question_id": "example", "question_type": "multi-session",
        "question": "What changed?", "answer": "DO NOT LEAK THIS REFERENCE",
        "question_date": "2023/04/11 (Tue) 12:00",
        "haystack_dates": ["2023/04/10 (Mon) 12:00", "2023/04/09 (Sun) 12:00"],
        "haystack_sessions": [[{"role": "user", "content": "later"}], [{"role": "user", "content": "earlier"}]],
    }


class LongMemEvalTests(unittest.TestCase):
    def test_preparation_preserves_every_original_field(self):
        source = example()
        source.update({
            "answer": 42,
            "haystack_session_ids": ["later-id", "earlier-id"],
            "answer_session_ids": ["earlier-id"],
            "extra_metadata": {"nested": [1, 2, 3]},
        })
        original = copy.deepcopy(source)
        prepared = prepare_example(source, CharacterTokenizer())
        for name, value in original.items():
            self.assertEqual(prepared[name], value)
        self.assertEqual(source, original)

    def test_three_configs_match_their_launchers(self):
        root = Path(__file__).resolve().parents[1]
        for mode in ("full", "swa", "fw_swa"):
            config_path = f"evaluation/configs/longmemeval_{mode}.yaml"
            config = yaml.safe_load((root / config_path).read_text())
            self.assertEqual(config["mode"], mode)
            self.assertEqual(config["dataset"]["variant"], "oracle")
            self.assertNotIn("execution_block_size", config)
            self.assertEqual(config["generation"]["execution_block_size"], 4096)
            self.assertNotIn("#", (root / config_path).read_text())
            launcher = (root / f"evaluation/fs_qwen_eval_{mode}.sbatch").read_text()
            self.assertIn(f"--config {config_path}", launcher)
            self.assertIn("conda activate fast-weight-memory", launcher)
            self.assertIn("--gres=gpu:h200:1", launcher)

    def test_prompt_boundaries_chronology_and_no_reference_leak(self):
        tokenizer = CharacterTokenizer()
        row = prepare_example(example(), tokenizer)
        history = tokenizer.decode(row["history_ids"])
        question = tokenizer.decode(row["question_ids"])
        self.assertTrue(history.startswith(PREFIX))
        self.assertEqual(tokenizer.decode(row["history_ids"][:row["persistent_prefix_length"]]), PREFIX)
        self.assertLess(history.index("earlier"), history.index("later"))
        self.assertNotIn(example()["answer"], history + question)
        self.assertNotIn(example()["question"], history)
        self.assertIn(example()["question"], question)
        self.assertEqual(row["prompt_length"], row["history_length"] + row["question_length"])

    def test_modes_do_not_change_trained_fast_weight_architecture(self):
        config = FWQwen3Config(fast_weight_layers=[])
        self.assertEqual(configure_model(config, "full").teacher_window_size, 8192)
        with self.assertRaises(ValueError):
            configure_model(config, "fw_swa")
        config = FWQwen3Config(fast_weight_layers=[0])
        self.assertEqual(configure_model(config, "fw_swa").teacher_window_size, 8192)

    @torch.inference_mode()
    def test_full_mode_blocked_prefill_matches_full_causal_forward(self):
        config = FWQwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            fast_weight_layers=[], teacher_window_size=15, student_window_size=2,
        )
        configure_model(config, "full")
        model = FWQwen3ForCausalLM(config).eval()
        ids = torch.arange(13)[None]
        whole = model(ids, logits_to_keep=1)
        blocked = model.prefill(ids, execution_block_size=3)
        torch.testing.assert_close(whole.logits, blocked.logits)
        self.assertEqual(blocked.state.past_key_values.layers[0].keys.shape[-2], 13)

    def test_each_example_starts_fresh_and_retains_prefix(self):
        config = FWQwen3Config(
            vocab_size=256, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            fast_weight_layers=[0], teacher_window_size=8, student_window_size=4,
            chunk_size=4, conv_kernel_size=2,
        )
        model = FWQwen3ForCausalLM(config).eval()
        row = {"question_id": "x", "history_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9],
               "question_ids": [10, 11], "persistent_prefix_length": 2, "prompt_length": 11}
        settings = {"seed": 42, "generation": {
            "execution_block_size": 4,
            "max_new_tokens": 2, "do_sample": True, "top_k": 5, "eos_token_id": [],
        }}
        with patch.object(model, "prefill", wraps=model.prefill) as prefill:
            first = generate_example(model, CharacterTokenizer(), row, settings, torch.device("cpu"))
            second = generate_example(model, CharacterTokenizer(), row, settings, torch.device("cpu"))
        self.assertEqual(first, second)
        self.assertEqual(prefill.call_count, 2)
        for call in prefill.call_args_list:
            self.assertIsNone(call.kwargs["state"])
            self.assertEqual(call.args[0].tolist(), [row["history_ids"] + row["question_ids"]])
            self.assertEqual(call.kwargs["persistent_mask"].tolist(), [True, True] + [False] * 9)
            self.assertEqual(call.kwargs["execution_block_size"], 4)
        self.assertEqual(first["generated_tokens"], 2)

    def test_resume_and_partial_final_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            append_result(path, {"question_id": "a", "hypothesis": "ok"})
            with path.open("ab") as file:
                file.write(b'{"question_id":')
            self.assertEqual(set(read_results(path)), {"a"})
            append_result(path, {"question_id": "b", "hypothesis": "ok"})
            self.assertEqual(set(read_results(path)), {"a", "b"})
            manifest = Path(directory) / "manifest.json"
            ensure_manifest(manifest, {"mode": "swa"})
            ensure_manifest(manifest, {"mode": "swa"})
            with self.assertRaises(ValueError):
                ensure_manifest(manifest, {"mode": "full"})

    def test_category_rules_and_incomplete_scores(self):
        row = {"question_id": "x", "question_type": "temporal-reasoning", "abstention": False}
        self.assertIn("off-by-one", rubric(row))
        row["abstention"] = True
        self.assertIn("insufficient", rubric(row))
        self.assertEqual(summarize([row], {})["missing"], 1)
        self.assertEqual(summarize([row], {})["scores"], {})


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_errors_retry_but_refusals_are_not_wrong_answers(self):
        response = SimpleNamespace(output_parsed=None)
        client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock(side_effect=[
            APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses")), response,
        ])))
        with patch("evaluation.judge.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaisesRegex(ValueError, "no verdict"):
                await request_verdict(client, {**example(), "abstention": False}, "answer", "gpt-5.6-luna")
        self.assertEqual(client.responses.parse.await_count, 2)

    async def test_structured_verdict_and_no_history_sent(self):
        client = SimpleNamespace(responses=SimpleNamespace(parse=AsyncMock(return_value=SimpleNamespace(
            output_parsed=Verdict(correct=False), id="test", model="gpt-5.6-luna", usage=None,
        ))))
        row = {**example(), "abstention": False}
        with patch("evaluation.judge.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await request_verdict(client, row, "I don't know", "gpt-5.6-luna")
            sleep.assert_awaited_once_with(1)
        self.assertFalse(result["correct"])
        payload = json.loads(client.responses.parse.call_args.kwargs["input"])
        self.assertEqual(set(payload), {"question", "reference", "response"})

    async def test_completed_judgments_are_not_requested_again(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = {**example(), "abstention": False}
            append_result(output / "predictions.jsonl", {"question_id": "example", "hypothesis": "answer"})
            settings = {"model": "gpt-5.6-luna", "concurrency": 32}
            client = AsyncMock()
            with patch("evaluation.judge.AsyncOpenAI", return_value=client), patch(
                "evaluation.judge.request_verdict", new_callable=AsyncMock,
                return_value={"correct": True},
            ) as request:
                await judge_all([row], output, settings)
                await judge_all([row], output, settings)
                self.assertEqual(request.await_count, 1)
            self.assertEqual(json.loads((output / "summary.json").read_text())["judged"], 1)


if __name__ == "__main__":
    unittest.main()
