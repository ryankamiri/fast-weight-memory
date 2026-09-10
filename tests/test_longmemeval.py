import asyncio
import copy
import json
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import torch
from openai import APIConnectionError
import httpx
import yaml
from datasets import Dataset
from transformers import Qwen3Config, Qwen3ForCausalLM

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from architectures.qwen.attention import sdpa_attention_forward
from evaluation.data import PREFIX, format_history_and_question, prepare_example
from evaluation.judge import BackgroundJudge, Verdict, judge_all, request_verdict, rubric, summarize
from evaluation.run import configure_model, generate_example, load_model
from evaluation.storage import append_result, ensure_manifest, read_results


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def decode(self, ids, **kwargs):
        return bytes(ids).decode(errors="replace")

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
            if not enable_thinking:
                text += "<think>\n\n</think>\n\n"
        return self.encode(text) if tokenize else text


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
            self.assertEqual(prepared[name], str(value) if name == "answer" else value)
        self.assertEqual(source, original)

    def test_mixed_answer_types_serialize_to_parquet(self):
        sources = [example(), {**example(), "question_id": "numeric", "answer": 42}]
        rows = [prepare_example(source, CharacterTokenizer()) for source in sources]
        dataset = Dataset.from_list(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.parquet"
            dataset.to_parquet(path)
            restored = Dataset.from_parquet(str(path))
        self.assertEqual(list(restored["answer"]), [sources[0]["answer"], "42"])
        self.assertTrue(set(sources[0]).issubset(restored.column_names))

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
            launcher = (root / f"evaluation/sbatch/fs_qwen_eval_{mode}.sbatch").read_text()
            self.assertIn(f"--config {config_path}", launcher)
            self.assertIn("conda activate fast-weight-memory", launcher)
            self.assertIn("--gres=gpu:h200:1", launcher)
            self.assertEqual(config["judge"]["model"], "gpt-5.6-terra")
            self.assertIn(f'${{2:-output/longmemeval/{mode}-${{SLURM_JOB_ID}}}}', launcher)

    def test_new_configs_and_matched_fw_read_ablation(self):
        root = Path(__file__).resolve().parents[1]
        configs = {}
        for name in ("instruct_full", "instruct_swa", "base_full", "base_swa", "fw_swa_no_reads", "fw_swa", "swa"):
            path = root / f"evaluation/configs/longmemeval_{name}.yaml"
            configs[name] = yaml.safe_load(path.read_text())
            self.assertNotIn("#", path.read_text())
            launcher = (root / f"evaluation/sbatch/fs_qwen_eval_{name}.sbatch").read_text()
            self.assertIn(f"--config evaluation/configs/longmemeval_{name}.yaml", launcher)
            self.assertIn("conda activate fast-weight-memory", launcher)
            position = 1 if name.startswith(("instruct", "base")) else 2
            self.assertIn(f'${{{position}:-output/longmemeval/{name}-${{SLURM_JOB_ID}}}}', launcher)
        self.assertTrue(configs["fw_swa"]["model"]["fast_weight_read_scale"])
        self.assertFalse(configs["fw_swa_no_reads"]["model"]["fast_weight_read_scale"])
        configs["fw_swa_no_reads"]["model"]["fast_weight_read_scale"] = 1.0
        self.assertEqual(configs["fw_swa_no_reads"], configs["fw_swa"])
        for name in ("swa", "fw_swa"):
            settings = configs[name]["model"]
            self.assertEqual([settings[key] for key in (
                "teacher_window_size", "student_window_size", "chunk_size",
            )], [4096, 2048, 1024])
        for name, window in (("instruct_full", 32768), ("instruct_swa", 4096)):
            self.assertEqual(configs[name]["model"]["model_id"], "Qwen/Qwen3-0.6B")
            self.assertEqual(configs[name]["model"]["teacher_window_size"], window)
            self.assertEqual(configs[name]["prompt_format"], "chat")
            self.assertIn(151645, configs[name]["generation"]["eos_token_id"])
        for mode in ("full", "swa"):
            base = configs[f"base_{mode}"]
            instruct = configs[f"instruct_{mode}"]
            self.assertEqual(base["mode"], mode)
            self.assertEqual(base["model"]["model_id"], "Qwen/Qwen3-0.6B-Base")
            self.assertEqual(base["prompt_format"], "completion")
            self.assertEqual(base["generation"]["eos_token_id"], [151643])
            matched = {
                **base,
                "prompt_format": "chat",
                "model": {**base["model"], "model_id": "Qwen/Qwen3-0.6B"},
                "generation": {**base["generation"], "eos_token_id": [151645, 151643]},
            }
            self.assertEqual(matched, instruct)

    def test_half_reads_config_matches_full_reads(self):
        root = Path(__file__).resolve().parents[1]
        config_path = "evaluation/configs/longmemeval_fw_swa_half_reads.yaml"
        half = yaml.safe_load((root / config_path).read_text())
        full = yaml.safe_load((root / "evaluation/configs/longmemeval_fw_swa.yaml").read_text())
        self.assertEqual(half["model"]["fast_weight_read_scale"], 0.5)
        self.assertEqual(half["dataset"]["revision"], "bdb33409edba22c15721d57cb0b5a76d1620e6ba")
        half["model"]["fast_weight_read_scale"] = 1.0
        half["dataset"]["revision"] = full["dataset"]["revision"]
        self.assertEqual(half, full)
        launcher = (root / "evaluation/sbatch/fs_qwen_eval_fw_swa_half_reads.sbatch").read_text()
        self.assertIn(f"--config {config_path}", launcher)
        self.assertIn("fw_swa_half_reads-${SLURM_JOB_ID}", launcher)

    def test_instruct_prompt_uses_raw_fields_and_persists_only_system_message(self):
        row = prepare_example(example(), CharacterTokenizer())
        row["history_ids"] = [999999]  # These completion tokens must not be reused.
        tokenizer = CharacterTokenizer()
        original = copy.deepcopy(row)
        with patch.object(tokenizer, "apply_chat_template", wraps=tokenizer.apply_chat_template) as template:
            prepared = prepare_example(row, tokenizer, prompt_format="chat")
        for call in template.call_args_list:
            self.assertFalse(call.kwargs["enable_thinking"])
        text = tokenizer.decode(prepared["input_ids"])
        prefix = tokenizer.decode(prepared["input_ids"][:prepared["persistent_prefix_length"]])
        self.assertEqual(prefix, f"<|im_start|>system\n{PREFIX.strip()}<|im_end|>\n")
        self.assertLess(text.index("earlier"), text.index("later"))
        self.assertIn(row["question"], text)
        self.assertNotIn(row["answer"], text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(prepared["prompt_length"], len(prepared["input_ids"]))
        self.assertEqual(original, row)

    def test_prompt_boundaries_chronology_and_no_reference_leak(self):
        tokenizer = CharacterTokenizer()
        row = prepare_example(example(), tokenizer)
        text = tokenizer.decode(row["input_ids"])
        self.assertTrue(text.startswith(PREFIX))
        self.assertEqual(tokenizer.decode(row["input_ids"][:row["persistent_prefix_length"]]), PREFIX)
        self.assertLess(text.index("earlier"), text.index("later"))
        self.assertNotIn(example()["answer"], text)
        self.assertIn(example()["question"], text)
        history, question = format_history_and_question(example())
        old_tokens = tokenizer.encode(PREFIX) + tokenizer.encode(history) + tokenizer.encode(question + "\nAnswer:")
        self.assertEqual(row["input_ids"], old_tokens)
        self.assertEqual(len(row["input_ids"]), row["prompt_length"])

    def test_one_preparation_schema_for_raw_and_legacy_records(self):
        source = example()
        legacy = {**source, "history_ids": [999], "question_ids": [999],
                  "history_length": 999, "question_length": 999, "input_ids": [999]}
        tokenizer = CharacterTokenizer()
        completion = prepare_example(source, tokenizer)
        chat = prepare_example(source, tokenizer, "chat")
        self.assertEqual(set(completion), set(chat))
        for prompt_format in ("completion", "chat"):
            self.assertEqual(
                prepare_example(legacy, tokenizer, prompt_format),
                prepare_example(source, tokenizer, prompt_format),
            )
        with self.assertRaisesRegex(ValueError, "prompt_format"):
            prepare_example(source, tokenizer, "unknown")

    def test_modes_do_not_change_trained_fast_weight_architecture(self):
        config = FWQwen3Config(fast_weight_layers=[])
        self.assertEqual(configure_model(config, "full").teacher_window_size, 8192)
        with self.assertRaises(ValueError):
            configure_model(config, "fw_swa")
        config = FWQwen3Config(fast_weight_layers=[0])
        self.assertEqual(configure_model(config, "fw_swa").teacher_window_size, 8192)

    def test_window_overrides_preserve_checkpoint_architecture(self):
        original = FWQwen3Config(fast_weight_layers=[0, 7])
        settings = {"teacher_window_size": 4096, "student_window_size": 2048, "chunk_size": 1024}
        updated = configure_model(original, "fw_swa", settings)
        self.assertEqual(updated.fast_weight_layers, [0, 7])
        self.assertEqual(updated.teacher_window_size, 4096)
        self.assertEqual(updated.student_window_size, 2048)
        self.assertEqual(updated.chunk_size, 1024)
        self.assertEqual(original.teacher_window_size, 8192)
        with self.assertRaises(ValueError):
            configure_model(original, "fw_swa", {**settings, "student_window_size": 8192})

    @torch.inference_mode()
    def test_untouched_qwen_loading_matches_native_and_has_no_metric_callback(self):
        native_config = Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        )
        native_config._attn_implementation = "sdpa"
        native = Qwen3ForCausalLM(native_config).eval()
        config = FWQwen3Config(**native_config.to_dict(), fast_weight_layers=[])
        with tempfile.TemporaryDirectory() as directory:
            native.save_pretrained(directory)
            wrapped = load_model(directory, config).eval()
            native = Qwen3ForCausalLM.from_pretrained(
                directory, dtype=torch.bfloat16, attn_implementation="sdpa",
            ).eval()
        self.assertEqual(set(wrapped.state_dict()), set(native.state_dict()))
        for name, weight in native.state_dict().items():
            torch.testing.assert_close(wrapped.state_dict()[name], weight)
        ids = torch.arange(6)[None]
        torch.testing.assert_close(wrapped(ids).logits, native(ids).logits)

    @torch.inference_mode()
    def test_loading_fw_reads_off_keeps_trained_weights_and_skips_student_attention(self):
        config = FWQwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            fast_weight_layers=[0], teacher_window_size=8, student_window_size=4, chunk_size=4,
        )
        trained = FWQwen3ForCausalLM(config).to(torch.bfloat16).eval()
        with tempfile.TemporaryDirectory() as directory:
            trained.save_pretrained(directory)
            settings = {"teacher_window_size": 4, "student_window_size": 2, "chunk_size": 2}
            updated = configure_model(config, "fw_swa", settings)
            loaded = load_model(directory, updated, fast_weight_read_scale=0.0).eval()
            half = load_model(directory, configure_model(config, "fw_swa", settings),
                              fast_weight_read_scale=0.5).eval()
            self.assertEqual(half.config.fast_weight_read_scale, 0.5)
            self.assertEqual(half.model.layers[0].mlp.fast_weight_read_scale, 0.5)
            half.save_pretrained(directory)
            restored = FWQwen3ForCausalLM.from_pretrained(directory)
            self.assertEqual(restored.model.layers[0].mlp.fast_weight_read_scale, 0.5)
        mlp = loaded.model.layers[0].mlp
        self.assertTrue(mlp.is_fast_weight_layer)
        self.assertFalse(mlp.fast_weight_read_scale)
        self.assertEqual(mlp.chunk_size, 2)
        for name, weight in trained.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], weight)
        with patch.object(mlp.student_conv, "forward", side_effect=AssertionError("Student conv ran")), \
             patch("architectures.qwen.attention.sdpa_attention_forward", wraps=sdpa_attention_forward) as attention:
            output = loaded(torch.arange(6)[None], use_cache=True)
        self.assertEqual(attention.call_count, 1)
        self.assertEqual(output.state.mlp_states[0].pending_count, 0)
        self.assertEqual(output.state.mlp_states[0].W_fast.count_nonzero().item(), 0)
        self.assertEqual(output.state.past_key_values.layers[0].keys.shape[-2], 3)

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
        row = {"question_id": "x", "input_ids": list(range(1, 12)),
               "persistent_prefix_length": 2, "prompt_length": 11}
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
            self.assertEqual(call.args[0].tolist(), [row["input_ids"]])
            self.assertEqual(call.kwargs["persistent_mask"].tolist(), [True, True] + [False] * 9)
            self.assertEqual(call.kwargs["execution_block_size"], 4)
        self.assertEqual(first["generated_tokens"], 2)
        self.assertNotIn("fast_weight_metrics", first)
        json.dumps(first, allow_nan=False)

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


class BackgroundJudgeTests(unittest.TestCase):
    def test_nonblocking_bounded_judging_and_duplicate_submission(self):
        started = Event()
        release = Event()
        active = 0
        peak = 0
        calls = []

        async def verdict(client, row, hypothesis, model):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            calls.append(row["question_id"])
            if active == 2:
                started.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
            active -= 1
            return {"correct": True}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [{**example(), "question_id": str(i), "abstention": False} for i in range(4)]
            with patch("evaluation.judge.AsyncOpenAI", return_value=AsyncMock()), \
                 patch("evaluation.judge.request_verdict", side_effect=verdict):
                with BackgroundJudge(rows, output, {"model": "test", "concurrency": 2}) as judge:
                    try:
                        for row in rows:
                            prediction = {"question_id": row["question_id"], "hypothesis": "answer"}
                            append_result(output / "predictions.jsonl", prediction)
                            judge.submit(prediction)
                            judge.submit(prediction)
                        self.assertTrue(started.wait(5))
                        # All generation submissions completed while the first judges are blocked.
                        self.assertEqual(len(read_results(output / "predictions.jsonl")), 4)
                    finally:
                        release.set()
            self.assertEqual(peak, 2)
            self.assertEqual(len(calls), 4)
            self.assertEqual(len(read_results(output / "judgments.jsonl")), 4)

    def test_failure_and_interrupted_tail_resume_only_missing_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [{**example(), "question_id": str(i), "abstention": False} for i in range(3)]
            settings = {"model": "test", "concurrency": 2}
            predictions = [{"question_id": str(i), "hypothesis": "answer"} for i in range(2)]
            for prediction in predictions:
                append_result(output / "predictions.jsonl", prediction)

            async def verdict(client, row, hypothesis, model):
                if row["question_id"] == "1":
                    raise RuntimeError("network unavailable")
                return {"correct": True}

            with patch("evaluation.judge.AsyncOpenAI", return_value=AsyncMock()), \
                 patch("evaluation.judge.request_verdict", side_effect=verdict):
                with self.assertRaisesRegex(RuntimeError, "Some judgments failed"):
                    asyncio.run(judge_all(rows, output, settings))
            self.assertEqual(set(read_results(output / "judgments.jsonl")), {"0"})
            with (output / "judgments.jsonl").open("ab") as file:
                file.write(b'{"question_id":')
            with patch("evaluation.judge.AsyncOpenAI", return_value=AsyncMock()), \
                 patch("evaluation.judge.request_verdict", new_callable=AsyncMock,
                       return_value={"correct": False}) as request:
                asyncio.run(judge_all(rows, output, settings))
                request.assert_awaited_once()
                self.assertEqual(request.call_args.args[1]["question_id"], "1")
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["judged"], 2)
            self.assertEqual(summary["missing"], 1)  # Third answer has not been generated yet.

    def test_generation_exception_keeps_completed_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = {**example(), "abstention": False}
            prediction = {"question_id": row["question_id"], "hypothesis": "answer"}
            append_result(output / "predictions.jsonl", prediction)
            with patch("evaluation.judge.AsyncOpenAI", return_value=AsyncMock()), \
                 patch("evaluation.judge.request_verdict", new_callable=AsyncMock,
                       return_value={"correct": True}):
                with self.assertRaisesRegex(RuntimeError, "generation failed"):
                    with BackgroundJudge([row], output, {"model": "test", "concurrency": 2}) as judge:
                        judge.submit(prediction)
                        raise RuntimeError("generation failed")
            self.assertIn(row["question_id"], read_results(output / "judgments.jsonl"))


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
