"""Prompt and scoring adapters; official benchmark semantics stay explicit."""

from pathlib import Path

from evaluation.data import PROMPT_VERSION, prepare_example
from evaluation.shared.contracts import Benchmark, Example, Prompt
from evaluation.shared.vendor.babilong import metrics, prompts
from evaluation.storage import file_sha256


def exact_match(hypothesis: str, reference: str) -> float:
    # A deliberately strict secondary metric, distinct from benchmark rubrics.
    return float(hypothesis.strip().casefold() == reference.strip().casefold())


def encode(tokenizer, text: str) -> Prompt:
    return Prompt(tuple(tokenizer.encode(text, add_special_tokens=False)))


class ControlledBenchmark(Benchmark):
    name = "controlled"
    prompt_version = "rendered-completion-v1"
    scorer_version = "exact-casefold-strip-v1"

    def prepare(self, row, tokenizer):
        if "prompt" not in row:
            raise ValueError(
                "Controlled rows require a rendered prompt, without the answer"
            )

        references = row.get("answers", [row.get("answer")])
        if not isinstance(references, list):
            raise ValueError("Controlled answers must be a list")

        metadata = dict(row.get("metadata", {}))
        if "answerable" in metadata:
            if type(metadata["answerable"]) is not bool:
                raise ValueError("answerable must be boolean")

            response = metadata.get("abstention_response")
            if not isinstance(response, str) or not response:
                raise ValueError(
                    "Abstention examples require a nonempty abstention_response"
                )

            has_abstention_reference = False
            for answer in references:
                if isinstance(answer, str) and exact_match(answer, response):
                    has_abstention_reference = True
            if metadata["answerable"] and has_abstention_reference:
                raise ValueError(
                    "Answerable examples cannot use the abstention response as their answer"
                )

            if not metadata["answerable"]:
                if len(references) != 1 or not has_abstention_reference:
                    raise ValueError(
                        "Unanswerable examples must have only the abstention response as reference"
                    )

        return Example(
            str(row["example_id"]),
            encode(tokenizer, row["prompt"]),
            tuple(references),
            metadata,
            row.get("group_id"),
        )

    def score(self, example, hypothesis):
        scores = {
            "exact_match": max(
                exact_match(hypothesis, answer) for answer in example.references
            )
        }
        if "answerable" not in example.metadata:
            return scores

        response = example.metadata["abstention_response"]
        abstained = exact_match(hypothesis, response)
        scores["abstention_rate"] = abstained
        if example.metadata["answerable"]:
            scores["false_abstention_rate"] = abstained
        else:
            scores["correct_abstention_rate"] = abstained
            scores["unanswerable_nonabstention_rate"] = 1.0 - abstained
        return scores


class RulerBenchmark(Benchmark):
    """Preserve official NIAH rendering and per-reference substring scoring."""

    name = "ruler"
    prompt_version = "official-rendered-input-plus-answer-prefix-v1"
    scorer_version = "ruler-db43f92-string-match-all-v1"

    def __init__(self, task: str):
        tasks = {
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey_1",
            "niah_multikey_2",
            "niah_multikey_3",
            "niah_multivalue",
            "niah_multiquery",
        }
        if task not in tasks:
            raise ValueError("This adapter supports RULER NIAH retrieval variants only")

        self.task = task
        self.prompt_version = f"{type(self).prompt_version}/{task}"

    def prepare(self, row, tokenizer):
        references = row["outputs"]
        if not isinstance(references, list):
            raise ValueError("RULER outputs must be a list")

        # The official preparation removes answer_prefix from input for generation.
        text = row["input"] + row.get("answer_prefix", "")
        metadata = {
            "task": self.task,
            "length": row.get("length"),
            "answer_position": row.get("token_position_answer"),
        }
        return Example(
            str(row["index"]), encode(tokenizer, text), tuple(references), metadata
        )

    def score(self, example, hypothesis):
        matched = [
            answer.lower() in hypothesis.lower() for answer in example.references
        ]
        return {
            "official_match": sum(matched) / len(matched),
            "all_references_present": float(all(matched)),
            "exact_match": exact_match(hypothesis, " ".join(example.references)),
        }


class BabilongBenchmark(Benchmark):
    """Use bundled official prompts/scoring with their content hashes recorded."""

    name = "babilong"

    def __init__(self, task: str):
        prompt_path = Path(prompts.__file__)
        metric_path = Path(metrics.__file__)
        self.prompts = prompts
        self.metrics = metrics
        if (
            task not in self.prompts.DEFAULT_PROMPTS
            or task not in self.metrics.TASK_LABELS
        ):
            raise ValueError(f"Unsupported BABILong task: {task}")

        self.task = task
        self.prompt_version = f"official-completion-{task}-" + file_sha256(prompt_path)
        self.scorer_version = "official-" + file_sha256(metric_path)

    def prepare(self, row, tokenizer):
        text = self.prompts.get_formatted_input(
            row["input"], row["question"], **self.prompts.DEFAULT_PROMPTS[self.task]
        )
        return Example(
            str(row["example_id"]),
            encode(tokenizer, text),
            (row["target"],),
            {"task": self.task, "question": row["question"]},
        )

    def score(self, example, hypothesis):
        correct = self.metrics.compare_answers(
            example.references[0],
            hypothesis,
            example.metadata["question"],
            self.metrics.TASK_LABELS[self.task],
        )
        return {
            "official_accuracy": float(correct),
            "exact_match": exact_match(hypothesis, example.references[0]),
        }


class LongMemEvalBenchmark(Benchmark):
    name = "longmemeval"
    scorer_version = "external-longmemeval-judge"

    def __init__(self, prompt_format="completion"):
        if prompt_format not in {"completion", "chat"}:
            raise ValueError("LongMemEval prompt_format must be completion or chat")

        self.prompt_format = prompt_format
        self.prompt_version = f"{PROMPT_VERSION}/{prompt_format}"

    def prepare(self, row, tokenizer):
        prepared = prepare_example(row, tokenizer, self.prompt_format)
        return Example(
            str(row["question_id"]),
            Prompt(tuple(prepared["input_ids"]), prepared["persistent_prefix_length"]),
            (prepared["answer"],),
            {key: prepared[key] for key in ("question", "question_type", "abstention")},
        )

    def score(self, example, hypothesis):
        # An exact match is not the official LongMemEval judge. No substitute score.
        return {}


def make_benchmark(settings: dict) -> Benchmark:
    name = settings["name"]
    if name == "controlled_prepared":
        return PreparedControlledBenchmark()
    if name == "controlled":
        return ControlledBenchmark()
    if name == "ruler":
        return RulerBenchmark(settings["task"])
    if name == "babilong":
        return BabilongBenchmark(settings["task"])
    if name == "longmemeval":
        return LongMemEvalBenchmark(settings.get("prompt_format", "completion"))
    raise ValueError(f"Unknown benchmark: {name}")


class PreparedControlledBenchmark(ControlledBenchmark):
    """Historical workshop rows already tokenized by a separately identified tokenizer."""

    name = "controlled_prepared"
    prompt_version = "prepared-workshop-completion-v1"

    def prepare(self, row, tokenizer):
        ids = row["input_ids"]
        if row.get("final_answer_position", len(ids)) != len(ids):
            raise ValueError("Prepared prompt must end immediately before the answer")

        metadata = {
            "condition": row.get("condition"),
            "fact_count": row.get("fact_count"),
            "gap": row.get("fact_to_answer_gap"),
            "query_variant": row.get("query_variant"),
        }
        # selection_group identifies paired questions, unlike matched intervention pairs.
        group = row.get("selection_group") or None
        return Example(
            str(row["example_id"]),
            Prompt(tuple(ids)),
            (row["answer"],),
            metadata,
            group,
        )
