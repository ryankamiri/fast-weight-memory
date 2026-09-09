"""LongMemEval history formatting for base and instruction-tuned Qwen."""

from datetime import datetime

SOURCE = "xiaowu0162/longmemeval-cleaned"
SOURCE_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
TOKENIZER = "Qwen/Qwen3-0.6B-Base"
FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
PREFIX = (
    "Use the conversation history to answer the final question. "
    "If the history does not provide enough information, say that you do not know.\n\n"
)
PROMPT_VERSION = "completion-v1"


def format_history_and_question(example):
    dates = example["haystack_dates"]
    sessions = example["haystack_sessions"]
    if len(dates) != len(sessions):
        raise ValueError("Session dates and conversations must align")
    ordered = sorted(zip(dates, sessions), key=lambda pair: datetime.strptime(pair[0], "%Y/%m/%d (%a) %H:%M"))
    history = "".join(
        f"Session {index + 1}\nDate: {date}\n"
        + "".join(f"{turn['role']}: {turn['content']}\n" for turn in turns)
        + "\n"
        for index, (date, turns) in enumerate(ordered)
    )
    question = f"Current date: {example['question_date']}\nQuestion: {example['question']}"
    return history, question


def prepare_example(example, tokenizer, prompt_format="completion"):
    """Prepare one source record with a shared token/metadata schema."""
    history, question = format_history_and_question(example)
    if prompt_format == "chat":
        system = [{"role": "system", "content": PREFIX.strip()}]
        prefix_ids = tokenizer.apply_chat_template(
            system, tokenize=True, add_generation_prompt=False, enable_thinking=False,
        )
        input_ids = tokenizer.apply_chat_template(
            system + [{"role": "user", "content": history + question}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if input_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Chat template must preserve the leading system message")
    elif prompt_format == "completion":
        # Keep the original segment tokenization so previous base-model runs
        # remain comparable. Joint BPE encoding can change boundary tokens.
        prefix_ids, history_ids, question_ids = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in (PREFIX, history, question + "\nAnswer:")
        ]
        input_ids = prefix_ids + history_ids + question_ids
    else:
        raise ValueError("prompt_format must be completion or chat")

    prepared = {
        **example,
        "answer": str(example["answer"]),
        "abstention": example["question_id"].endswith("_abs"),
        "input_ids": input_ids,
        "persistent_prefix_length": len(prefix_ids),
        "prompt_length": len(input_ids),
    }
    # Discard only obsolete derived columns from older prepared datasets.
    for name in ("history_ids", "question_ids", "history_length", "question_length"):
        prepared.pop(name, None)
    return prepared
