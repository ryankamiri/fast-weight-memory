"""A fixed completion-style prompt for base Qwen checkpoints."""

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


def prepare_example(example, tokenizer):
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
    question = f"Current date: {example['question_date']}\nQuestion: {example['question']}\nAnswer:"
    # Encode the three explicit input segments separately. This defines our input
    # format and gives exact persistent/history/question boundaries, without
    # assuming that independently encoded pieces equal a joint BPE encoding.
    prefix_ids, history_ids, question_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in (PREFIX, history, question)
    ]
    return {
        **example,
        "answer": str(example["answer"]),
        "abstention": example["question_id"].endswith("_abs"),
        "history_ids": prefix_ids + history_ids,
        "question_ids": question_ids,
        "persistent_prefix_length": len(prefix_ids),
        "history_length": len(prefix_ids) + len(history_ids),
        "question_length": len(question_ids),
        "prompt_length": len(prefix_ids) + len(history_ids) + len(question_ids),
    }
