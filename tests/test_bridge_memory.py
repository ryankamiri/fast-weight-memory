import json

import torch

from evaluation.bridge_data import BridgeGeometry, build_bridge_example
from evaluation.bridge_memory import score_logits, summarize
from evaluation.storage import read_results


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(token) if token < 256 else f"<{token}>" for token in ids)


def make_example(condition, query_variant="exact"):
    return build_bridge_example(
        CharacterTokenizer(),
        [7] * 5000,
        fact_index=3,
        answer_text="4827",
        target_token_id=999,
        candidate_token_ids=[997, 998, 999, 1000],
        condition=condition,
        query_variant=query_variant,
        geometry=BridgeGeometry(teacher_window_size=512, student_window_size=256, chunk_size=128),
    )


def test_bridge_and_no_bridge_are_position_matched():
    bridge = make_example("bridge")
    control = make_example("no_bridge")

    assert bridge["prompt_length"] == control["prompt_length"]
    assert bridge["final_query_position"] == control["final_query_position"]
    assert (
        bridge["input_ids"][:bridge["bridge_segment_start"]]
        == control["input_ids"][:control["bridge_segment_start"]]
    )
    assert bridge["input_ids"][bridge["bridge_segment_end"]:] == control["input_ids"][control["bridge_segment_end"]:]
    assert bridge["input_ids"].count(999) == 2
    assert control["input_ids"].count(999) == 1
    assert control["bridge_answer_position"] is None


def test_positions_enforce_teacher_only_bridge_and_final_eviction():
    geometry = BridgeGeometry(teacher_window_size=512, student_window_size=256, chunk_size=128)
    bridge = make_example("bridge", "paraphrased")
    visible = make_example("visible")

    bridge_gap = bridge["bridge_answer_position"] - bridge["fact_position"]
    assert geometry.student_window_size < bridge_gap < geometry.teacher_window_size
    assert bridge["final_answer_position"] - bridge["bridge_answer_position"] > geometry.teacher_window_size
    visible_gap = visible["final_answer_position"] - visible["fact_position"]
    assert geometry.student_window_size < visible_gap < geometry.teacher_window_size


def test_geometry_is_configurable_and_validated():
    geometry = BridgeGeometry(teacher_window_size=2048, student_window_size=1024, chunk_size=512)
    geometry.validate()
    assert geometry.config_name == "t2048-s1024-c512"
    assert geometry.bridge_distance == 1536
    assert geometry.eviction_distance == 2304

    try:
        BridgeGeometry(teacher_window_size=1024, student_window_size=1024).validate()
    except ValueError as error:
        assert "smaller" in str(error)
    else:
        raise AssertionError("Equal teacher and student windows must be rejected")


def test_objective_scoring_and_summary():
    example = make_example("bridge")
    logits = torch.zeros(1100)
    logits[999] = 4
    logits[998] = 2
    scored = score_logits(logits, example, CharacterTokenizer())

    assert scored["target_rank"] == 1
    assert scored["candidate_correct"]
    assert scored["vocabulary_top_1_correct"]
    row = {
        **scored,
        "condition": "bridge",
        "query_variant": "exact",
    }
    summary = summarize([row])
    assert summary["bridge/exact"]["candidate_accuracy"] == 1


def test_result_reader_can_use_example_ids(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"example_id": "example-1", "value": 4}) + "\n")
    assert read_results(path, id_field="example_id")["example-1"]["value"] == 4
