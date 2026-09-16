from dataclasses import dataclass


LABEL_WORDS = """
amber anchor apple arrow atlas bamboo basket beacon berry blossom bottle breeze bridge bronze
bucket button candle canyon cedar cherry circle cloud comet coral cotton crystal delta desert
dolphin ember feather fern forest fossil garden ginger glacier harbor honey island ivory jacket
kettle lantern lemon linen maple marble mint mirror mosaic mountain nickel ocean olive paper pepper
pine planet plum pocket prism quartz ribbon river robin rose ruby saddle shadow silver spice stone
sunset timber valley velvet violet walnut winter almond basil cabin canvas caramel castle citrus
copper creek crown field flame frost galaxy hawk iron jade lake lime lunar mango moss oasis orange
pearl pond rain reef ridge sage shell shore slate spring star storm teal tide tower trail vine wave
wheat wood wool cardinal crane deer eagle fox lizard panda rabbit salmon tiger turtle whale wolf
barley bean cocoa coffee corn fig grape maize onion peach pear pumpkin rice sesame squash vanilla
""".split()


@dataclass(frozen=True)
class BridgeGeometry:
    teacher_window_size: int = 4096
    student_window_size: int = 2048
    chunk_size: int = 1024

    def validate(self):
        for name, value in vars(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.student_window_size >= self.teacher_window_size:
            raise ValueError("student_window_size must be smaller than teacher_window_size")
        if self.teacher_window_size - self.student_window_size < 2:
            raise ValueError("teacher and student windows need an interior teacher-only position")

    @property
    def bridge_distance(self):
        """Place the bridge midway through the teacher-only region."""
        return self.student_window_size + (
            self.teacher_window_size - self.student_window_size
        ) // 2

    @property
    def eviction_distance(self):
        """Place the final answer beyond the teacher window from the bridge."""
        return self.teacher_window_size + max(1, self.chunk_size // 2)

    @property
    def config_name(self):
        return (
            f"t{self.teacher_window_size}-s{self.student_window_size}"
            f"-c{self.chunk_size}"
        )


def single_token_labels(tokenizer, count):
    """Return stable word labels represented by one leading-space token."""
    labels = []
    for word in LABEL_WORDS:
        ids = tokenizer.encode(f" {word}", add_special_tokens=False)
        if len(ids) != 1 or tokenizer.decode(ids) != f" {word}":
            continue
        labels.append((word, ids[0]))
        if len(labels) == count:
            return labels
    raise ValueError(f"Tokenizer supplied only {len(labels)} usable one-token labels")


def _encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _take_filler(filler_ids, start, length):
    end = start + length
    if length < 0:
        raise ValueError("Geometry leaves insufficient room for a prompt segment")
    if end > len(filler_ids):
        raise ValueError(f"Filler needs at least {end} tokens; received {len(filler_ids)}")
    return filler_ids[start:end], end


def _templates(record_name):
    fact_prefix = f"User: Remember that {record_name}'s archive label is"
    fact_suffix = ".\nAssistant: Understood.\n\n"
    bridge_prefix = (
        f"User: What is {record_name}'s archive label?\n"
        "Assistant: The archive label is"
    )
    bridge_suffix = ".\n\n"
    final = {
        "exact": (
            f"User: What is {record_name}'s archive label?\n"
            "Assistant: The archive label is"
        ),
        "paraphrased": (
            f"User: Remind me of the word assigned to {record_name}.\n"
            "Assistant: That word is"
        ),
    }
    return fact_prefix, fact_suffix, bridge_prefix, bridge_suffix, final


def build_bridge_example(
    tokenizer,
    filler_ids,
    *,
    fact_index,
    answer_text,
    target_token_id,
    candidate_token_ids,
    condition,
    query_variant,
    geometry,
):
    """Build one prompt whose answer is the token immediately after ``input_ids``."""
    geometry.validate()
    if condition not in {"visible", "no_bridge", "bridge"}:
        raise ValueError("condition must be visible, no_bridge, or bridge")
    if query_variant not in {"exact", "paraphrased"}:
        raise ValueError("query_variant must be exact or paraphrased")
    if condition == "visible" and query_variant != "exact":
        raise ValueError("visible controls use the exact query")
    if target_token_id not in candidate_token_ids:
        raise ValueError("candidate_token_ids must contain the target")
    if len(candidate_token_ids) != len(set(candidate_token_ids)):
        raise ValueError("candidate_token_ids must be unique")

    record_name = f"Record R{fact_index:04d}"
    fact_prefix, fact_suffix, bridge_prefix, bridge_suffix, final = _templates(record_name)
    fact_prefix_ids = _encode(tokenizer, fact_prefix)
    fact_ids = fact_prefix_ids + [target_token_id] + _encode(tokenizer, fact_suffix)
    fact_position = len(fact_prefix_ids)
    final_prefix_ids = _encode(tokenizer, final[query_variant])

    if condition == "visible":
        final_answer_position = fact_position + geometry.bridge_distance
        filler_length = final_answer_position - len(fact_ids) - len(final_prefix_ids)
        filler, _ = _take_filler(filler_ids, 0, filler_length)
        final_query_position = len(fact_ids) + len(filler)
        input_ids = fact_ids + filler + final_prefix_ids
        bridge_answer_position = None
        bridge_segment_start = None
        bridge_segment_end = None
    else:
        bridge_prefix_ids = _encode(tokenizer, bridge_prefix)
        bridge_suffix_ids = _encode(tokenizer, bridge_suffix)
        bridge_segment = bridge_prefix_ids + [target_token_id] + bridge_suffix_ids

        bridge_anchor_position = fact_position + geometry.bridge_distance
        filler_one_length = (
            bridge_anchor_position - len(fact_ids) - len(bridge_prefix_ids)
        )
        filler_one, cursor = _take_filler(filler_ids, 0, filler_one_length)
        bridge_segment_start = len(fact_ids) + len(filler_one)
        bridge_segment_end = bridge_segment_start + len(bridge_segment)
        neutral_segment, cursor = _take_filler(filler_ids, cursor, len(bridge_segment))
        selected_segment = bridge_segment if condition == "bridge" else neutral_segment

        final_answer_position = bridge_anchor_position + geometry.eviction_distance
        filler_two_length = (
            final_answer_position
            - (len(fact_ids) + len(filler_one) + len(selected_segment))
            - len(final_prefix_ids)
        )
        filler_two, _ = _take_filler(filler_ids, cursor, filler_two_length)
        final_query_position = (
            len(fact_ids) + len(filler_one) + len(selected_segment) + len(filler_two)
        )
        input_ids = fact_ids + filler_one + selected_segment + filler_two + final_prefix_ids
        bridge_answer_position = bridge_anchor_position if condition == "bridge" else None

    if len(input_ids) != final_answer_position:
        raise AssertionError("Final answer position does not match the prompt length")
    if condition == "bridge":
        bridge_gap = bridge_answer_position - fact_position
        if not geometry.student_window_size < bridge_gap < geometry.teacher_window_size:
            raise AssertionError("Fact is not in the teacher-only region at the bridge")
        if final_answer_position - bridge_answer_position <= geometry.teacher_window_size:
            raise AssertionError("Bridge remains visible at the final answer")
    if condition == "no_bridge" and final_answer_position - fact_position <= geometry.teacher_window_size:
        raise AssertionError("Fact remains visible in the no-bridge control")
    if condition == "visible":
        visible_gap = final_answer_position - fact_position
        if not geometry.student_window_size < visible_gap < geometry.teacher_window_size:
            raise AssertionError("Visible control is not in the teacher-only region")

    return {
        "example_id": f"fact-{fact_index:04d}-{condition}-{query_variant}",
        "fact_id": fact_index,
        "record_name": record_name,
        "answer": answer_text,
        "condition": condition,
        "query_variant": query_variant,
        "input_ids": input_ids,
        "target_token_id": target_token_id,
        "candidate_token_ids": list(candidate_token_ids),
        "fact_position": fact_position,
        "bridge_answer_position": bridge_answer_position,
        "bridge_segment_start": bridge_segment_start,
        "bridge_segment_end": bridge_segment_end,
        "final_query_position": final_query_position,
        "final_answer_position": final_answer_position,
        "prompt_length": len(input_ids),
        "teacher_window_size": geometry.teacher_window_size,
        "student_window_size": geometry.student_window_size,
        "chunk_size": geometry.chunk_size,
    }
