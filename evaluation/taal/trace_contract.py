from dataclasses import asdict, dataclass, field
from typing import Literal


SCHEMA_VERSION = "taal-memory-trace/v1"


@dataclass(frozen=True)
class TraceToken:
    position: int
    token_id: int
    text: str
    phase: Literal["prompt", "generated"]


@dataclass(frozen=True)
class WriteEvent:
    layer: int
    position: int | None
    segment_index: int
    internal_index: int | None
    write_enabled: bool
    write_strength: float
    proposed_write_norm: float
    previous_weight_norm: float
    net_weight_change_norm: float
    other_movement_norm: float | None
    write_to_net_alignment: float | None
    chunk_size: int = 1
    chunk_boundary: bool = True
    updates_enabled: bool = True


@dataclass(frozen=True)
class ReadEvent:
    layer: int
    position: int
    segment_index: int
    enabled: bool
    read_scale: float
    residual_gate: float
    injected_norm: float
    incoming_norm: float
    relative_injection: float | None
    state_timing: Literal["pre-write", "post-write"] = "post-write"


@dataclass(frozen=True)
class InternalPrefix:
    layer: int
    segment_index: int
    before_position: int
    count: int
    net_weight_change_norm: float


@dataclass(frozen=True)
class TraceComparison:
    run_id: str
    example_id: str
    baseline_condition: str
    variant_condition: str
    intervention: Literal["read_scale", "memory_state"]
    scope: Literal["whole_query"]
    identical_text_prefix: bool
    same_starting_kv: bool
    same_starting_memory: bool
    scored_position: int
    scored_token_id: int
    baseline_log_probability: float
    variant_log_probability: float
    difference_log_probability: float
    schema: str = SCHEMA_VERSION


@dataclass
class TraceEpisode:
    run_id: str
    example_id: str
    condition_id: str
    checkpoint: str
    tokenizer_id: str
    tokens: list[TraceToken]
    writes: list[WriteEvent]
    reads: list[ReadEvent]
    internal_prefixes: list[InternalPrefix]
    outcome: dict | None = None
    metadata: dict = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"Unsupported trace schema: {self.schema}")
        positions = [token.position for token in self.tokens]
        if positions != sorted(set(positions)):
            raise ValueError("Visible token positions must be unique and ordered")
        visible = set(positions)
        for event in self.writes:
            if (event.position is None) == (event.internal_index is None):
                raise ValueError("Write must belong to one visible or internal token")
            if event.position is not None and event.position not in visible:
                raise ValueError("Write refers to an absent visible token")
        for event in self.reads:
            if event.position not in visible:
                raise ValueError("Read refers to an absent visible token")
        for prefix in self.internal_prefixes:
            if prefix.before_position not in visible:
                raise ValueError("Internal prefix has no visible anchor")
        return asdict(self)
