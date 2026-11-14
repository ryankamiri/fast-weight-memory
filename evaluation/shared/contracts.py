"""Model-independent inputs and outputs for ordinary benchmark evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.storage import file_sha256


@dataclass(frozen=True)
class Prompt:
    # Gold answers live in Example, never in the object passed to a backend.
    token_ids: tuple[int, ...]
    persistent_prefix_length: int = 0

    def __post_init__(self):
        if not self.token_ids:
            raise ValueError("A prompt requires nonempty token IDs")

        for token in self.token_ids:
            if type(token) is not int or token < 0:
                raise ValueError("Prompt token IDs must be nonnegative integers")

        prefix_length = self.persistent_prefix_length
        if type(prefix_length) is not int:
            raise ValueError("Persistent prefix length must be an integer")

        if not 0 <= prefix_length <= len(self.token_ids):
            raise ValueError("Persistent prefix must fit within the prompt")


@dataclass(frozen=True)
class Example:
    example_id: str
    prompt: Prompt
    references: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.example_id, str) or not self.example_id:
            raise ValueError("example_id must be a nonempty string")

        if not self.references:
            raise ValueError("Examples require references")

        for answer in self.references:
            if not isinstance(answer, str) or not answer:
                raise ValueError("References must be nonempty strings")

        if self.group_id is not None and (
            not isinstance(self.group_id, str) or not self.group_id
        ):
            raise ValueError("group_id must be a nonempty string when supplied")


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str
    kind: str

    @classmethod
    def from_path(cls, path: Path, root: Path, kind: str) -> "Artifact":
        return cls(str(path.relative_to(root)), file_sha256(path), kind)


@dataclass
class Prediction:
    example_id: str
    hypothesis: str
    generated_token_ids: list[int]
    stop_reason: str
    prompt_tokens: int
    generated_tokens: int
    seconds: float
    peak_allocated_bytes: int | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    status: str = "completed"


@dataclass(frozen=True)
class Capabilities:
    token_trace: bool = True
    memory_trace: bool = False
    read_scale: bool = False
    continuation_score: bool = False
    snapshot: bool = False


class Backend(ABC):
    """Backends own model state. Each generate call starts an independent episode."""

    capabilities: Capabilities

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Return checkpoint, tokenizer, precision and execution provenance."""

    @abstractmethod
    def generate(
        self,
        example_id: str,
        prompt: Prompt,
        output: Path,
        seed: int,
        save_traces: bool,
    ) -> Prediction:
        """Generate without receiving references; save backend-specific artifacts."""

    def require(self, capabilities: list[str]):
        for name in capabilities:
            if (
                not hasattr(self.capabilities, name)
                or getattr(self.capabilities, name) is not True
            ):
                raise ValueError(
                    f"Backend does not support required capability: {name}"
                )

    def validate_saved(self, output: Path, predictions: dict):
        """Validate optional backend-owned trace indices before accepting resume."""

    def validate(self, prompts: list[Prompt], save_traces: bool):
        if type(save_traces) is not bool:
            raise ValueError("save_traces must be boolean")

        if save_traces and not self.capabilities.token_trace:
            raise ValueError("This backend cannot provide required token traces")


class Benchmark(ABC):
    name: str
    prompt_version: str
    scorer_version: str

    @abstractmethod
    def prepare(self, row: dict, tokenizer) -> Example:
        pass

    @abstractmethod
    def score(self, example: Example, hypothesis: str) -> dict[str, float]:
        pass
