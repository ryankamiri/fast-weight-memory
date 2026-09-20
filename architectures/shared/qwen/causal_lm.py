from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel

from inference.generation import GenerationOutput, sample_token
from .state import QwenSessionState

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except ModuleNotFoundError as error:
    if error.name != "liger_kernel":
        raise
    LigerFusedLinearCrossEntropyLoss = None


@dataclass
class StatefulQwen3CausalLMOutput(CausalLMOutputWithPast):
    state: QwenSessionState | None = None


@dataclass
class StatefulQwen3BridgeMemoryOutput(StatefulQwen3CausalLMOutput):
    all_token_loss: Float[torch.Tensor, ""] | None = None
    delayed_answer_loss: Float[torch.Tensor, ""] | None = None


def causal_lm_loss(
    lm_head: nn.Linear,
    hidden_states: Float[torch.Tensor, "N D_model"],
    labels: Int[torch.Tensor, "B S"],
) -> Float[torch.Tensor, ""]:
    """Compute next-token loss without materializing full CUDA logits."""
    # N = B * (S - 1): every next-token prediction flattened across batch and sequence.
    targets: Int[torch.Tensor, "N"] = labels[:, 1:].reshape(-1).to(
        hidden_states.device
    )
    if hidden_states.is_cuda:
        if LigerFusedLinearCrossEntropyLoss is None:
            raise ImportError("CUDA loss requires Liger.")
        return LigerFusedLinearCrossEntropyLoss(accum_dtype=torch.float32)(
            lm_head.weight,
            hidden_states,
            targets,
        )
    return F.cross_entropy(lm_head(hidden_states).float(), targets)


class StatefulQwen3ForCausalLM(Qwen3PreTrainedModel, ABC):
    """Shared language-model mechanics for stateful Qwen architectures."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = self._build_model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @abstractmethod
    def _build_model(self, config: Qwen3Config) -> nn.Module:
        """Construct the architecture-specific stateful Qwen backbone."""

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @staticmethod
    def _validate_lm_inputs(
        input_ids: Int[torch.Tensor, "B S"],
        labels: Int[torch.Tensor, "B S"] | None,
        logits_to_keep: int,
    ) -> None:
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids")

    def _causal_head(
        self,
        hidden_states: Float[torch.Tensor, "B S D_model"],
        labels: Int[torch.Tensor, "B S"] | None,
        logits_to_keep: int,
    ) -> tuple[
        Float[torch.Tensor, ""] | None,
        Float[torch.Tensor, "B S_logits vocab_size"] | None,
    ]:
        loss = None
        if labels is not None:
            # Hidden state at t predicts label at t+1; N = B * (S - 1).
            flattened: Float[torch.Tensor, "N D_model"] = (
                hidden_states[:, :-1].reshape(-1, self.config.hidden_size)
            )
            loss = causal_lm_loss(self.lm_head, flattened, labels)

        logits = None
        if labels is None or logits_to_keep > 0:
            # [B, S_logits, D_model] -> [B, S_logits, vocab_size]; -0 keeps all S.
            logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])
        return loss, logits

    def _bridge_head(
        self,
        hidden_states: Float[torch.Tensor, "B S D_model"],
        delayed_labels: Int[torch.Tensor, "B S"],
        all_token_loss_weight: float,
        delayed_answer_loss_weight: float,
        labels: Int[torch.Tensor, "B S"] | None,
        logits_to_keep: int,
    ) -> tuple[
        Float[torch.Tensor, ""],
        Float[torch.Tensor, "B S_logits vocab_size"] | None,
        Float[torch.Tensor, ""] | None,
        Float[torch.Tensor, ""],
    ]:
        for name, weight in (
            ("all_token_loss_weight", all_token_loss_weight),
            ("delayed_answer_loss_weight", delayed_answer_loss_weight),
        ):
            if type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if delayed_answer_loss_weight == 0:
            raise ValueError("delayed_answer_loss_weight must be positive")
        if delayed_labels.shape[:2] != hidden_states.shape[:2]:
            raise ValueError("delayed_labels must match input_ids")

        flattened: Float[torch.Tensor, "N D_model"] = (
            hidden_states[:, :-1].reshape(-1, self.config.hidden_size)
        )
        all_token_loss = None
        terms = []
        if labels is not None and all_token_loss_weight > 0:
            all_token_loss = causal_lm_loss(self.lm_head, flattened, labels)
            terms.append(all_token_loss_weight * all_token_loss)
        delayed_answer_loss = causal_lm_loss(
            self.lm_head,
            flattened,
            delayed_labels,
        )
        terms.append(delayed_answer_loss_weight * delayed_answer_loss)
        loss = torch.stack(terms).sum()

        logits = None
        if logits_to_keep > 0:
            logits = self.lm_head(hidden_states[:, -logits_to_keep:, :])
        return loss, logits, all_token_loss, delayed_answer_loss

    def _validate_generation(
        self,
        input_ids: Int[torch.Tensor, "1 S"],
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_token_id: int | list[int] | None,
        generator: torch.Generator | None,
    ) -> list[int]:
        if self.training:
            raise ValueError("generate requires evaluation mode; call model.eval() first")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if do_sample:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive")
            if not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            if type(top_k) is not int or top_k < 0:
                raise ValueError("top_k must be a nonnegative integer; 0 disables filtering")
            if generator is not None and torch.device(generator.device) != input_ids.device:
                raise ValueError("generator must be on the input device")
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        eos_ids = [] if eos_token_id is None else eos_token_id
        if type(eos_ids) is int:
            eos_ids = [eos_ids]
        if any(
            type(token) is not int or not 0 <= token < self.vocab_size
            for token in eos_ids
        ):
            raise ValueError("EOS token IDs must be valid vocabulary IDs")
        return eos_ids

    def _generate_from_prefill(
        self,
        output: StatefulQwen3CausalLMOutput,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
        eos_ids: list[int],
        generator: torch.Generator | None,
        decode_token: Callable[
            [Int[torch.Tensor, "1 1"], QwenSessionState],
            StatefulQwen3CausalLMOutput,
        ],
    ) -> GenerationOutput[QwenSessionState]:
        if output.logits is None or output.state is None:
            raise ValueError("Prefill must return logits and complete session state")
        generated = []
        stop_reason = "max_new_tokens"
        for _ in range(max_new_tokens):
            next_token = sample_token(
                output.logits[:, -1, :],
                do_sample,
                temperature,
                top_k,
                top_p,
                generator,
            )
            generated.append(next_token)
            output = decode_token(next_token, output.state)
            if next_token.item() in eos_ids:
                stop_reason = "eos"
                break
        return GenerationOutput(
            token_ids=torch.cat(generated, dim=1),
            state=output.state,
            stop_reason=stop_reason,
        )
