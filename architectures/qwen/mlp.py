import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from beartype import beartype
from jaxtyping import Float, jaxtyped
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP


from ..states.mlp_state import FWMLPState


class FWQwen3MLP(Qwen3MLP):
    """Qwen SwiGLU with optional stateful teacher/student fast-weight updates."""

    def __init__(
        self,
        config, 
        is_fast_weight_layer: bool = False,
        chunk_size: int = 1024, 
        lr: float = 0.3,
        use_projection: bool = True, 
        use_conv: bool = True,
        conv_kernel_size: int = 5, 
        dynamic_beta: bool = True,
    ):
        super().__init__(config)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        if type(conv_kernel_size) is not int or conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be a positive integer")
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        self.is_fast_weight_layer = is_fast_weight_layer
        self.chunk_size = chunk_size
        self.lr = float(lr)

        if is_fast_weight_layer:
            self.W_proj = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size)) if use_projection else None
            self.beta_proj = nn.Parameter(torch.empty(self.hidden_size)) if dynamic_beta else None
            self.teacher_conv = nn.Conv1d(
                self.intermediate_size,
                self.intermediate_size,
                conv_kernel_size,
                groups=self.intermediate_size,
                bias=False,
            ) if use_conv else None
            self.student_conv = nn.Conv1d(
                self.intermediate_size,
                self.intermediate_size,
                conv_kernel_size,
                groups=self.intermediate_size,
                bias=False,
            ) if use_conv else None
            self.reset_fast_weight_parameters()

    @torch.no_grad()
    def reset_fast_weight_parameters(self):
        """Initialize only the added learned parameters, not Qwen weights or session state."""
        if not self.is_fast_weight_layer:
            return
        if self.W_proj is not None:
            nn.init.eye_(self.W_proj)
        if self.beta_proj is not None:
            nn.init.zeros_(self.beta_proj)
        for conv in (self.teacher_conv, self.student_conv):
            if conv is not None:
                # Identity causal filter: the last tap multiplies this token.
                nn.init.zeros_(conv.weight)
                conv.weight[:, 0, -1] = 1

    @property
    def W_base(self):
        """Alias the pretrained down projection without duplicating parameters."""
        return self.down_proj.weight

    @staticmethod
    def _convolve(z, conv, history: Float[torch.Tensor, "B d_mlp S_conv"] | None):
        if conv is None:
            return z, None
        z = z.transpose(1, 2)  # [B, d_mlp, S]
        B, d_mlp, S = z.shape
        history_size = conv.kernel_size[0] - 1
        if history is None:
            history = z.new_zeros(B, d_mlp, history_size)
        inputs = torch.cat((history, z), dim=-1)
        # conv(inputs): [B, d_mlp, S]; transpose: [B, S, d_mlp].
        output: Float[torch.Tensor, "B S d_mlp"] = conv(inputs).transpose(1, 2)
        history = inputs[..., -history_size:] if history_size != 0 else inputs[..., :0]
        return output, history

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        hidden_states: Float[torch.Tensor, "B S d_model"],
        student_hidden_states: Float[torch.Tensor, "B S d_model"] | None = None,
        state: FWMLPState | None = None,
    ) -> Float[torch.Tensor, "B S d_model"] | tuple[Float[torch.Tensor, "B S d_model"], FWMLPState]:
        if not self.is_fast_weight_layer:
            if state is not None:
                raise ValueError("state requires is_fast_weight_layer=True")
            return super().forward(hidden_states)
        
        if student_hidden_states is None:
            raise ValueError("Fast-weight mode requires student_hidden_states")

        B, S, d_model = hidden_states.shape
        if state is None:
            state = FWMLPState(
                W_fast=hidden_states.new_zeros(B, d_model, self.intermediate_size, dtype=torch.float32)
            )
        else:
            # Fields are replaced below, never modified in-place.
            state = replace(state)

        z_teacher: Float[torch.Tensor, "B S d_mlp"] = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        z_student: Float[torch.Tensor, "B S d_mlp"] = self.act_fn(self.gate_proj(student_hidden_states)) * self.up_proj(student_hidden_states)
        
        beta: Float[torch.Tensor, "B S 1"] = (student_hidden_states @ self.beta_proj).unsqueeze(-1).sigmoid() if self.beta_proj is not None else hidden_states.new_ones(B, S, 1)

        # Chunk updates
        outputs: list[Float[torch.Tensor, "B S_chunk d_model"]] = []
        start = 0
        while start < S:
            end = min(S, start + self.chunk_size - state.pending_count)
            # S_chunk = end - start (may be shorter than a full chunk).
            z_teacher_hat: Float[torch.Tensor, "B S_chunk d_mlp"]
            z_student_hat: Float[torch.Tensor, "B S_chunk d_mlp"]
            z_teacher_hat, state.teacher_conv_state = self._convolve(
                z_teacher[:, start:end], self.teacher_conv, state.teacher_conv_state)
            z_student_hat, state.student_conv_state = self._convolve(
                z_student[:, start:end], self.student_conv, state.student_conv_state)

            # Read the incoming state BEFORE committing this chunk's writes.
            z_output: Float[torch.Tensor, "B S_chunk d_mlp"] = z_teacher[:, start:end]
            # [B, S_chunk, d_mlp] @ [B, d_mlp, d_model] -> [B, S_chunk, d_model].
            output: Float[torch.Tensor, "B S_chunk d_model"] = self.down_proj(z_output) + (
                z_output @ state.W_fast.to(z_output.dtype).transpose(1, 2)
            )
            outputs.append(output)

            diff: Float[torch.Tensor, "B S_chunk d_mlp"] = z_teacher_hat - z_student_hat
            correction: Float[torch.Tensor, "B S_chunk d_model"] = self.down_proj(diff)
            if self.W_proj is not None:
                # [B, S_chunk, d_model] @ [d_model, d_model] -> [B, S_chunk, d_model].
                correction = correction @ self.W_proj
            # Scale each token's output correction by its learned write gate beta.
            r = (beta[:, start:end] * correction).float()
            # L2-normalize each token's student features to form its memory write key.
            k = F.normalize(z_student_hat, p=2, dim=-1, eps=1e-6).float()
            state.pending_r = r if state.pending_r is None else torch.cat((state.pending_r, r), dim=1)
            state.pending_k = k if state.pending_k is None else torch.cat((state.pending_k, k), dim=1)

            if state.pending_count == self.chunk_size:
                delta = self.lr * (
                    state.pending_r.transpose(1, 2) @ state.pending_k
                )
                state.W_fast = state.W_fast + delta
                state.pending_r = None
                state.pending_k = None
                state.teacher_conv_state = None
                state.student_conv_state = None
            start = end
        return torch.cat(outputs, dim=1), state
