"""Cross saved post-write weights and normalized queries at one memory read.

The model still follows the background writer trajectory. Substituted weights
are read-only: they never replace recurrent state or participate in its writes.
The outer adapter observes the actual substituted injection for trace export.
"""

from types import MethodType

import torch
from jaxtyping import Float

from evaluation.taal.interventions import ReadCrossPolicy
from evaluation.taal.query_write_replay import QueryWriteReplay


class ReadCrossReplay:
    def __init__(
        self, model, token_count: int, policy: ReadCrossPolicy,
        references: dict[str, QueryWriteReplay],
    ):
        if model.training or model.config.memory_chunk_size != 1:
            raise ValueError("Read crossing requires eval and memory_chunk_size=1")
        if (policy.layer != len(model.model.layers) - 1
                or policy.scope not in ("whole_query", "prediction_token")
                or type(token_count) is not int or token_count < 1):
            raise ValueError("Read crossing requires the last layer and a valid query scope")
        if set(references) != {"own", "donor"}:
            raise ValueError("Read crossing requires own and donor reference runs")
        for source in (policy.weights, policy.query):
            if source not in references:
                raise ValueError("Unknown read-cross reference source")
            reference = references[source]
            if (reference.token_count != token_count
                    or len(reference.events[policy.layer]) != token_count):
                raise ValueError("Read references must cover every prompt-query token")
        self.memory = model.model.layers[policy.layer].taal.neural_memory
        self.token_count = token_count
        self.policy = policy
        self.references = references
        self.count = 0
        self.events: list[dict] = []
        self.original = None

    def __enter__(self):
        self.original = self.memory._process_chunk

        def process(memory, state, queries, keys, values, inputs, write_mask, output_dtype):
            B, C, D = queries.shape
            index = self.count
            if (B, C) != (1, 1) or index >= self.token_count:
                raise ValueError("Read crossing requires exactly B=C=1 per query token")
            output, after = self.original(
                state, queries, keys, values, inputs, write_mask, output_dtype,
            )
            self.count += 1
            if self.policy.scope == "prediction_token" and index != self.token_count - 1:
                return output, after

            weight_event = self.references[self.policy.weights].events[self.policy.layer][index]
            query_event = self.references[self.policy.query].events[self.policy.layer][index]
            # Titans read y_t = M_{W_t}(q_t). W_t is the captured POST-write
            # state, q_t the captured post-convolution/SiLU/normalized query.
            result = memory.memory_mlp(
                query_event.queries.float(), weights=weight_event.weights_after,
            )
            patched: Float[torch.Tensor, "B C D"] = result.predicted_values.to(output_dtype)
            if not torch.isfinite(patched).all():
                raise ValueError(f"Nonfinite crossed read at query token {index}")
            self.events.append({
                "query_index": index,
                "natural_read_norm": float(output.float().norm().item()),
                "patched_read_norm": float(patched.float().norm().item()),
                "read_difference_norm": float((patched.float() - output.float()).norm().item()),
                "query_difference_norm": float((query_event.queries.float() - queries.float()).norm().item()),
            })
            return patched, after

        self.memory._process_chunk = MethodType(process, self.memory)
        return self

    def __exit__(self, error_type, *_):
        self.memory._process_chunk = self.original
        if error_type is None and self.count != self.token_count:
            raise ValueError("Read crossing did not consume every prompt-query token")
