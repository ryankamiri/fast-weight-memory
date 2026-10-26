"""Measure nominal reads and the actual rounded residual change."""

from contextlib import ExitStack
import json
from pathlib import Path

import torch


class ReadRoundingRecorder:
    def __init__(self, model, path):
        self.model = model
        self.path = Path(path)
        self.stack = ExitStack()
        self.corrections = {}
        self.call = 0
        self.stream = None

    def correction_hook(self, index):
        def hook(module, args, output):
            self.corrections[index] = output.detach()

        return hook

    def adapter_hook(self, index):
        def hook(module, args, kwargs, output):
            incoming = args[0] if args else kwargs["hidden_states"]
            scale = (
                kwargs.get("memory_read_scale", 1.0) if module.reads_enabled else 0.0
            )
            nominal = scale * module.residual_gate.tanh() * self.corrections.pop(index)
            incoming = incoming[:, -1].detach()
            actual = output[0][:, -1].detach().double() - incoming.double()
            nominal = nominal[:, -1].double()
            packed = torch.stack(
                [
                    incoming.double().norm(),
                    nominal.norm(),
                    actual.norm(),
                    (actual - nominal).norm(),
                    (actual == 0).double().mean(),
                    (nominal != 0).double().mean(),
                    module.residual_gate.detach().double().tanh(),
                ]
            ).cpu()
            names = [
                "incoming_norm",
                "nominal_injection_norm",
                "actual_delta_norm",
                "rounding_error_norm",
                "unchanged_fraction",
                "nominal_nonzero_fraction",
                "gate",
            ]
            values = packed.tolist()
            metrics = {
                name: value if bool(torch.isfinite(packed[i])) else None
                for i, (name, value) in enumerate(zip(names, values))
            }
            self.stream.write(
                json.dumps(
                    {
                        "call": self.call,
                        "layer": index,
                        "last_token_of_call": True,
                        "read_scale": scale,
                        "metrics": metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            if index == len(self.model.model.layers) - 1:
                self.call += 1

        return hook

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.stack.enter_context(self.path.open("w"))
        for index, layer in enumerate(self.model.model.layers):
            handle = layer.taal.memory_projection_out.register_forward_hook(
                self.correction_hook(index)
            )
            self.stack.callback(handle.remove)
            handle = layer.taal.register_forward_hook(
                self.adapter_hook(index), with_kwargs=True
            )
            self.stack.callback(handle.remove)
        return self

    def __exit__(self, *error):
        self.stack.close()
