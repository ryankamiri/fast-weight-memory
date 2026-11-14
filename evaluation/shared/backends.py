"""Adapters for the existing Qwen hosts; no memory update equations live here."""

import hashlib
import inspect
import json
from dataclasses import asdict

import torch
from jaxtyping import Int

from architectures.taal.qwen.configuration import TaalQwen3Config
from evaluation.models import validate_read_scale
from evaluation.shared.contracts import Artifact, Backend, Capabilities, Prediction
from evaluation.shared.generation import capture_generation, resolve_generation_settings
from evaluation.storage import atomic_json
from evaluation.taal.trace_export import TaalTraceExporter, validate_generation_trace


class QwenBackend(Backend):
    def __init__(
        self,
        model,
        tokenizer,
        provenance: dict,
        generation: dict,
        memory_read_scale: float = 1.0,
        retain_task_prefix: bool = False,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.provenance = provenance
        self.generation = dict(generation)
        self.is_taal = isinstance(model.config, TaalQwen3Config)
        self.capabilities = Capabilities(
            memory_trace=self.is_taal, read_scale=self.is_taal
        )
        self.memory_read_scale = memory_read_scale
        self.retain_task_prefix = retain_task_prefix
        if type(retain_task_prefix) is not bool:
            raise ValueError("retain_task_prefix must be boolean")

        validate_read_scale(memory_read_scale)

        if not self.is_taal and memory_read_scale != 1:
            raise ValueError(
                "Whole-episode memory_read_scale is only supported by the TaaL backend"
            )

        if self.is_taal:
            self.generation = resolve_generation_settings(
                self.generation, model.config.working_memory_size
            )
        reserved = {
            "state",
            "input_ids",
            "persistent_mask",
            "generator",
            "write_mask",
            "memory_read_scale",
            "prepend_memory_tokens",
        }
        unknown = set(self.generation) - set(
            inspect.signature(model.generate).parameters
        )
        if unknown or reserved.intersection(self.generation):
            raise ValueError(
                f"Unsupported generation settings: {sorted(unknown | reserved.intersection(self.generation))}"
            )

        if type(self.generation.get("do_sample", True)) is not bool:
            raise ValueError("do_sample must be boolean")

        block_size = self.generation.get("execution_block_size")
        if block_size is not None and (type(block_size) is not int or block_size < 1):
            raise ValueError("execution_block_size must be a positive integer or null")

        validation = {}
        defaults = inspect.signature(model.generate).parameters
        for name in (
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "eos_token_id",
        ):
            validation[name] = self.generation.get(name, defaults[name].default)
        model._validate_generation(
            torch.zeros((1, 1), dtype=torch.long), generator=None, **validation
        )
        self.exporter = None

    def identity(self):
        return {
            **self.provenance,
            "model_config": self.model.config.to_dict(),
            "generation": self.generation,
            "memory_read_scale": self.memory_read_scale,
            "read_scale_scope": "whole_episode",
            "retain_task_prefix": self.retain_task_prefix,
            "capabilities": asdict(self.capabilities),
            "timing_scope": "synchronized generation including recorder hooks; excludes artifact serialization",
            "session_policy": "fresh-state-full-prefix-replay-per-question",
        }

    def validate(self, prompts, save_traces):
        super().validate(prompts, save_traces=save_traces)
        for prompt in prompts:
            if max(prompt.token_ids) >= self.model.config.vocab_size:
                raise ValueError(
                    "Prompt contains token IDs outside the checkpoint vocabulary"
                )

            if not self.retain_task_prefix:
                continue

            maximum = self.model.config.max_persistent_kv_tokens
            if prompt.persistent_prefix_length > maximum:
                raise ValueError("Task prefix exceeds checkpoint persistent KV budget")

    def validate_saved(self, output, predictions):
        manifest = None
        for key, row in predictions.items():
            memory_artifacts = []
            for item in row.get("artifacts", []):
                if item["kind"] == "memory_trace":
                    memory_artifacts.append(item)
            if not memory_artifacts:
                continue
            root = output / "memory_traces"
            if manifest is None:
                path = root / "manifest.json"
                if not path.is_file():
                    raise ValueError(f"Missing TaaL trace index for {key}")

                manifest = json.loads(path.read_text())
            expected_path = output / memory_artifacts[0]["path"]
            validate_generation_trace(root, manifest, key, row, expected_path)

    @torch.inference_mode()
    def generate(self, example_id, prompt, output, seed, save_traces):
        self.validate([prompt], save_traces=save_traces)
        device = next(self.model.parameters()).device
        ids: Int[torch.Tensor, "1 S"] = torch.tensor(
            [prompt.token_ids], dtype=torch.long, device=device
        )
        S = len(prompt.token_ids)
        settings = {
            **self.generation,
            "generator": torch.Generator(device=device).manual_seed(seed),
        }
        if self.retain_task_prefix:
            settings["persistent_mask"] = (
                torch.arange(S, device=device) < prompt.persistent_prefix_length
            )
        if self.is_taal:
            settings["memory_read_scale"] = self.memory_read_scale
        captured = capture_generation(
            self.model, ids, settings, trace_memory=save_traces and self.is_taal
        )
        generated_ids = captured.token_ids
        recorder = captured.recorder
        prediction = Prediction(
            example_id,
            self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            generated_ids,
            captured.stop_reason,
            S,
            len(generated_ids),
            captured.seconds,
            captured.peak_allocated_bytes,
        )
        if save_traces:
            key = hashlib.sha256(example_id.encode()).hexdigest()
            path = output / "token_traces" / f"{key}.json"
            atomic_json(
                path,
                {
                    "schema": "evaluation-token-trace/v1",
                    "example_id": example_id,
                    "prompt_token_ids": list(prompt.token_ids),
                    "generated_token_ids": generated_ids,
                    "hypothesis": prediction.hypothesis,
                    "stop_reason": prediction.stop_reason,
                },
            )
            prediction.artifacts.append(Artifact.from_path(path, output, "token_trace"))
            if recorder is not None:
                if (
                    self.exporter is None
                    or self.exporter.root != output / "memory_traces"
                ):
                    self.exporter = TaalTraceExporter(
                        output / "memory_traces",
                        run_metadata={
                            "checkpoint": self.provenance["checkpoint"],
                            "tokenizer": self.provenance["tokenizer"],
                            "layers": list(range(self.model.config.num_hidden_layers)),
                            "evaluation": "shared",
                            "backend": json.loads(json.dumps(self.identity())),
                        },
                    )
                trace_path = self.exporter.export_generation(
                    example_id,
                    list(prompt.token_ids),
                    generated_ids,
                    self.tokenizer,
                    recorder,
                    outcome={
                        "hypothesis": prediction.hypothesis,
                        "generated_token_ids": generated_ids,
                    },
                    metadata={
                        "read_scale_scope": "whole_episode",
                        "memory_read_scale": self.memory_read_scale,
                        "timing_note": "Generated-token events follow the prediction that chose that token",
                    },
                )
                prediction.artifacts.append(
                    Artifact.from_path(trace_path, output, "memory_trace")
                )
        return prediction
