import argparse
from dataclasses import asdict
import os
import re
import signal
from threading import Event

from huggingface_hub import HfApi
import torch
from transformers import Qwen3Config
import wandb

from architectures.ttcd.qwen.causal_lm import TTCDQwen3ForCausalLM
from architectures.ttcd.qwen.configuration import TTCDQwen3Config
from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.taal.qwen.configuration import TaalQwen3Config
from data.dataloader import create_dataloader
from utils.seed import seed_everything
from .config import TaalModelConfig, TrainingConfig, TTCDModelConfig, load_config
from .engine import Progress, build_scheduler, train
from .checkpoints import ModelCheckpoints


def verify_loading(model, loading_info):
    allowed_missing = set()
    if isinstance(model.config, TTCDQwen3Config):
        for index in model.config.fast_weight_layers:
            mlp = model.model.layers[index].mlp
            for name in ("W_proj", "beta_proj", "teacher_conv", "student_conv"):
                value = getattr(mlp, name, None)
                if value is not None:
                    suffix = f"{name}.weight" if isinstance(value, torch.nn.Conv1d) else name
                    allowed_missing.add(f"model.layers.{index}.mlp.{suffix}")
    elif isinstance(model.config, TaalQwen3Config):
        allowed_missing = {
            name for name, _ in model.named_parameters() if ".taal." in name
        }
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
    unexpected_missing = set(loading_info.get("missing_keys", [])) - allowed_missing
    errors = {
        "missing_base_weights": sorted(unexpected_missing),
        "unexpected_keys": loading_info.get("unexpected_keys", []),
        "mismatched_keys": loading_info.get("mismatched_keys", []),
        "error_msgs": loading_info.get("error_msgs", []),
    }
    if any(errors.values()):
        raise ValueError(f"Pretrained weight loading failed verification: {errors}")
    print(f"Loaded Qwen weights; initialized {len(loading_info.get('missing_keys', []))} added tensors.", flush=True)


def load_model(config: TrainingConfig):
    settings = config.model
    base = Qwen3Config.from_pretrained(settings.model_id, revision=settings.revision)
    overrides = asdict(settings)
    for name in ("model_id", "revision"):
        overrides.pop(name)
    if isinstance(settings, TTCDModelConfig):
        overrides["lr"] = overrides.pop("fast_weight_lr")
        model_config = TTCDQwen3Config.from_dict(base.to_dict() | overrides)
        model_type = TTCDQwen3ForCausalLM
    elif isinstance(settings, TaalModelConfig):
        model_config = TaalQwen3Config.from_dict(base.to_dict() | overrides)
        model_type = TaalQwen3ForCausalLM
    else:
        raise TypeError(f"Unsupported model config: {type(settings).__name__}")
    model, loading_info = model_type.from_pretrained(
        settings.model_id, revision=settings.revision, config=model_config,
        dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True,
    )
    verify_loading(model, loading_info)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model


def configure_trainable_parameters(model, scope: str):
    """Select optimizer parameters without changing the serialized architecture."""
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif scope == "ttcd_only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for index in model.config.fast_weight_layers:
            mlp = model.model.layers[index].mlp
            for name in ("W_proj", "beta_proj", "teacher_conv", "student_conv"):
                value = getattr(mlp, name, None)
                if isinstance(value, torch.nn.Parameter):
                    value.requires_grad_(True)
                elif isinstance(value, torch.nn.Module):
                    for parameter in value.parameters():
                        parameter.requires_grad_(True)
    elif scope == "taal_only":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if ".taal." in name:
                parameter.requires_grad_(True)
    else:
        raise ValueError(f"Unsupported trainable parameter scope: {scope}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters remain after applying the parameter scope")
    trainable = sum(parameter.numel() for parameter in parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%).", flush=True)
    return parameters


def resolve_revision(api: HfApi, repo_id: str, revision: str | None, *, dataset: bool) -> str:
    """Keep immutable commit pins without requiring a network lookup at launch."""
    if revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    if dataset:
        return api.dataset_info(repo_id, revision=revision).sha
    return api.model_info(repo_id, revision=revision).sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/ttcd/qwen3_0_6b.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Only one GPU is supported; distributed training is not implemented")
    if not torch.cuda.is_available():
        raise RuntimeError("This training command requires a CUDA GPU")
    
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This training recipe requires BF16 support")
    seed_everything(config.training.seed)

    # Resolve symbolic refs once; an already-pinned commit needs no network call.
    api = HfApi()
    config.model.revision = resolve_revision(
        api, config.model.model_id, config.model.revision, dataset=False,
    )
    config.data.revision = resolve_revision(
        api, config.data.dataset_id, config.data.revision, dataset=True,
    )
    model = load_model(config).to(device)
    trainable_parameters = configure_trainable_parameters(
        model, config.training.trainable_parameters,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    train_loader = create_dataloader(
        config.data, config.data.train, config.loss,
        shuffle=True, seed=config.training.seed,
    )
    val_loader = create_dataloader(
        config.data, config.data.val, config.loss,
        shuffle=False, seed=config.training.seed,
    )

    stop = Event()

    def request_stop(signum, frame):
        if stop.is_set():
            return
        print("Stop requested. Skipping final validation and saving the final model at the next safe boundary.", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        run_name = config.wandb.name or config.wandb.project
        config.wandb.name = f"{run_name}-{slurm_job_id}"

    print(f"Training on {torch.cuda.get_device_name(device)} for up to {config.training.max_steps} optimizer updates.", flush=True)
    
    with wandb.init(**asdict(config.wandb), config=config.to_dict(), mode="online") as run:
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")

        def log(metrics):
            run.log(metrics)
            print(" | ".join(f"{key}={value:.6g}" for key, value in metrics.items()), flush=True)

        checkpoints = ModelCheckpoints(config, run.id)
        progress = Progress()
        reason = "exception"
        try:
            train(
                model, train_loader, val_loader, optimizer, scheduler, config, device, log,
                stop.is_set, on_validation=checkpoints.on_validation, progress=progress,
            )
            reason = "signal" if stop.is_set() else "completed"
        finally:
            if stop.is_set():
                reason = "signal"
            try:
                checkpoints.save_final(model, progress, reason)
            except Exception as error:
                if reason != "exception" and not stop.is_set():
                    raise
                print(f"Final model save failed: {error}", flush=True)


if __name__ == "__main__":
    main()
