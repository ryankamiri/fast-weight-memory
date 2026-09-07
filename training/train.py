import argparse
from dataclasses import asdict
import os
import random
import signal
from threading import Event

from huggingface_hub import HfApi
import numpy as np
import torch
from transformers import Qwen3Config
import wandb

from architectures.qwen.causal_lm import FWQwen3ForCausalLM
from architectures.qwen.configuration import FWQwen3Config
from data.dataloader import create_dataloader
from .config import TrainingConfig, load_config
from .engine import build_scheduler, fast_weight_metrics, train


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def verify_loading(model, loading_info):
    allowed_missing = set()
    for index in model.config.fast_weight_layers:
        mlp = model.model.layers[index].mlp
        for name in ("W_proj", "beta_proj", "teacher_conv", "student_conv"):
            value = getattr(mlp, name, None)
            if value is not None:
                suffix = f"{name}.weight" if isinstance(value, torch.nn.Conv1d) else name
                allowed_missing.add(f"model.layers.{index}.mlp.{suffix}")
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
    overrides["lr"] = overrides.pop("fast_weight_lr")
    model_config = FWQwen3Config.from_dict(base.to_dict() | overrides)
    model, loading_info = FWQwen3ForCausalLM.from_pretrained(
        settings.model_id, revision=settings.revision, config=model_config,
        dtype=torch.float32, attn_implementation="sdpa", output_loading_info=True,
        metrics_fn=fast_weight_metrics,
    )
    verify_loading(model, loading_info)
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training/configs/qwen3_0_6b.yaml")
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

    # Pin both snapshots once so all loaders and the model use the same revisions.
    api = HfApi()
    config.model.revision = api.model_info(config.model.model_id, revision=config.model.revision).sha
    config.data.revision = api.dataset_info(config.data.dataset_id, revision=config.data.revision).sha
    model = load_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.lr,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps, weight_decay=config.optimizer.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    common = {
        "dataset_id": config.data.dataset_id, "revision": config.data.revision,
        "batch_size": config.data.batch_size, "seq_len": config.data.seq_len,
        "num_workers": config.data.num_workers, "seed": config.training.seed,
        "shuffle_buffer_size": config.data.shuffle_buffer_size,
    }
    train_loader = create_dataloader(**common, **asdict(config.data.train), shuffle=True)
    val_loader = create_dataloader(**common, **asdict(config.data.val), shuffle=False)

    stop = Event()

    def request_stop(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt("Second interrupt: aborting without saving weights")
        print("Stop requested. Finishing the current microbatch; final validation is best-effort.", flush=True)
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

        train(model, train_loader, val_loader, optimizer, scheduler, config, device, log, stop.is_set)


if __name__ == "__main__":
    main()
