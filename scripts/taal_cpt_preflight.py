"""Validate the pilot corpus, checkpoint provenance and longest training graph."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from data.dataloader import CausalLMCollator
from training.config import load_config
from training.engine import precision_context
from training.optimization import build_optimizer, clip_training_gradients
from training.train import configure_trainable_parameters, initialize_taal_gates, load_model


def validate_data(arm):
    root = Path('data/taal_cpt')
    manifest = json.loads((root / 'manifest.json').read_text())
    for name in (f'{arm}.jsonl', 'retrieval.jsonl'):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if digest != manifest['files'][name]:
            raise ValueError(f'Data checksum mismatch: {name}')
    settings = load_config(f'training/configs/taal/cpt_{arm}.yaml')
    rows = [json.loads(line) for line in Path(settings.data.dataset_id).read_text().splitlines()]
    if len(rows) != 89 or [row['source_record_index'] for row in rows] != list(range(89)):
        raise ValueError('Expected 80 training and 9 held-out records in order')
    if {r['document_sha256'] for r in rows[:80]} & {r['document_sha256'] for r in rows[80:]}:
        raise ValueError('Train/validation document overlap')
    for index, row in enumerate(rows[:80]):
        batch = CausalLMCollator()([row])
        expected = 1 if arm == 'D' and index % 4 == 3 else len(row['input_ids']) - 1
        if int((batch['labels'][:, 1:] != -100).sum()) != expected:
            raise ValueError(f'Incorrect loss mask for {arm}/{index}')
    if sum(len(row['input_ids']) for row in rows[:80]) != 40960:
        raise ValueError('Incorrect token budget')
    print('CPT_DATA_VALID', arm, json.dumps(manifest['arms'][arm]), flush=True)
    return settings, rows


def validate_checkpoint(checkpoint, settings):
    for name in ('model.safetensors', 'config.json', 'training_metadata.json'):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    metadata = json.loads((checkpoint / 'training_metadata.json').read_text())
    config = json.loads((checkpoint / 'config.json').read_text())
    if metadata['reason'] not in ('completed', 'signal') or metadata['progress']['step'] < 1:
        raise ValueError('No usable training checkpoint; never audit an exception snapshot')
    saved = metadata['training_config']
    expected = settings.to_dict()
    for section in ('model', 'data', 'loss', 'optimizer', 'scheduler', 'training', 'validation', 'checkpoints'):
        if saved[section] != expected[section]:
            raise ValueError(f'Checkpoint recipe differs: {section}')
    if config.get('memory_max_write_grad_norm') != 1.0:
        raise ValueError('Saved checkpoint does not preserve fresh-write clipping')
    print('CPT_CHECKPOINT_VALID', json.dumps(metadata['progress']), 'reason=' + metadata['reason'], flush=True)
    return metadata


def gpu_training_check(settings, rows):
    if not torch.cuda.is_available() or 'H200' not in torch.cuda.get_device_name():
        raise RuntimeError('This pilot requires an allocated H200')
    torch.manual_seed(settings.training.seed)
    device = torch.device('cuda')
    model = load_model(settings).to(device).train()
    initialize_taal_gates(model, settings.training.initial_taal_gate)
    configure_trainable_parameters(model, settings.training.trainable_parameters)
    optimizer = build_optimizer(model, settings)
    row = max(rows[:80], key=lambda row: len(row['input_ids']))
    batch = {name: tensor.to(device) for name, tensor in CausalLMCollator()([row]).items()}
    start = time.monotonic()
    with precision_context(device):
        output = model(**batch)
    if not torch.isfinite(output.loss):
        raise ValueError('Longest-row preflight loss is nonfinite')
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError('Longest-row preflight outer gradients are nonfinite')
    clip_training_gradients(model, settings)
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Preflight optimizer update produced a nonfinite parameter')
    torch.cuda.synchronize()
    print('CPT_GPU_TRAIN_VALID', json.dumps({'tokens': len(row['input_ids']), 'loss': output.loss.item(),
          'seconds': time.monotonic() - start, 'peak_gib': torch.cuda.max_memory_allocated() / 2**30}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=list('ABCD'))
    parser.add_argument('--gpu-train', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    settings, rows = validate_data(args.arm)
    if args.checkpoint:
        validate_checkpoint(args.checkpoint, settings)
    if args.gpu_train:
        gpu_training_check(settings, rows)


if __name__ == '__main__':
    main()
