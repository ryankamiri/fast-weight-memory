"""Document loss and free-generation traces for the bounded CPT pilot."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from evaluation.taal.state_audit import load_model
from evaluation.taal.trace_contract import TraceEpisode
from evaluation.taal.trace_export import TaalTraceExporter, TaalTraceRecorder, make_trace_tokens
from training.engine import precision_context


@torch.inference_mode()
def export_documents(model, tokenizer, rows, checkpoint, directory, scales=(1.0, 0.0), max_new_tokens=32):
    directory.mkdir(parents=True, exist_ok=True)
    layers = list(range(len(model.model.layers)))
    exporter = TaalTraceExporter(directory / 'memory_traces', run_metadata={
        'checkpoint': str(checkpoint), 'tokenizer': tokenizer.name_or_path,
        'layers': layers, 'capture': 'Whole-document reads on/off, next-token loss and greedy continuation',
        'memory_max_write_grad_norm': model.config.memory_max_write_grad_norm,
        'parameter_dtype': str(next(model.parameters()).dtype),
        'write_signal': 'Fresh proposed gradient after configured norm clipping',
    })
    results = []
    model.eval()
    for row in rows:
        ids = row['input_ids']
        for scale in scales:
            condition = 'correct_full' if scale == 1 else 'reads_disabled'
            recorder = TaalTraceRecorder(model, layers, len(ids))
            input_ids = torch.tensor([ids], device=model.device)
            with precision_context(model.device), recorder:
                output = model(input_ids, use_cache=True, memory_read_scale=scale)
            loss_tensor = F.cross_entropy(output.logits[0, :-1].float(), input_ids[0, 1:])
            if not torch.isfinite(output.logits).all() or not torch.isfinite(loss_tensor):
                raise ValueError(f'Nonfinite document {row["source_record_index"]}/{condition}')
            loss = loss_tensor.item()
            state = output.state
            tokens = make_trace_tokens(ids, tokenizer)
            writes, reads, prefixes = recorder.writes, recorder.reads, recorder.internal_prefixes
            generated = []
            for step in range(max_new_tokens):
                token = int(output.logits[0, -1].argmax())
                generated.append(token)
                position = len(ids) + step
                current = TaalTraceRecorder(model, layers, 1, position_offset=position)
                with precision_context(model.device), current:
                    output = model(torch.tensor([[token]], device=model.device), state=state, use_cache=True,
                                   memory_read_scale=scale, prepend_memory_tokens=False)
                if not torch.isfinite(output.logits).all():
                    raise ValueError(f'Nonfinite generation token {position}')
                state = output.state
                tokens += make_trace_tokens([token], tokenizer, offset=position, phase='generated')
                writes += current.writes
                reads += current.reads
                prefixes += current.internal_prefixes
                if token == tokenizer.eos_token_id:
                    break
            outcome = {'example_id': str(row['source_record_index']), 'condition': condition,
                       'source': row['source'], 'loss': loss, 'perplexity': math.exp(loss),
                       'generated_ids': generated, 'generated_text': tokenizer.decode(generated),
                       'unique_generated_fraction': len(set(generated)) / len(generated)}
            exporter.export(TraceEpisode(directory.name, outcome['example_id'], condition, str(checkpoint),
                tokenizer.name_or_path, tokens, writes, reads, prefixes, outcome,
                {'phase': 'heldout_document', 'read_policy': 'whole_episode', 'document_sha256': row['document_sha256']}))
            results.append(outcome)
            (directory / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.checkpoint / 'training_metadata.json').read_text())
    settings = metadata['training_config']
    tokenizer = AutoTokenizer.from_pretrained(settings['model']['model_id'], revision=settings['model']['revision'])
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    rows = [row for row in rows if row['source_record_index'] >= 80]
    model = load_model(args.checkpoint, dtype=torch.float32).to('cuda').eval()
    export_documents(model, tokenizer, rows, args.checkpoint, args.output_dir)


if __name__ == '__main__':
    main()
