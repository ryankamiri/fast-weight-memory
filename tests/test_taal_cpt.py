import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from architectures.taal.qwen.causal_lm import TaalQwen3ForCausalLM
from architectures.titans.configuration import NeuralMemoryConfig
from architectures.titans.neural_memory import ChunkGradientResult, NeuralMemory, clip_write_gradient
from data.dataloader import CausalLMCollator, create_dataloader
from evaluation.taal.cpt_trace import export_documents
from training.config import load_config
from training.checkpoints import ModelCheckpoints
from training.engine import train
from training.train import build_scheduler
from scripts.taal_cpt_preflight import validate_checkpoint
from training.train import configure_trainable_parameters, initialize_taal_gates
from tests.test_neural_memory import always_vmap_gradient
from tests import test_taal_state_audit


class Tokenizer:
    name_or_path = 'fixture'
    eos_token_id = None

    def decode(self, tokens, **kwargs):
        return ' '.join(str(token) for token in tokens)


class CptTests(unittest.TestCase):
    def test_training_loop_dense_and_masked_steps_with_checkpoint_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data.jsonl'
            rows = [
                {'source_record_index': 0, 'input_ids': [2, 3, 4, 5], 'labels': [2, 3, 4, 5]},
                {'source_record_index': 1, 'input_ids': [4, 5, 6, 7, 8], 'labels': [-100] * 4 + [8]},
                {'source_record_index': 2, 'input_ids': [6, 7, 8, 9], 'labels': [6, 7, 8, 9]},
            ]
            dataset.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            settings = load_config('training/configs/taal/cpt_D.yaml')
            settings.data.dataset_id = str(dataset)
            settings.data.train.end = 2
            settings.data.val.start, settings.data.val.end = 2, 3
            settings.training.max_steps = 2
            settings.training.eval_every_steps = 2
            settings.checkpoints.save_steps = [2]
            settings.checkpoints.output_dir = str(root)
            config = test_taal_state_audit.TaalStateAuditTests.config()
            config.memory_max_write_grad_norm = 1.0
            model = TaalQwen3ForCausalLM(config).train()
            initialize_taal_gates(model, 0.05)
            parameters = configure_trainable_parameters(model, 'taal_only')
            optimizer = torch.optim.AdamW(parameters, lr=settings.optimizer.lr)
            scheduler = build_scheduler(optimizer, settings)
            loader = create_dataloader(settings.data, settings.data.train, settings.loss, True, 42)
            validation = create_dataloader(settings.data, settings.data.val, settings.loss, False, 42)
            metrics = []
            saves = ModelCheckpoints(settings, 'fixture')
            progress = train(model, loader, validation, optimizer, scheduler, settings, torch.device('cpu'),
                             metrics.append, lambda: False, on_validation=saves.on_validation)
            self.assertEqual((progress.step, progress.tokens_seen, progress.skipped_updates), (2, 9, 0))
            self.assertEqual(metrics[-1]['val/records'], 1)
            saves.save_final(model, progress, 'completed')
            validate_checkpoint(saves.directory / 'final', settings)
            metadata_path = saves.directory / 'final/training_metadata.json'
            metadata = json.loads(metadata_path.read_text())
            metadata['reason'] = 'exception'
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                validate_checkpoint(saves.directory / 'final', settings)

    def test_active_clipping_matches_autograd_and_outer_gradient(self):
        torch.manual_seed(42)
        for B in (1, 2):
            memory = NeuralMemory(NeuralMemoryConfig(dim=128, depth=2, chunk_size=1)).train()
            weights = memory.initial_state(B).weights
            keys = torch.randn(B, 1, 128, requires_grad=True)
            values = (30 * torch.randn(B, 1, 128)).requires_grad_()
            strength = torch.ones(B, 1, requires_grad=True)
            actual = memory._chunk_gradient(weights, keys, values, strength)
            expected = always_vmap_gradient(memory, weights, keys, values, strength)
            norm = torch.cat([g.flatten(1) for g in expected.values()], dim=1).double().norm(dim=1)
            self.assertTrue((norm > 1).all())
            expected = {name: g * (1 / norm).float().reshape(B, 1, 1) for name, g in expected.items()}
            clipped = clip_write_gradient(actual, 1.0).chunk_gradients
            for name in expected:
                torch.testing.assert_close(clipped[name], expected[name], atol=2e-6, rtol=2e-5)
            direction = {name: torch.randn_like(g) for name, g in expected.items()}
            actual_loss = sum((g * direction[name]).sum() for name, g in clipped.items())
            expected_loss = sum((g * direction[name]).sum() for name, g in expected.items())
            actual_outer = torch.autograd.grad(actual_loss, (keys, values, strength), retain_graph=True)
            expected_outer = torch.autograd.grad(expected_loss, (keys, values, strength))
            for actual_g, expected_g in zip(actual_outer, expected_outer):
                self.assertTrue(torch.isfinite(actual_g).all())
                torch.testing.assert_close(actual_g, expected_g, atol=2e-6, rtol=2e-4)

    def test_zero_and_subthreshold_writes_have_finite_identity_backward(self):
        for magnitude in (0.0, 0.01):
            g = torch.full((2, 3, 3), magnitude, requires_grad=True)
            result = clip_write_gradient(ChunkGradientResult({'W': g}, {'W': g.unsqueeze(1)}), 1.0)
            torch.testing.assert_close(result.chunk_gradients['W'], g)
            torch.testing.assert_close(result.token_gradients['W'][:, 0], g)
            result.chunk_gradients['W'].sum().backward()
            torch.testing.assert_close(g.grad, torch.ones_like(g))

    def test_clipping_is_used_during_training_and_survives_reload_and_traces(self):
        torch.manual_seed(42)
        config = test_taal_state_audit.TaalStateAuditTests.config()
        config.memory_max_write_grad_norm = 1e-12
        model = TaalQwen3ForCausalLM(config).train()
        initialize_taal_gates(model, 0.05)
        configure_trainable_parameters(model, 'taal_only')
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-5)
        ids = torch.tensor([[2, 7, 3, 8, 5, 9, 4, 6]])
        norms = []

        def observe(result, maximum):
            before = torch.cat([g.flatten(1) for g in result.chunk_gradients.values()], dim=1).norm(dim=1)
            result = clip_write_gradient(result, maximum)
            after = torch.cat([g.flatten(1) for g in result.chunk_gradients.values()], dim=1).norm(dim=1)
            norms.append((before.detach(), after.detach()))
            return result

        with patch('architectures.titans.neural_memory.clip_write_gradient', side_effect=observe):
            for masked in (False, True):
                labels = ids.clone()
                if masked:
                    labels[:, :-1] = -100
                optimizer.zero_grad()
                loss = model(ids, labels=labels).loss
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().max() > 0 for g in gradients))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        self.assertTrue(any((before > 1e-12).any() for before, after in norms))
        self.assertTrue(all((after <= 1.00001e-12).all() for before, after in norms))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model.eval().save_pretrained(root)
            restored = TaalQwen3ForCausalLM.from_pretrained(root).eval()
            self.assertEqual(restored.config.taal_layer_config().memory.max_write_grad_norm, 1e-12)
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, restored(ids).logits)
            rows = [{'input_ids': ids[0].tolist(), 'source_record_index': 80,
                     'source': 'fixture', 'document_sha256': 'fixture'}]
            results = export_documents(restored, Tokenizer(), rows, root, root / 'traces', max_new_tokens=2)
            self.assertEqual(len(results), 2)
            manifest = json.loads((root / 'traces/memory_traces/manifest.json').read_text())
            self.assertEqual(len(manifest['episodes']), 2)
            self.assertEqual(len(results[0]['generated_ids']), 2)

    def test_real_pilot_data_masks_order_and_exposure(self):
        if not Path('data/taal_cpt/A.jsonl').exists():
            self.skipTest('Run scripts.prepare_taal_cpt for pinned corpus integration test')
        for arm in 'ABCD':
            config = load_config(f'training/configs/taal/cpt_{arm}.yaml')
            loader = create_dataloader(config.data, config.data.train, config.loss, True, 42)
            batches = list(loader)
            self.assertEqual(len(batches), 80)
            self.assertEqual(sum(batch['input_ids'].numel() for batch in batches), 40960)
            rows = [json.loads(line) for line in Path(config.data.dataset_id).read_text().splitlines()]
            for index, batch in enumerate(batches):
                self.assertEqual(batch['input_ids'][0].tolist(), rows[index]['input_ids'])
                expected = 1 if arm == 'D' and index % 4 == 3 else batch['input_ids'].numel() - 1
                self.assertEqual(int((batch['labels'][:, 1:] != -100).sum()), expected)
            validation = list(create_dataloader(config.data, config.data.val, config.loss, False, 42))
            self.assertEqual(len(validation), 9)

    def test_invalid_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            CausalLMCollator()([{'input_ids': [1, 2], 'labels': [1, 3]}])


if __name__ == '__main__':
    unittest.main()
