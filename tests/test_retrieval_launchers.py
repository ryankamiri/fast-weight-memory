"""Matched LR recipes and local launch-contract checks; no Slurm submission."""

import copy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

import yaml

from scripts.retrieval_acquisition import prepare_evaluation
from evaluation.storage import file_sha256
from training.checkpoints import ModelCheckpoints
from training.config import load_config
from training.engine import Progress
from training.resume import TrainingState
from training.retrieval_validation import RetrievalValidation
from tests.test_acquisition_training import fixture_config, tiny_training


class RetrievalLaunchTests(unittest.TestCase):
    def test_only_learning_rate_and_name_differ(self):
        base = load_config('training/configs/taal/retrieval_acquisition.yaml')
        for arm, lr in [('lr3e5',3e-5), ('lr3e4',3e-4)]:
            actual = load_config(f'training/configs/taal/retrieval_acquisition_{arm}.yaml')
            expected = copy.deepcopy(base)
            expected.optimizer.lr = lr
            expected.scheduler.final_lr = lr
            expected.wandb.name = actual.wandb.name
            self.assertEqual(actual, expected)
        evaluation = yaml.safe_load(Path('evaluation/configs/shared/retrieval_acquisition_val.yaml').read_text())
        self.assertEqual(evaluation['dataset']['split'], 'val')
        self.assertTrue(evaluation['save_traces'])
        self.assertEqual(evaluation['model']['dtype'], 'float32')
        self.assertEqual(evaluation['model']['autocast_dtype'], 'bfloat16')

    def test_full_context_reference_and_coverage_guard(self):
        base = load_config('training/configs/taal/retrieval_acquisition.yaml')
        actual = load_config('training/configs/taal/retrieval_acquisition_fullcontext.yaml')
        base.model.working_memory_size = 1024
        base.validation.context_access = 'full_context'
        base.wandb.name = actual.wandb.name
        self.assertEqual(actual, base)
        with TemporaryDirectory() as directory:
            config, tokenizer = fixture_config(Path(directory))
            config.model.working_memory_size = 2048
            with self.assertRaisesRegex(ValueError, 'recent-KV'):
                RetrievalValidation(config, tokenizer, Path(directory) / 'probe')
            config.validation.context_access = 'full_context'
            RetrievalValidation(config, tokenizer, Path(directory) / 'probe')
            config.model.working_memory_size = 128
            with self.assertRaisesRegex(ValueError, 'whole episode'):
                RetrievalValidation(config, tokenizer, Path(directory) / 'probe')

    def test_shell_syntax_resources_dependencies_and_no_smoke(self):
        paths = [Path('scripts/queue_retrieval_acquisition.sh'), Path('scripts/explorer_retrieval_env.sh'),
                 Path('training/sbatch/taal/fs_qwen_retrieval_acquisition_short.sbatch'),
                 Path('evaluation/sbatch/taal/fs_qwen_retrieval_acquisition_short.sbatch')]
        for path in paths:
            subprocess.run(['bash','-n',str(path)],check=True)
        for path in paths[2:]:
            source = path.read_text()
            memory = '--mem=24G' if path == paths[2] else '--mem=48G'
            for setting in ['--partition=gpu-short',memory,'--time=02:00:00','--gres=gpu:1']:
                self.assertIn(setting,source)
            self.assertNotIn('preflight',source)
        queue = paths[0].read_text()
        self.assertIn('--dependency="afterok:$train_job"',queue)
        self.assertIn('for arm in lr3e5 lr1e4 lr3e4 fullcontext',queue)

    def test_saved_tokenizer_and_matched_full_val_preparation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config, tokenizer = fixture_config(root)
            train_path = root/'recipe.yaml'
            contents = config.to_dict()
            contents['loss'].pop('objective')
            train_path.write_text(yaml.safe_dump(contents))
            model,loader,optimizer,scheduler = tiny_training(config)
            # Use the real data identity while keeping the checkpoint fixture CPU-only.
            from_data = json.loads(Path(config.data.manifest_path).read_text())
            self.assertEqual(from_data['files']['val']['rows'],16)
            loader.dataset.identity = {'manifest_sha256':file_sha256(Path(config.data.manifest_path))}
            manager = ModelCheckpoints(config,'fixture',TrainingState(optimizer,scheduler,loader),tokenizer)
            manager.save_final(model,Progress(step=399),'signal')
            source=manager.directory/'final'
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                prepare_evaluation(train_path,source,root/'audit')
            manager.save_final(model,Progress(step=400),'completed')
            prepare_evaluation(train_path,source,root/'audit')
            prepared=yaml.safe_load((root/'audit/config.yaml').read_text())
            self.assertEqual(prepared['model']['checkpoint'],str(source))
            self.assertEqual(prepared['dataset']['path'],config.data.split_files['val'])
            prepare_evaluation(train_path,source,root/'reads-off',0.0)
            disabled=yaml.safe_load((root/'reads-off/config.yaml').read_text())
            self.assertEqual(disabled['model']['settings']['memory_read_scale'],0.0)
            self.assertTrue(disabled['save_traces'])
            self.assertEqual(prepared['model']['settings']['memory_read_scale'],1.0)
            self.assertTrue((source/'tokenizer.json').is_file())
            self.assertEqual(json.loads((root/'audit/training_receipt.json').read_text())['validation_questions'],16)


if __name__ == '__main__':
    unittest.main()
