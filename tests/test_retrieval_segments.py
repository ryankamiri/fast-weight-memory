"""Allocation lifecycle checks, without launching a model or submitting Slurm jobs."""

import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.retrieval_training_segment import finish_segment, run_segment


def save_checkpoint(path, step, reason="signal"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.pt").write_bytes(b"fixture")
    (path / "training_metadata.json").write_text(json.dumps({"progress": {"step": step}, "reason": reason}))


class SegmentTests(unittest.TestCase):
    def test_completion_and_failure_boundaries(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            save_checkpoint(path, 177)
            self.assertEqual(finish_segment(path, 0, 400, True, 0), "requeue")
            for previous, timed_out, restart in [(177, True, 0), (0, False, 0), (0, True, 5)]:
                with self.assertRaises(ValueError):
                    finish_segment(path, previous, 400, timed_out, restart)
            save_checkpoint(path, 400, "completed")
            self.assertEqual(finish_segment(path, 177, 400, False, 2), "complete")
            (path / "trainer_state.pt").unlink()
            with self.assertRaises(ValueError):
                finish_segment(path, 177, 400, False, 2)

    def test_resume_signal_forwarding_and_requeue(self):
        for mode in ("timeout", "complete", "cancel", "crash", "requeue_rejected"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                path = Path(directory) / "stable-root" / "final"
                save_checkpoint(path, 177)
                config = SimpleNamespace(checkpoints=SimpleNamespace(output_dir=directory), training=SimpleNamespace(max_steps=400))
                sent = []

                def wait():
                    if mode == "crash":
                        return 1
                    if mode != "complete":
                        sig = signal.SIGTERM if mode == "cancel" else signal.SIGUSR1
                        signal.raise_signal(sig)
                    save_checkpoint(path, 400 if mode == "complete" else 300, "completed" if mode == "complete" else "signal")
                    return 0

                child = SimpleNamespace(wait=wait, poll=lambda: None, send_signal=sent.append)
                with patch('scripts.retrieval_training_segment.load_config', return_value=config), patch('scripts.retrieval_training_segment.subprocess.Popen', return_value=child) as launch, patch('scripts.retrieval_training_segment.subprocess.run') as requeue:
                    if mode == "requeue_rejected":
                        requeue.side_effect = subprocess.CalledProcessError(1, ['scontrol'])
                    if mode == "complete":
                        run_segment(Path('recipe.yaml'), 'stable-root', '123', 1)
                    elif mode == "timeout":
                        with self.assertRaises(SystemExit) as stopped:
                            run_segment(Path('recipe.yaml'), 'stable-root', '123', 1)
                        self.assertEqual(stopped.exception.code, 75)
                    else:
                        with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                            run_segment(Path('recipe.yaml'), 'stable-root', '123', 1)
                    command = launch.call_args.args[0]
                    self.assertEqual(command[-2:], ['--resume', str(path)])
                    self.assertEqual(launch.call_args.kwargs['env']['CHECKPOINT_RUN_ID'], 'stable-root')
                    if mode in ('timeout', 'requeue_rejected'):
                        requeue.assert_called_once_with(['scontrol', 'requeue', '123'], check=True, timeout=60)
                    else:
                        requeue.assert_not_called()
                    if mode not in ('complete', 'crash'):
                        self.assertEqual(sent, [signal.SIGTERM])

    def test_queue_connects_all_arms_to_stable_roots(self):
        script = Path('scripts/queue_retrieval_acquisition.sh').resolve()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'git').write_text('#!/bin/bash\necho titans-impl\n')
            (root / 'sbatch').write_text('#!/bin/bash\necho "$*" >> "$RECEIPTS"\nwc -l < "$RECEIPTS" | tr -d " "\n')
            for name in ('git', 'sbatch'):
                (root / name).chmod(0o755)
            receipts = root / 'receipts'
            environment = dict(os.environ, PATH=str(root) + ':' + os.environ['PATH'], RECEIPTS=str(receipts))
            subprocess.run(['bash', '-c', 'PWD=/projects/llpr/amiri.ry/dev/fast-weight-memory; source "$1" 11868529', 'test', str(script)], cwd=root, env=environment, check=True, capture_output=True)
            lines = receipts.read_text().splitlines()
            self.assertEqual(len(lines), 8)
            for index, arm in enumerate(('lr3e5', 'lr1e4', 'lr3e4', 'fullcontext')):
                train_job = 2 * index + 1
                checkpoint_root = 11868529 if index == 0 else train_job
                self.assertIn(f'--dependency=afterok:{train_job}', lines[2 * index + 1])
                self.assertTrue(lines[2 * index + 1].endswith(f'{arm} {checkpoint_root}'))
            self.assertTrue(lines[0].endswith('lr3e5 11868529'))

    def test_restart_never_silently_starts_fresh(self):
        with TemporaryDirectory() as directory:
            config = SimpleNamespace(checkpoints=SimpleNamespace(output_dir=directory), training=SimpleNamespace(max_steps=400))
            with patch('scripts.retrieval_training_segment.load_config', return_value=config), patch('scripts.retrieval_training_segment.subprocess.Popen') as launch:
                with self.assertRaisesRegex(ValueError, 'no final checkpoint'):
                    run_segment(Path('recipe.yaml'), 'missing', '123', 1)
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
