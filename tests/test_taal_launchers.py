"""Exercise batch argument handling without Conda, Slurm, or GPU execution."""

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
CPT = ROOT / "evaluation/sbatch/taal/fs_qwen_cpt_audit_short.sbatch"
TRAIN = ROOT / "training/sbatch/taal/fs_qwen_workshop_short.sbatch"
AUDIT = ROOT / "evaluation/sbatch/taal/fs_qwen_workshop_short.sbatch"


class LauncherTests(unittest.TestCase):
    def run_launcher(self, path, *arguments, commit="new-runtime-commit"):
        # Run the real script, mocking only cluster dependencies and Python work.
        # The checkout revision can change while a job waits in the queue.
        harness = r'''
source() { :; }
conda() { :; }
cd() { export PWD="$1"; }
git() {
    if [[ "$1" == rev-parse ]]; then echo "$TEST_COMMIT";
    elif [[ "$1" == branch ]]; then echo titans-impl;
    else return 1; fi
}
mkdir() { :; }
cp() { :; }
python() { printf 'PYTHON'; printf ' %s' "$@"; printf '\n'; }
exec() { "$@"; }
export SLURM_SUBMIT_DIR=/projects/llpr/amiri.ry/dev/fast-weight-memory
export SLURM_JOB_ID=123
TEST_COMMIT="$1"
shift
script="$1"
shift
. "$script"
'''
        return subprocess.run(
            ["bash", "-c", harness, "test-launcher", commit, str(path), *arguments],
            text=True, capture_output=True,
        )

    def test_all_launch_paths_log_actual_revision_and_forward_arguments(self):
        cases = [
            (CPT, ("A", "11538464"), "--checkpoint checkpoints/taal-cpt-A-11538464/final"),
            (TRAIN, ("narrow",), "--config training/configs/taal/workshop_narrow.yaml"),
            (TRAIN, ("diverse",), "--arm diverse"),
            (AUDIT, ("baseline",), "baseline --output-dir output/taal/workshop-baseline-123"),
            (AUDIT, ("matched", "narrow-100", "diverse-101"), "--narrow-run narrow-100 --diverse-run diverse-101"),
        ]
        for path, arguments, forwarded in cases:
            for commit in ("first-revision", "later-compatible-revision"):
                with self.subTest(path=path, arguments=arguments, commit=commit):
                    result = self.run_launcher(path, *arguments, commit=commit)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"commit={commit}", result.stdout)
                    self.assertIn(forwarded, result.stdout)

    def test_bad_arguments_still_fail_before_model_execution(self):
        for path, arguments in [
            (CPT, ()), (CPT, ("E", "1")), (CPT, ("A", "not-a-job")),
            (TRAIN, ()), (TRAIN, ("typo",)), (AUDIT, ()),
            (AUDIT, ("matched", "missing-diverse")), (AUDIT, ("typo",)),
        ]:
            with self.subTest(path=path, arguments=arguments):
                result = self.run_launcher(path, *arguments)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("PYTHON", result.stdout)

    def test_launchers_have_valid_shell_syntax(self):
        for path in (CPT, TRAIN, AUDIT):
            subprocess.run(["bash", "-n", str(path)], check=True)
