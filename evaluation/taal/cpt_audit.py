"""Run the standard audit with the CPT training precision policy."""

import torch

from evaluation.taal import state_audit


if __name__ == '__main__':
    # Keep FP32 checkpoint parameters and use BF16 CUDA operations, as in training.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        state_audit.main()
