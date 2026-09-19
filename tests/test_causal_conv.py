import copy
import unittest

import torch

from architectures.shared.causal_conv import CausalDepthwiseConv1d


class CausalDepthwiseConvTests(unittest.TestCase):
    def test_split_calls_match_concatenated_call(self):
        torch.manual_seed(4)
        full_conv = CausalDepthwiseConv1d(5, 4)
        split_conv = copy.deepcopy(full_conv)
        inputs = torch.randn(2, 9, 5)

        full, full_history = full_conv(inputs)
        history = None
        pieces = []
        for start, end in ((0, 2), (2, 3), (3, 7), (7, 9)):
            output, history = split_conv(inputs[:, start:end], history)
            pieces.append(output)

        torch.testing.assert_close(torch.cat(pieces, dim=1), full)
        torch.testing.assert_close(history, full_history)

    def test_kernel_size_one_has_empty_history(self):
        conv = CausalDepthwiseConv1d(3, 1)
        inputs = torch.randn(2, 4, 3)
        output, history = conv(inputs)
        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(history.shape, (2, 3, 0))


if __name__ == "__main__":
    unittest.main()
