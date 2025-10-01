# Owner(s): ["module: nn"]
import os
# import unittest

import torch
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM
from torch.nn.modules.linear import BitLinear

AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()


if TEST_WITH_ROCM:
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM"] = "1"


class TestBitLinearNN(NNTestCase):
    """Test suite for BitLinear implementation"""

    def setUp(self):
        """Set up test fixtures"""
        torch.manual_seed(42)
        self.in_features = 128
        self.out_features = 64
        self.batch_size = 16
        self.layer = BitLinear(self.in_features, self.out_features, bits=8)

    def test_initialization(self):
        """Test that layer initializes correctly"""
        self.assertEqual(self.layer.in_features, self.in_features)
        self.assertEqual(self.layer.out_features, self.out_features)
        self.assertEqual(self.layer.bits, 8)
        self.assertEqual(self.layer.Q_b, 128)  # 2^(8-1)
        self.assertIsNotNone(self.layer.weight)
        self.assertEqual(self.layer.weight.shape, (self.out_features, self.in_features))

    def test_weight_binarization(self):
        """Test that the weights are binarized correctly"""

        w_binary, beta = self.layer.binarize_weights(self.layer.weight)

        unique_values = torch.unique(w_binary)

        self.assertEqual(w_binary.shape, self.layer.weight.shape)
        self.assertEqual(len(unique_values), 2, msg="Must be only 2 unique values")
        self.assertTrue(
            torch.allclose(unique_values, torch.tensor([-1.0, 1.0])),
            msg="The weights must be -1 or 1",
        )
        self.assertGreater(beta.item(), 0, msg="Beta must be > 0")

    def test_activation_quantization(self):
        """Test that the activations are quantized correctly"""

        x = torch.randn(self.batch_size, self.in_features)
        x_quant, gamma = self.layer.activation_quantization(x)

        # Check quantized values are within [-Q_b, Q_b]
        self.assertTrue(torch.all(x_quant >= -self.layer.Q_b))
        self.assertTrue(torch.all(x_quant <= self.layer.Q_b))
        self.assertAlmostEqual(gamma.item(), x.abs().max().item(), places=5)

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape"""

        x = torch.randn(self.batch_size, self.in_features)
        output = self.layer(x)

        self.assertEqual(output.shape, (self.batch_size, self.out_features))

    def test_backward_pass(self):
        """Test that gradients flow correctly through the layer"""

        x = torch.randn(self.batch_size, self.in_features, requires_grad=True)
        output = self.layer(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are not NaN
        self.assertIsNotNone(self.layer.weight.grad)
        self.assertFalse(torch.any(torch.isnan(self.layer.weight.grad)))
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.any(torch.isnan(x.grad)))

    def test_different_bit_widths(self):
        """Test layer works with different bit widths"""

        for bits in [4, 8, 16]:
            layer = BitLinear(self.in_features, self.out_features, bits=bits)
            x = torch.randn(self.batch_size, self.in_features)
            output = layer(x)

            self.assertEqual(output.shape, (self.batch_size, self.out_features))
            self.assertEqual(layer.Q_b, 2 ** (bits - 1))


if __name__ == "__main__":
    run_tests()
