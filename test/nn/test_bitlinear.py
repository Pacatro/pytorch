# Owner(s): ["module: nn"]
import os
# import unittest

import torch
from torch.nn import functional as F
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM
from torch.nn.modules.linear import BitLinear

AMPERE_OR_ROCM = TEST_WITH_ROCM or torch.cuda.is_tf32_supported()


if TEST_WITH_ROCM:
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC_BATCHNORM"] = "1"


class TestBitLinearNN(NNTestCase):
    def test_binarize_weights(self):
        bit_linear = BitLinear(10, 10)

        w_binary, _ = bit_linear.binarize_weights(bit_linear.weight)

        unique_values = torch.unique(w_binary)

        assert len(unique_values) == 2, (
            f"Must be only 2 unique values, got {len(unique_values)}"
        )

        assert torch.allclose(unique_values, torch.tensor([-1.0, 1.0])), (
            "The weights must be -1 or 1"
        )

    def test_activation_quantization(self):
        bit_linear = BitLinear(128, 256, bits=8)

        batch_size, seq_len, features = 4, 32, 128
        x = torch.randn(batch_size, seq_len, features)

        norm_x = F.layer_norm(x, x.shape, bias=None)

        x_quant, _ = bit_linear.quant(norm_x)

        # Verify that the quantized values are in the range [-Q_b, Q_b]
        assert x_quant.min() >= -bit_linear.Q_b, (
            f"Max value {x_quant.min()} < -Q_b={-bit_linear.Q_b}"
        )
        assert x_quant.max() <= bit_linear.Q_b, (
            f"Max value {x_quant.max()} > Q_b={bit_linear.Q_b}"
        )


if __name__ == "__main__":
    run_tests()
