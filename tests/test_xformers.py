import itertools
import unittest

import parameterized
import torch
import torch.nn.functional
import xformers.ops as xops

from opensora.utils.xformers import memory_efficient_attention

TENSOR_SHAPES = [
    (8, 8, 64, 32),
    (16, 16, 128, 64),
    (32, 32, 128, 128),
    (64, 64, 64, 64),
]


# Helper functions
def generate_tensors(shape, device="cpu"):
    # Generate QKV
    query = torch.rand(shape).to(device)
    key = torch.rand(shape).to(device)
    value = torch.rand(shape).to(device)

    # Generate attn bias
    batch_size, num_heads, seq_len, _ = shape
    attn_bias = torch.rand((batch_size, seq_len, num_heads, num_heads)).to(device)

    return query, key, value, attn_bias


@parameterized.parameterized_class(("shape", "use_attn_bias"), itertools.product(TENSOR_SHAPES, [True, False]))
class XformersTest(unittest.TestCase):
    def test_memory_efficient_attention(self):
        # Fixed dropout to 0.0 -> reproducibility
        p = 0.0

        # Generate QKV
        query, key, value, attn_bias = generate_tensors(self.shape, device="cuda")
        if not self.use_attn_bias:
            attn_bias = None

        # Outputs
        xformers_output = xops.memory_efficient_attention(query, key, value, p=p, attn_bias=attn_bias)
        alternative_output = memory_efficient_attention(query, key, value, p=p, attn_bias=attn_bias)

        # Check shape
        self.assertEqual(
            xformers_output.shape,
            alternative_output.shape,
            "Output shape not matched! {} != {}".format(xformers_output.shape, alternative_output.shape),
        )

        # Check equal
        tolerance = 1e-6
        absolute_difference = torch.abs(xformers_output - alternative_output)
        max_absolute_difference = torch.max(absolute_difference).item()
        self.assertTrue(
            max_absolute_difference <= tolerance,
            f"Maximum absolute difference ({max_absolute_difference}) exceeds the tolerance level ({tolerance})",
        )
