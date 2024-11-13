import itertools
import unittest

import torch
import torch.nn.functional
import xformers.ops as xops

from opensora.utils.custom.xformers import block_diagonal_mask, memory_efficient_attention


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


# Test case for xformers equivalent functions
class XformersTest(unittest.TestCase):
    def test_memory_efficient_attention(self):
        TENSOR_SHAPES = [
            (8, 8, 64, 32),
            (16, 16, 128, 64),
            (32, 32, 128, 128),
            (64, 64, 64, 64),
        ]

        for shape, use_attn_bias in itertools.product(TENSOR_SHAPES, [True, False]):
            # Fixed dropout to 0.0 -> reproducibility
            p = 0.0

            # Generate QKV
            query, key, value, attn_bias = generate_tensors(shape, device="cuda")
            if not use_attn_bias:
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

    def test_block_diagonal_mask(self):
        DTYPE = torch.float32
        DEVICE = torch.device("cpu")

        Q_SEQLEN = [
            [2] * 3,
            [3] * 4,
            [4] * 5,
        ]
        KV_SEQLEN = [[4] * 3, [8] * 4, [12] * 5]

        for q_seqlen, kv_seqlen in zip(Q_SEQLEN, KV_SEQLEN):
            # True output
            xformers_output = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen).materialize(
                (sum(q_seqlen), sum(kv_seqlen)), DTYPE, DEVICE
            )

            # Alternative output
            alternative_output = block_diagonal_mask(q_seqlen, kv_seqlen, DTYPE, DEVICE)

            is_equal = torch.equal(xformers_output, alternative_output)
            self.assertTrue(is_equal, "Outputs are not equal!")
