"""Fused Attention."""

import itertools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_brute_force_configs():
    configs = []
    BLOCK_M = [64, 128, 256]
    BLOCK_N = [32, 64, 128]
    WAVES_PER_EU = [1, 2, 3, 4]
    PRE_LOAD_V = [True, False]
    NUM_STAGES = range(1, 10)
    NUM_WARPS = [
        4,
        8,
    ]
    MATRIX_INSTR_NONKDIM = [16, 32]

    space = itertools.product(BLOCK_M, BLOCK_N, WAVES_PER_EU, PRE_LOAD_V, NUM_STAGES, NUM_WARPS, MATRIX_INSTR_NONKDIM)

    for instance in space:
        (
            block_m,
            block_n,
            waves_per_eu,
            pre_load_v,
            num_stages,
            num_warps,
            matrix_instr_nonkdim,
        ) = instance
        configs.append(
            triton.Config(
                {
                    "BLOCK_M": block_m,
                    "BLOCK_N": block_n,
                    "PRE_LOAD_V": pre_load_v,
                    "waves_per_eu": waves_per_eu,
                    "matrix_instr_nonkdim": matrix_instr_nonkdim,
                },
                num_stages=num_stages,
                num_warps=num_warps,
            )
        )

    print(f"Len {len(configs)}")
    return configs


def get_cdna_autotune_configs():
    return [
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "waves_per_eu": 2, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 2, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 3, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "waves_per_eu": 1, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "waves_per_eu": 2, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
        # Fall-back config.
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "waves_per_eu": 1, "PRE_LOAD_V": False},
            num_stages=1,
            num_warps=4,
        ),
    ]


current_best_configs = [
    triton.Config(
        {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "waves_per_eu": 2,
            "PRE_LOAD_V": True,
            "matrix_instr_nonkdim": 32,
        },
        num_warps=4,
        num_stages=7,
    )
]


@triton.autotune(
    configs=current_best_configs,
    key=["stride_qh", "stride_qm", "stride_kh", "stride_kn", "stride_vh", "stride_vk"],
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    QK_SCALE: tl.constexpr,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # n_extra_tokens = N_CTX % BLOCK_N

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h * stride_qh
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h * stride_kh
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h * stride_vh
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    q = (q * QK_SCALE).to(q.type.element_ty)

    # Padding on Q does not need to be masked in the FA loop.
    # masked_blocks = n_extra_tokens != 0
    # masked_blocks = min(masked_blocks, n_blocks)
    n_blocks = cdiv_fn(N_CTX, BLOCK_N)
    n_full_blocks = n_blocks - 1
    # block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    # block_min = 0
    block_max = n_full_blocks * BLOCK_N
    # loop over k, v, and update accumulator
    k_loop_ptrs, v_loop_ptrs = k_ptrs, v_ptrs
    for _ in range(0, block_max, BLOCK_N):
        k_offs_k = None
        k_offs_n = None
        k = load_fn(k_loop_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, N_CTX)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_loop_ptrs, k_offs_n, k_offs_k, N_CTX, BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # -- compute qk ----
        qk += tl.dot(q, k)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_loop_ptrs, k_offs_n, k_offs_k, N_CTX, BLOCK_DMODEL)
        # -- update m_i and l_i
        l_ij = tl.sum(p, 1)
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v)
        k_loop_ptrs += BLOCK_N * stride_kn
        v_loop_ptrs += BLOCK_N * stride_vk

    block_min = block_max
    # block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    k_ptrs += n_full_blocks * BLOCK_N * stride_kn
    v_ptrs += n_full_blocks * BLOCK_N * stride_vk
    # loop over k, v, and update accumulator
    # For padded blocks, we will overrun the tensor size if
    # we load all BLOCK_N. For others, the blocks are all within range.
    k_offs_n = block_min + tl.arange(0, BLOCK_N)
    k_offs_k = None
    k = load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, N_CTX)
    if PRE_LOAD_V:
        # We can use the same offsets as k, just with dims transposed.
        v = load_fn(v_ptrs, k_offs_n, k_offs_k, N_CTX, BLOCK_DMODEL)
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # We start from end of seqlen_k so only the first iteration would need
    # to be checked for padding if it is not a multiple of block_n
    # If this is the last block / iteration, we want to
    # mask if the sequence length is not a multiple of block size
    # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
    # last step might get wasted but that is okay. check if this masking works For
    # that case.
    # if (block_min + BLOCK_N == block_max) and (n_extra_tokens != 0):
    boundary_m = tl.full([BLOCK_M], N_CTX, dtype=tl.int32)
    size_n = block_min + offs_n[None, :]
    mask = size_n < boundary_m[:, None]
    qk = tl.where(mask, qk, float("-inf"))
    # -- compute qk ----
    qk += tl.dot(q, k)
    # softmax
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)

    # -- update output accumulator --
    alpha = tl.math.exp2(m_i - m_ij)
    acc = acc * alpha[:, None]
    if not PRE_LOAD_V:
        v = load_fn(v_ptrs, k_offs_n, k_offs_k, N_CTX, BLOCK_DMODEL)
    # -- update l_i
    l_ij = tl.sum(p, 1)
    l_i = l_i * alpha + l_ij
    acc += tl.dot(p.to(v.type.element_ty), v)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip

    # write back O
    o_offset = Out + off_z * stride_oz + off_h * stride_oh
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    acc = acc.to(Out.type.element_ty)

    end_m_idx = (start_m + 1) * BLOCK_M
    overflow_size = end_m_idx - N_CTX
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < N_CTX)
    tl.store(o_ptrs, acc, mask=o_ptrs_mask)


def get_shape_from_layout(q, layout):
    if layout == "bhsd":
        batch, n_heads, n_ctx, head_size = q.shape
    elif layout == "bshd":
        batch, n_ctx, n_heads, head_size = q.shape
    else:
        assert False, "Got unsupported layout."
    return batch, n_heads, n_ctx, head_size


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout(q, k, v, o, layout):
    if layout == "bhsd":
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif layout == "bshd":
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, "Got unsupported layout."
    return q_strides, k_strides, v_strides, o_strides


@torch.library.custom_op("triton_fmha::triton_op", mutates_args=(), device_types="cuda")
def fmha_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str = "bhsd") -> torch.Tensor:
    o = torch.empty_like(q, dtype=v.dtype)

    batch, n_heads, n_ctx, head_size = get_shape_from_layout(q, layout)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    grid = lambda META: (triton.cdiv(n_ctx, META["BLOCK_M"]), n_heads, batch)  # noqa: E731

    # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = 1.0 / (head_size**0.5) * math.log2(math.e)
    compiled_kernel = attn_fwd[grid](
        q,
        k,
        v,
        qk_scale,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        N_CTX=n_ctx,
        BLOCK_DMODEL=head_size,
        # BLOCK_M=128,
        # BLOCK_N=128,
        # waves_per_eu=2,
        # PRE_LOAD_V=False,
        # num_stages=1,
        # num_warps=4,
    )

    return o


@fmha_fwd.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layout: str = "bhsd") -> torch.Tensor:
    return torch.empty_like(q)


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == "bhsd":
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == "bshd":
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, "Got unsupported tensor layout"
    q = torch.randn(q_tensor_shape, dtype=dtype, device="cuda")
    k = torch.randn(k_tensor_shape, dtype=dtype, device="cuda")
    v = torch.randn(k_tensor_shape, dtype=dtype, device="cuda")

    return q, k, v


attention = fmha_fwd
