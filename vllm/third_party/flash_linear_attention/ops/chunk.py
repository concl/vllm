# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch
import triton
import triton.language as tl

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    g = chunk_local_cumsum(
        g, chunk_size=FLA_CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


@triton.jit
def _gdn_sequential_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    o_ptr,
    h0_ptr,
    ht_ptr,
    cu_ptr,
    T,
    HG,
    H,
    K,
    V,
    sqb,
    sqt,
    sqh,
    svb,
    svt,
    svh,
    sgb,
    sgt,
    sgh,
    scale,
    HAS_INITIAL_STATE: tl.constexpr,
    USE_CU_SEQLENS: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    pid = tl.program_id(0)
    num_v_blocks = tl.cdiv(V, BV)
    vb = pid % num_v_blocks
    rest = pid // num_v_blocks
    head = rest % H
    seq = rest // H

    offs_k = tl.arange(0, BK)
    offs_v = vb * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    state_mask = mask_v[:, None] & mask_k[None, :]

    if HAS_INITIAL_STATE:
        state = tl.load(
            h0_ptr + ((seq * H + head) * V + offs_v[:, None]) * K + offs_k[None, :],
            mask=state_mask,
            other=0.0,
        ).to(tl.float32)
    else:
        state = tl.zeros((BV, BK), dtype=tl.float32)

    if USE_CU_SEQLENS:
        start = tl.load(cu_ptr + seq).to(tl.int32)
        end = tl.load(cu_ptr + seq + 1).to(tl.int32)
        batch = 0
    else:
        start = 0
        end = T
        batch = seq

    # Key/query heads are fewer than value heads: value head h reads q/k head
    # h // (H // HG), matching the chunked kernels.
    qk_head = head // (H // HG)
    q_base = batch * sqb + qk_head * sqh
    v_base = batch * svb + head * svh
    g_base = batch * sgb + head * sgh

    for t in tl.range(start, end):
        qk_off = q_base + t * sqt
        q = tl.load(q_ptr + qk_off + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + qk_off + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        q = q * scale

        v_off = v_base + t * svt
        value = tl.load(v_ptr + v_off + offs_v, mask=mask_v, other=0.0).to(tl.float32)

        gb_off = g_base + t * sgt
        log_gate = tl.load(g_ptr + gb_off).to(tl.float32)
        beta_t = tl.load(beta_ptr + gb_off).to(tl.float32)

        state = state * tl.exp(log_gate)

        current = tl.sum(state * k[None, :], axis=1)
        delta = beta_t * (value - current)
        state = state + delta[:, None] * k[None, :]

        out = tl.sum(state * q[None, :], axis=1)
        tl.store(o_ptr + v_off + offs_v, out.to(o_ptr.dtype.element_ty), mask=mask_v)

    if STORE_FINAL_STATE:
        tl.store(
            ht_ptr + ((seq * H + head) * V + offs_v[:, None]) * K + offs_k[None, :],
            state,
            mask=state_mask,
        )


def gdn_test_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
    *,
    v_block: int = 16,
):
    """Token-by-token gated delta rule, one Triton kernel launch.

    Same contract as ``chunk_gated_delta_rule``: inputs are ``[B, T, H, K]`` /
    ``[B, T, H, V]`` / ``[B, T, H]``, outputs are ``o`` in the input dtype and
    the final state in fp32. ``chunk_indices``/``chunk_offsets`` are accepted
    for signature compatibility and ignored: chunking is what this
    implementation exists to avoid. ``use_qk_l2norm_in_kernel`` is ignored too,
    so q/k must already be normalized when the delta rule requires it.
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    assert q.shape == k.shape
    assert q.stride() == k.stride()

    batch, num_tokens, num_qk_heads, head_k_dim = q.shape
    num_heads, head_v_dim = v.shape[2], v.shape[-1]
    assert num_heads % num_qk_heads == 0, (
        f"value heads ({num_heads}) must be a multiple of q/k heads "
        f"({num_qk_heads})"
    )

    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        num_seqs = cu_seqlens.numel() - 1
    else:
        num_seqs = batch

    if initial_state is not None:
        initial_state = initial_state.contiguous()

    o = torch.empty_like(v)
    final_state = (
        torch.empty(
            num_seqs,
            num_heads,
            head_v_dim,
            head_k_dim,
            dtype=torch.float32,
            device=q.device,
        )
        if output_final_state
        else None
    )

    grid = (num_seqs * num_heads * triton.cdiv(head_v_dim, v_block),)
    unused = q
    _gdn_sequential_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        o,
        initial_state if initial_state is not None else unused,
        final_state if final_state is not None else unused,
        cu_seqlens if cu_seqlens is not None else unused,
        num_tokens,
        num_qk_heads,
        num_heads,
        head_k_dim,
        head_v_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        g.stride(0),
        g.stride(1),
        g.stride(2),
        float(scale),
        HAS_INITIAL_STATE=initial_state is not None,
        USE_CU_SEQLENS=cu_seqlens is not None,
        STORE_FINAL_STATE=final_state is not None,
        BK=triton.next_power_of_2(head_k_dim),
        BV=v_block,
    )

    if core_attn_out is not None:
        core_attn_out.reshape(-1)[: o.numel()].copy_(o.reshape(-1))

    return o, final_state


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        # Educational path: sequential token-by-token kernel (see gdn_test_fwd).
        # The chunked implementation is kept below for reference.
        #
        # g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
        #     q=q,
        #     k=k,
        #     v=v,
        #     g=g,
        #     beta=beta,
        #     scale=scale,
        #     initial_state=initial_state,
        #     output_final_state=output_final_state,
        #     cu_seqlens=cu_seqlens,
        #     chunk_indices=chunk_indices,
        #     chunk_offsets=chunk_offsets,
        #     core_attn_out=core_attn_out,
        # )
        o, final_state = gdn_test_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            core_attn_out=core_attn_out,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        if core_attn_out is not None:
            assert not torch.is_grad_enabled(), (
                "core_attn_out buffer reuse is only supported for inference"
            )
            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        core_attn_out,
    )
    return o, final_state
