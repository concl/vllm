# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Submission test for the chunked gated delta rule.

``chunk_gated_delta_rule`` is imported from its source file
(vllm/third_party/flash_linear_attention/ops/chunk.py), so a passing branch must
actually reimplement it there.

The oracle is an fp32 token-by-token recurrence -- the definition of the op --
not another kernel, so it does not depend on FLA's numerics and gives a much
sharper error signal than a bf16-vs-bf16 comparison.

Run:

    .venv/bin/python -m pytest tests/kernels/fla/test_submission.py -v

"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_gated_delta_rule,
)

if not (current_platform.is_cuda() or current_platform.is_rocm()):
    pytest.skip(
        reason="The chunked kernel needs CUDA or ROCm.",
        allow_module_level=True,
    )

DEVICE = current_platform.device_type
DTYPE = torch.bfloat16
NUM_HEADS = 4
HEAD_K_DIM = 64
HEAD_V_DIM = 64

# Measured max error of the current implementation against the fp32 recurrence
# is 7.8e-3 at T=160, i.e. ~1-2 ulp of bf16 for outputs whose magnitude is O(1).
ATOL = 2e-2
MEAN_ATOL = 1e-3


def _recurrence(q, k, v, g, beta, state):
    """One sequence, fp32.

    q/k: [L,H,K], v: [L,H,V], g/beta: [L,H], state: [H,V,K].
    """
    length = q.shape[0]
    out = torch.zeros(
        length, q.shape[1], v.shape[-1], dtype=torch.float32, device=q.device
    )
    for t in range(length):
        state = state * torch.exp(g[t]).unsqueeze(-1).unsqueeze(-1)
        k_t = k[t].unsqueeze(-1)  # [H, K, 1]
        beta_t = beta[t].unsqueeze(-1).unsqueeze(-1)  # [H, 1, 1]
        delta = beta_t * (v[t].unsqueeze(-1) - torch.matmul(state, k_t))
        state = state + torch.matmul(delta, k_t.transpose(-1, -2))
        out[t] = torch.matmul(state, q[t].unsqueeze(-1)).squeeze(-1)
    return out, state


def golden_recurrent_ref(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
):
    """Ground-truth sequential delta rule, computed in fp32.

        S_t = exp(g_t) * S_{t-1} + beta_t * (v_t - S_{t-1} k_t) k_t^T
        o_t = S_t (q_t / sqrt(D_k))

    with S in R^{V x K}, so initial_state/final_state are [N, H, V, K] and o is
    [B, T, H, V]. final_state is always fp32.
    """
    batch, _, num_heads, head_k_dim = q.shape
    num_v_heads, head_v_dim = v.shape[2], v.shape[-1]
    if scale is None:
        scale = head_k_dim**-0.5

    orig_dtype = q.dtype
    q_f = (q * scale).to(torch.float32)
    k_f, v_f, g_f, beta_f = (x.to(torch.float32) for x in (k, v, g, beta))
    out = torch.zeros(
        batch,
        q.shape[1],
        num_v_heads,
        head_v_dim,
        dtype=torch.float32,
        device=q.device,
    )

    if cu_seqlens is None:
        spans = [(b, 0, q.shape[1]) for b in range(batch)]
    else:
        spans = [
            (0, int(cu_seqlens[i]), int(cu_seqlens[i + 1]))
            for i in range(len(cu_seqlens) - 1)
        ]

    finals = []
    for seq_idx, (b, start, end) in enumerate(spans):
        if initial_state is None:
            state = torch.zeros(
                num_heads, head_v_dim, head_k_dim,
                dtype=torch.float32, device=q.device,
            )
        else:
            state = initial_state[seq_idx].to(dtype=torch.float32, device=q.device)
            state = state.clone()
        o_seq, state = _recurrence(
            q_f[b, start:end],
            k_f[b, start:end],
            v_f[b, start:end],
            g_f[b, start:end],
            beta_f[b, start:end],
            state,
        )
        out[b, start:end] = o_seq
        finals.append(state)

    final_state = torch.stack(finals) if output_final_state else None
    return out.to(orig_dtype), final_state


def make_inputs(batch, num_tokens, seed=1337, device=DEVICE):
    """Numerically well-conditioned inputs.

    Unit-norm keys and non-positive log gates keep the state bounded, so
    tolerance failures mean a logic error rather than an explosion.
    """
    torch.manual_seed(seed)
    q = torch.randn(
        batch, num_tokens, NUM_HEADS, HEAD_K_DIM, device=device, dtype=DTYPE
    )
    k = F.normalize(
        torch.randn(
            batch, num_tokens, NUM_HEADS, HEAD_K_DIM, device=device, dtype=DTYPE
        ),
        dim=-1,
    )
    v = torch.randn(
        batch, num_tokens, NUM_HEADS, HEAD_V_DIM, device=device, dtype=DTYPE
    )
    # Decay in (0, 1]: exp of a non-positive log gate.
    g = -F.softplus(torch.randn(batch, num_tokens, NUM_HEADS, device=device))
    # Interpolation in (0, 1).
    beta = torch.sigmoid(torch.randn(batch, num_tokens, NUM_HEADS, device=device))
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta}


def assert_matches(actual, expected, name):
    assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
    error = (actual.float() - expected.float()).abs()
    assert error.max().item() < ATOL, (
        f"{name}: max abs error {error.max().item():.3e} >= {ATOL:.1e}"
    )
    assert error.mean().item() < MEAN_ATOL, (
        f"{name}: mean abs error {error.mean().item():.3e} >= {MEAN_ATOL:.1e}"
    )


@pytest.mark.parametrize("num_tokens", [16, 64, 128, 160])
def test_matches_recurrent_reference(num_tokens):
    """16 < one chunk, 64 == one chunk, 128 == two, 160 == two plus a ragged tail."""
    batch = 2
    inputs = make_inputs(batch, num_tokens)

    ref_o, ref_state = golden_recurrent_ref(**inputs, output_final_state=True)
    out, final_state = chunk_gated_delta_rule(
        **inputs, output_final_state=True, use_qk_l2norm_in_kernel=False
    )

    assert out.shape == (batch, num_tokens, NUM_HEADS, HEAD_V_DIM)
    assert out.dtype == DTYPE
    assert final_state.shape == (batch, NUM_HEADS, HEAD_V_DIM, HEAD_K_DIM)
    assert final_state.dtype == torch.float32

    assert_matches(out, ref_o, "o")
    assert_matches(final_state, ref_state, "final_state")


def test_initial_state_carryover():
    """A prefill that starts from an existing decode state."""
    batch, num_tokens = 2, 64
    inputs = make_inputs(batch, num_tokens, seed=2024)
    torch.manual_seed(7)
    initial_state = torch.randn(
        batch, NUM_HEADS, HEAD_V_DIM, HEAD_K_DIM, device=DEVICE, dtype=DTYPE
    )

    ref_o, _ = golden_recurrent_ref(**inputs, initial_state=initial_state)
    out, _ = chunk_gated_delta_rule(
        **inputs, initial_state=initial_state, use_qk_l2norm_in_kernel=False
    )
    assert_matches(out, ref_o, "o with initial_state")


def test_varlen_cu_seqlens():
    """Ragged batch: chunk boundaries must not leak across sequences."""
    seq_lengths = [40, 128, 7]
    total = sum(seq_lengths)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lengths).cumsum(0).tolist()],
        dtype=torch.int32,
        device=DEVICE,
    )
    inputs = make_inputs(1, total, seed=99)
    torch.manual_seed(11)
    initial_state = torch.randn(
        len(seq_lengths),
        NUM_HEADS,
        HEAD_V_DIM,
        HEAD_K_DIM,
        device=DEVICE,
        dtype=DTYPE,
    )

    ref_o, ref_state = golden_recurrent_ref(
        **inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    out, final_state = chunk_gated_delta_rule(
        **inputs,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    assert_matches(out, ref_o, "o (varlen)")
    assert_matches(final_state, ref_state, "final_state (varlen)")


def test_output_final_state_false():
    inputs = make_inputs(1, 32)
    _, final_state = chunk_gated_delta_rule(**inputs, use_qk_l2norm_in_kernel=False)
    assert final_state is None
