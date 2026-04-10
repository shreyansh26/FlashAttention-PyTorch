"""FA1: baseline tiled online-softmax attention.

Simplified algorithm in this module:
1. Split the query rows and key/value rows into tiles.
2. Keep one output tile together with per-row running statistics:
   the running row max and the running normalization sum.
3. Stream over K/V tiles outside the Q loop, compute the local score block,
   update the online softmax state, and immediately fold the weighted value
   contribution into the normalized output tile.
4. In backward, recompute the local probabilities from the saved LSE-style
   statistics instead of storing the full attention matrix.

This is the closest educational version to the original FlashAttention paper.
Its main improvement over naive attention is IO-awareness: it never
materializes the full score or probability matrix in memory, and instead keeps
only compact row-wise statistics plus the running output tile.
"""

from __future__ import annotations

from flash_attention_core.common import (
    backward_step,
    block_scores_and_mask,
    compute_local_statistics,
    iter_block_slices,
    lse_from_state,
    merge_state_normalized,
    prepare_inputs,
)
from flash_attention_core.config import FlashAttentionConfig
from flash_attention_core.types import BackwardResult, ForwardResult

import torch


def forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> ForwardResult:
    config = config or FlashAttentionConfig()
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        causal=causal,
        key_padding_mask=key_padding_mask,
    )

    q_slices = iter_block_slices(q.shape[2], config.block_size_q)
    k_slices = iter_block_slices(k.shape[2], config.block_size_kv)
    out_blocks = [torch.zeros_like(q[:, :, q_slice, :]) for q_slice in q_slices]
    normalizer_blocks = [
        torch.zeros(
            q.shape[0],
            q.shape[1],
            (q_slice.stop or q.shape[2]) - (q_slice.start or 0),
            1,
            device=q.device,
            dtype=torch.float32,
        )
        for q_slice in q_slices
    ]
    row_max_blocks = [torch.full_like(block, float("-inf")) for block in normalizer_blocks]
    debug_state = {"loop_order": "kv_outer_q_inner"} if config.keep_debug_state else {}

    for k_slice in k_slices:
        k_block = k[:, :, k_slice, :]
        v_block = v[:, :, k_slice, :]

        for q_index, q_slice in enumerate(q_slices):
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            block_max, block_sum, weighted_values = compute_local_statistics(scores, valid_mask, v_block)
            out_blocks[q_index], normalizer_blocks[q_index], row_max_blocks[q_index] = merge_state_normalized(
                out_blocks[q_index],
                normalizer_blocks[q_index],
                row_max_blocks[q_index],
                block_max,
                block_sum,
                weighted_values,
            )

    out = torch.cat(out_blocks, dim=2)
    normalizers = torch.cat(normalizer_blocks, dim=2)
    row_max = torch.cat(row_max_blocks, dim=2)

    return ForwardResult(
        out=out,
        lse=lse_from_state(normalizers, row_max),
        normalizers=normalizers,
        row_max=row_max,
        saved_state=debug_state,
    )


def backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    forward_result: ForwardResult,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> BackwardResult:
    config = config or FlashAttentionConfig()
    q, k, key_padding_mask = prepare_inputs(
        q,
        k,
        v,
        causal=causal,
        key_padding_mask=key_padding_mask,
    )
    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)

    q_slices = iter_block_slices(q.shape[2], config.block_size_q)
    k_slices = iter_block_slices(k.shape[2], config.block_size_kv)

    for k_slice in k_slices:
        k_block = k[:, :, k_slice, :]
        v_block = v[:, :, k_slice, :]
        dK_block = torch.zeros_like(k_block)
        dV_block = torch.zeros_like(v_block)

        for q_slice in q_slices:
            q_block = q[:, :, q_slice, :]
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            local_dQ, local_dK, local_dV = backward_step(
                q_block=q_block,
                k_block=k_block,
                v_block=v_block,
                out_block=forward_result.out[:, :, q_slice, :],
                grad_out_block=grad_out[:, :, q_slice, :],
                scores=scores,
                valid_mask=valid_mask,
                lse_block=forward_result.lse[:, :, q_slice, :],
            )
            dQ[:, :, q_slice, :] += local_dQ
            dK_block += local_dK
            dV_block += local_dV

        dK[:, :, k_slice, :] = dK_block
        dV[:, :, k_slice, :] = dV_block

    return BackwardResult(dQ=dQ, dK=dK, dV=dV)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
    config: FlashAttentionConfig | None = None,
) -> torch.Tensor:
    return forward(
        q,
        k,
        v,
        causal=causal,
        key_padding_mask=key_padding_mask,
        config=config,
    ).out
