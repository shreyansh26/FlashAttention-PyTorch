"""FA2: sequence-parallel ownership with deferred output normalization.

Simplified algorithm in this module:
1. Assign ownership of one query tile to each outer-loop iteration.
2. For that owned query tile, stream over all K/V tiles and accumulate an
   unnormalized output tile together with the running row max and row sum.
3. Only after all K/V tiles have been processed for that query tile, apply the
   final normalization step.
4. In backward, keep the same ownership-oriented orchestration so the control
   flow reflects the split-Q / sequence-parallel idea instead of the FA1
   KV-outer loop structure.

Compared with FA1, the educational improvement is the work partitioning:
the code is organized around query-tile ownership, deferred normalization, and
LSE-centered saved state. Real FA2 uses this style to expose more sequence
parallelism and reduce non-matmul overhead; here we mirror that algorithmic
shape without reproducing the CUDA launch details.
"""

from __future__ import annotations

import torch

from flash_attention_core.common import (
    backward_step,
    block_scores_and_mask,
    compute_local_statistics,
    finalize_unnormalized,
    iter_block_slices,
    lse_from_state,
    merge_state_unnormalized,
    prepare_inputs,
)
from flash_attention_core.config import FlashAttentionConfig
from flash_attention_core.types import BackwardResult, ForwardResult


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
    # FA2 changes the ownership model: the outer loop now "owns" one query tile
    # and streams every K/V tile through it. This mirrors split-Q style work
    # partitioning, which improves occupancy when sequence length is large.
    out_acc_blocks = [torch.zeros_like(q[:, :, q_slice, :]) for q_slice in q_slices]
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
    query_owners = []

    for owner_id, q_slice in enumerate(q_slices):
        query_owners.append(
            {
                "owner_id": owner_id,
                "q_start": q_slice.start or 0,
                "q_end": q_slice.stop or q.shape[2],
                "num_kv_tiles": len(k_slices),
            }
        )

        for k_slice in k_slices:
            # Real FA2 would launch multiple CTAs over query tiles so these owners
            # run concurrently. Here we keep the same control-flow structure but
            # execute it sequentially for clarity.
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                v[:, :, k_slice, :],
            )
            # Unlike FA1, we intentionally keep the output tile unnormalized while
            # streaming K/V tiles. The final normalization is deferred until the
            # owner has seen the full K/V sequence.
            out_acc_blocks[owner_id], normalizer_blocks[owner_id], row_max_blocks[owner_id] = merge_state_unnormalized(
                out_acc_blocks[owner_id],
                normalizer_blocks[owner_id],
                row_max_blocks[owner_id],
                block_max,
                block_sum,
                weighted_values,
            )

    out_acc = torch.cat(out_acc_blocks, dim=2)
    normalizers = torch.cat(normalizer_blocks, dim=2)
    row_max = torch.cat(row_max_blocks, dim=2)
    # This late normalization is one of the main algorithmic cleanups in FA2:
    # fewer rescale operations are performed inside the tiled main loop.
    out = finalize_unnormalized(out_acc, normalizers)
    debug_state = {
        "query_owners": query_owners,
        "deferred_normalization": True,
        # Real FA2 stores log-sum-exp style state and performs sequence-parallel work
        # in different CUDA threadblocks. This simplified version keeps those ideas
        # visible without reproducing kernel-level launches.
    }
    return ForwardResult(
        out=out,
        lse=lse_from_state(normalizers, row_max),
        normalizers=normalizers,
        row_max=row_max,
        saved_state=debug_state if config.keep_debug_state else {},
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
    owner_trace = []

    for owner_id, q_slice in enumerate(q_slices):
        q_block = q[:, :, q_slice, :]
        # Keep gradients local to the owning query tile first, then write the
        # finished dQ tile back once all K/V tiles have been processed.
        dQ_block = torch.zeros_like(q_block)

        for k_slice in k_slices:
            k_block = k[:, :, k_slice, :]
            v_block = v[:, :, k_slice, :]
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            # The derivative formulas are the same exact attention derivatives as
            # FA1. The educational difference is purely in the orchestration.
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
            dQ_block += local_dQ
            dK[:, :, k_slice, :] += local_dK
            dV[:, :, k_slice, :] += local_dV

        dQ[:, :, q_slice, :] = dQ_block
        owner_trace.append({"owner_id": owner_id, "num_kv_tiles": len(k_slices)})

    return BackwardResult(
        dQ=dQ,
        dK=dK,
        dV=dV,
        debug_state={"query_owner_trace": owner_trace} if config.keep_debug_state else {},
    )


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
