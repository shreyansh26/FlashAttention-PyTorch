"""FA3: staged producer/consumer pipeline with ping-pong buffers.

Simplified algorithm in this module:
1. Keep the FA2-style exact attention math, but reorganize execution into
   explicit pipeline stages rather than a single monolithic loop body.
2. Model a producer/consumer handoff with two logical tile buffers: one buffer
   is "active" for the current K/V tile while the next tile is logically
   prefetched into the other buffer.
3. For each query tile, run the stages in order: load tile, compute score
   block, update online softmax statistics, and apply the value contribution.
4. Mirror the same staged structure in backward so the simplified code still
   depicts the overlap-friendly orchestration.

Compared with FA2, the educational improvement is pipeline structure. Real FA3
uses Hopper features such as TMA, WGMMA, warp specialization, and ping-pong
buffering to overlap data movement and compute. This module leaves those
hardware mechanisms out, but keeps the stage boundaries and double-buffered
control flow visible in plain PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class TileBuffer:
    buffer_id: int
    tile_id: int
    tile_slice: slice
    k_block: torch.Tensor
    v_block: torch.Tensor


def _load_tile(
    *,
    buffer_id: int,
    tile_id: int,
    tile_slice: slice,
    k: torch.Tensor,
    v: torch.Tensor,
) -> TileBuffer:
    return TileBuffer(
        buffer_id=buffer_id,
        tile_id=tile_id,
        tile_slice=tile_slice,
        k_block=k[:, :, tile_slice, :],
        v_block=v[:, :, tile_slice, :],
    )


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
    pipeline_trace = []

    for q_tile_id, q_slice in enumerate(q_slices):
        if not k_slices:
            continue

        active_buffer = _load_tile(
            buffer_id=0,
            tile_id=0,
            tile_slice=k_slices[0],
            k=k,
            v=v,
        )

        for kv_tile_id, _ in enumerate(k_slices):
            next_buffer = None
            if kv_tile_id + 1 < len(k_slices):
                next_buffer = _load_tile(
                    buffer_id=(active_buffer.buffer_id + 1) % max(config.num_stages, 2),
                    tile_id=kv_tile_id + 1,
                    tile_slice=k_slices[kv_tile_id + 1],
                    k=k,
                    v=v,
                )

            # Real FA3 overlaps producer and consumer warpgroups here. We keep the
            # stages explicit, but execute them sequentially for clarity.
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=active_buffer.tile_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                active_buffer.v_block,
            )
            out_acc_blocks[q_tile_id], normalizer_blocks[q_tile_id], row_max_blocks[q_tile_id] = merge_state_unnormalized(
                out_acc_blocks[q_tile_id],
                normalizer_blocks[q_tile_id],
                row_max_blocks[q_tile_id],
                block_max,
                block_sum,
                weighted_values,
            )
            pipeline_trace.append(
                {
                    "q_tile": q_tile_id,
                    "kv_tile": active_buffer.tile_id,
                    "buffer_id": active_buffer.buffer_id,
                    "prefetched_next_tile": next_buffer.tile_id if next_buffer is not None else -1,
                }
            )
            active_buffer = next_buffer
            if active_buffer is None:
                break

    out_acc = torch.cat(out_acc_blocks, dim=2)
    normalizers = torch.cat(normalizer_blocks, dim=2)
    row_max = torch.cat(row_max_blocks, dim=2)
    out = finalize_unnormalized(out_acc, normalizers)
    return ForwardResult(
        out=out,
        lse=lse_from_state(normalizers, row_max),
        normalizers=normalizers,
        row_max=row_max,
        saved_state={"pipeline_trace": pipeline_trace} if config.keep_debug_state else {},
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
    pipeline_trace = []

    for q_tile_id, q_slice in enumerate(q_slices):
        if not k_slices:
            continue

        q_block = q[:, :, q_slice, :]
        dQ_block = torch.zeros_like(q_block)
        active_buffer = _load_tile(
            buffer_id=0,
            tile_id=0,
            tile_slice=k_slices[0],
            k=k,
            v=v,
        )

        for kv_tile_id, _ in enumerate(k_slices):
            next_buffer = None
            if kv_tile_id + 1 < len(k_slices):
                next_buffer = _load_tile(
                    buffer_id=(active_buffer.buffer_id + 1) % max(config.num_stages, 2),
                    tile_id=kv_tile_id + 1,
                    tile_slice=k_slices[kv_tile_id + 1],
                    k=k,
                    v=v,
                )

            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=q_slice,
                k_slice=active_buffer.tile_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            local_dQ, local_dK, local_dV = backward_step(
                q_block=q_block,
                k_block=active_buffer.k_block,
                v_block=active_buffer.v_block,
                out_block=forward_result.out[:, :, q_slice, :],
                grad_out_block=grad_out[:, :, q_slice, :],
                scores=scores,
                valid_mask=valid_mask,
                lse_block=forward_result.lse[:, :, q_slice, :],
            )
            dQ_block += local_dQ
            dK[:, :, active_buffer.tile_slice, :] += local_dK
            dV[:, :, active_buffer.tile_slice, :] += local_dV
            pipeline_trace.append(
                {
                    "q_tile": q_tile_id,
                    "kv_tile": active_buffer.tile_id,
                    "buffer_id": active_buffer.buffer_id,
                }
            )
            active_buffer = next_buffer
            if active_buffer is None:
                break

        dQ[:, :, q_slice, :] = dQ_block

    return BackwardResult(
        dQ=dQ,
        dK=dK,
        dV=dV,
        debug_state={"pipeline_trace": pipeline_trace} if config.keep_debug_state else {},
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
