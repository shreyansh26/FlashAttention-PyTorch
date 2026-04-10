"""FA4: scheduled tiles with explicit main/softmax/correction phases.

This educational version keeps the exact math identical to regular attention,
but exposes the FA4-style role split and conditional rescaling. Real FA4 uses
TMEM, async MMA pipelines, and more sophisticated tile schedulers; those
hardware details are left in comments instead of being emulated directly.
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


@dataclass(frozen=True)
class ScheduledTile:
    wave_id: int
    query_tile: int
    key_tile: int
    q_slice: slice
    k_slice: slice


def _build_schedule(q_slices: list[slice], k_slices: list[slice]) -> list[list[ScheduledTile]]:
    waves = []
    for wave_id, k_slice in enumerate(k_slices):
        waves.append(
            [
                ScheduledTile(
                    wave_id=wave_id,
                    query_tile=query_tile,
                    key_tile=wave_id,
                    q_slice=q_slice,
                    k_slice=k_slice,
                )
                for query_tile, q_slice in enumerate(q_slices)
            ]
        )
    return waves


def _correction_merge(
    *,
    out_acc_block: torch.Tensor,
    normalizer_block: torch.Tensor,
    row_max_block: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    requires_rescale = block_max > row_max_block
    merged_out_acc, merged_normalizer, merged_row_max = merge_state_unnormalized(
        out_acc_block,
        normalizer_block,
        row_max_block,
        block_max,
        block_sum,
        weighted_values,
    )
    return merged_out_acc, merged_normalizer, merged_row_max, requires_rescale


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
    schedule = _build_schedule(q_slices, k_slices)
    scheduler_trace = []

    for wave in schedule:
        main_outputs = []
        for task in wave:
            main_outputs.append(
                (
                    task,
                    *block_scores_and_mask(
                        q=q,
                        k=k,
                        q_slice=task.q_slice,
                        k_slice=task.k_slice,
                        causal=causal,
                        key_padding_mask=key_padding_mask,
                    ),
                )
            )

        softmax_outputs = []
        for task, scores, valid_mask in main_outputs:
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                v[:, :, task.k_slice, :],
            )
            softmax_outputs.append((task, block_max, block_sum, weighted_values))

        for task, block_max, block_sum, weighted_values in softmax_outputs:
            out_acc_blocks[task.query_tile], normalizer_blocks[task.query_tile], row_max_blocks[task.query_tile], rescaled = (
                _correction_merge(
                    out_acc_block=out_acc_blocks[task.query_tile],
                    normalizer_block=normalizer_blocks[task.query_tile],
                    row_max_block=row_max_blocks[task.query_tile],
                    block_max=block_max,
                    block_sum=block_sum,
                    weighted_values=weighted_values,
                )
            )
            scheduler_trace.append(
                {
                    "wave_id": task.wave_id,
                    "query_tile": task.query_tile,
                    "key_tile": task.key_tile,
                    "rescaled": bool(rescaled.any().item()),
                }
            )

    out_acc = torch.cat(out_acc_blocks, dim=2)
    normalizers = torch.cat(normalizer_blocks, dim=2)
    row_max = torch.cat(row_max_blocks, dim=2)
    out = finalize_unnormalized(out_acc, normalizers)
    return ForwardResult(
        out=out,
        lse=lse_from_state(normalizers, row_max),
        normalizers=normalizers,
        row_max=row_max,
        saved_state={"scheduler_trace": scheduler_trace} if config.keep_debug_state else {},
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
    schedule = _build_schedule(q_slices, k_slices)
    scheduler_trace = []

    for wave in schedule:
        main_outputs = []
        for task in wave:
            scores, valid_mask = block_scores_and_mask(
                q=q,
                k=k,
                q_slice=task.q_slice,
                k_slice=task.k_slice,
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            main_outputs.append((task, scores, valid_mask))

        softmax_outputs = []
        for task, scores, valid_mask in main_outputs:
            local_dQ, local_dK, local_dV = backward_step(
                q_block=q[:, :, task.q_slice, :],
                k_block=k[:, :, task.k_slice, :],
                v_block=v[:, :, task.k_slice, :],
                out_block=forward_result.out[:, :, task.q_slice, :],
                grad_out_block=grad_out[:, :, task.q_slice, :],
                scores=scores,
                valid_mask=valid_mask,
                lse_block=forward_result.lse[:, :, task.q_slice, :],
            )
            softmax_outputs.append((task, local_dQ, local_dK, local_dV))

        for task, local_dQ, local_dK, local_dV in softmax_outputs:
            dQ[:, :, task.q_slice, :] += local_dQ
            dK[:, :, task.k_slice, :] += local_dK
            dV[:, :, task.k_slice, :] += local_dV
            scheduler_trace.append(
                {
                    "wave_id": task.wave_id,
                    "query_tile": task.query_tile,
                    "key_tile": task.key_tile,
                }
            )

    return BackwardResult(
        dQ=dQ,
        dK=dK,
        dV=dV,
        debug_state={"scheduler_trace": scheduler_trace} if config.keep_debug_state else {},
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
