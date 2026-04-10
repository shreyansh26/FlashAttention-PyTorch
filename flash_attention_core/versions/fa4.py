"""FA4: scheduled tiles with explicit main/softmax/correction phases.

Simplified algorithm in this module:
1. Build an explicit tile schedule that groups work into waves, so execution is
   driven by scheduler metadata rather than only by nested loops.
2. Split each wave into three conceptual roles:
   a main score-production phase, a softmax-statistics phase, and a correction
   phase that merges the new contribution into the running output state.
3. Track whether a merge requires rescaling the accumulated state, so the code
   makes the late / conditional rescaling decision visible even though the
   actual tensor path remains mathematically exact and sequential.
4. Reuse the same scheduled-wave view in backward so the version still reads
   like a scheduler-driven algorithm rather than a generic tiled kernel.

Compared with FA3, the educational improvement is the explicit scheduler and
role split. Real FA4 further co-designs the algorithm around Blackwell-era
features such as TMEM, async MMA, multi-role warpgroups, and deeper overlap.
This module does not simulate those hardware details, but it does expose the
main/softmax/correction decomposition and the conditional-rescaling idea that
differentiate FA4 from the earlier versions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

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


def _fa4_rescale_threshold(dtype: torch.dtype) -> float:
    # Official FA4 sets `rescale_threshold=8.0` for 16-bit query types and 0.0
    # otherwise. We mirror that policy here.
    return 8.0 if dtype in (torch.float16, torch.bfloat16) else 0.0


def _build_schedule(q_slices: list[slice], k_slices: list[slice]) -> list[list[ScheduledTile]]:
    waves = []
    for wave_id, q_slice in enumerate(q_slices):
        # The schedule is intentionally explicit so readers can see that FA4 is
        # driven by scheduler metadata, not just by plain nested loops. Real FA4
        # uses a richer scheduler to map tiles to warpgroups / CTA roles.
        waves.append(
            [
                ScheduledTile(
                    wave_id=wave_id,
                    query_tile=wave_id,
                    key_tile=key_tile,
                    q_slice=q_slice,
                    k_slice=k_slice,
                )
                for key_tile, k_slice in enumerate(k_slices)
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
    scale_log2: float,
    rescale_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_row_max_block = torch.where(torch.isfinite(row_max_block), row_max_block, block_max)
    acc_scale_log2 = (safe_row_max_block - block_max) * scale_log2
    requires_rescale = block_max > row_max_block
    if rescale_threshold > 0.0:
        requires_rescale = requires_rescale & (acc_scale_log2 < -rescale_threshold)

    # This helper represents the "correction" role in FA4. In the real kernels,
    # that role is separated so output rescaling and correction do not block the
    # main compute path. Here we keep the separation conceptually, but execute it
    # inline in a plain tensor function.
    merged_out_acc, merged_normalizer, merged_row_max = merge_state_unnormalized(
        out_acc_block,
        normalizer_block,
        row_max_block,
        block_max,
        block_sum,
        weighted_values,
    )

    # Official FA4 uses a thresholded selective-rescaling rule. Small enough max
    # changes keep the old row max and use unit rescaling instead of fully
    # rescaling the accumulated state. We reproduce that structure here.
    same_scale = ~requires_rescale
    relative_scale = torch.exp(block_max - safe_row_max_block)
    same_scale_out_acc = out_acc_block + relative_scale.to(weighted_values.dtype) * weighted_values
    same_scale_normalizer = normalizer_block + relative_scale * block_sum
    same_scale_row_max = row_max_block

    merged_out_acc = torch.where(same_scale, same_scale_out_acc, merged_out_acc)
    merged_normalizer = torch.where(same_scale, same_scale_normalizer, merged_normalizer)
    merged_row_max = torch.where(same_scale, same_scale_row_max, merged_row_max)
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
    scale_log2 = 1.0 / math.log(2.0)
    rescale_threshold = _fa4_rescale_threshold(q.dtype)
    # Each scheduled wave now owns one query tile and iterates across the K/V
    # tiles for that query tile. That matches the FA4 forward mental model from
    # the paper/blog more closely: load a Q tile, then loop over K/V blocks.
    #
    # Each wave is processed in three conceptual roles:
    # 1. main score production
    # 2. softmax-statistics update
    # 3. correction / rescaling merge
    #
    # Real FA4 couples these roles to Blackwell-era hardware features such as
    # TMEM, async MMA, and multi-role warpgroups. We keep the role split and the
    # scheduling metadata, but intentionally do not simulate those primitives.
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
            # Main role: with one Q tile resident for this wave, step through the
            # K/V tiles and produce the corresponding score tiles.
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
            # Softmax role: update the per-row statistics and build the local
            # weighted-value contribution from the score tile.
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                v[:, :, task.k_slice, :],
            )
            softmax_outputs.append((task, block_max, block_sum, weighted_values))

        for task, block_max, block_sum, weighted_values in softmax_outputs:
            # Correction role: merge the new tile's contribution into the running
            # output state and record whether a rescale was conceptually needed.
            out_acc_blocks[task.query_tile], normalizer_blocks[task.query_tile], row_max_blocks[task.query_tile], rescaled = (
                _correction_merge(
                    out_acc_block=out_acc_blocks[task.query_tile],
                    normalizer_block=normalizer_blocks[task.query_tile],
                    row_max_block=row_max_blocks[task.query_tile],
                    block_max=block_max,
                    block_sum=block_sum,
                    weighted_values=weighted_values,
                    scale_log2=scale_log2,
                    rescale_threshold=rescale_threshold,
                )
            )
            scheduler_trace.append(
                {
                    "wave_id": task.wave_id,
                    "query_tile": task.query_tile,
                    "key_tile": task.key_tile,
                    "rescaled": bool(rescaled.any().item()),
                    "rescale_threshold": rescale_threshold,
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
            # Backward follows the same Q-major wave ordering so the educational
            # implementation stays aligned with the forward scheduling story.
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
            # The derivative formulas remain exact attention derivatives. The FA4
            # distinction here is the role/schedule decomposition, not new math.
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
