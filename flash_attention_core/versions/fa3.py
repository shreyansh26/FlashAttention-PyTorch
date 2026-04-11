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
5. In `fp8` mode, simulate FA3's block-quantized FP8 forward path with per-tile
   scales and dequantize-on-use computation.

Compared with FA2, the educational improvement is pipeline structure. Real FA3
uses Hopper features such as TMA, WGMMA, warp specialization, and ping-pong
buffering to overlap data movement and compute. This module leaves those
hardware mechanisms out, but keeps the stage boundaries and double-buffered
control flow visible in plain PyTorch. The released official FA3 path supports
FP8 forward, but not FP8 backward, and this simplified implementation follows
that same support boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from flash_attention_core.common import (
    backward_step,
    build_block_mask,
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
    fp8_meta: dict[str, float] | None = None


FP8_E4M3_MAX = 448.0


def _simulate_fp8_block(tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    # FA3's published FP8 path uses true float8 formats plus block quantization.
    # PyTorch portability is better if we simulate that with a per-block scale and
    # a dequantize-on-use path rather than requiring hardware float8 kernels.
    amax = tensor.abs().max().item()
    scale = max(amax / FP8_E4M3_MAX, 1e-8)
    quantized = torch.clamp(torch.round(tensor / scale), -FP8_E4M3_MAX, FP8_E4M3_MAX)
    dequantized = quantized * scale
    return dequantized, {"amax": amax, "scale": scale, "format_max": FP8_E4M3_MAX}


def _compute_scores_from_blocks(
    *,
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    q_slice: slice,
    k_slice: slice,
    q_len: int,
    kv_len: int,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    scaled_q = q_block * (1.0 / math.sqrt(q_block.shape[-1]))
    scores = torch.einsum("bhid,bhjd->bhij", scaled_q, k_block).to(torch.float32)
    valid_mask = build_block_mask(
        batch_size=q_block.shape[0],
        q_start=q_slice.start or 0,
        q_end=q_slice.stop or q_len,
        k_start=k_slice.start or 0,
        k_end=k_slice.stop or kv_len,
        q_len=q_len,
        kv_len=kv_len,
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q_block.device,
    )
    return scores, valid_mask


def _load_tile(
    *,
    buffer_id: int,
    tile_id: int,
    tile_slice: slice,
    k: torch.Tensor,
    v: torch.Tensor,
    fp8: bool,
) -> TileBuffer:
    # In real FA3 this "load" would be backed by TMA into shared memory and then
    # consumed by WGMMA-based compute warpgroups. Here it is just a lightweight
    # Python object so the pipeline structure is explicit in the code.
    k_block = k[:, :, tile_slice, :]
    v_block = v[:, :, tile_slice, :]
    fp8_meta = None
    if fp8:
        # This is a simplified depiction of FA3's block-quantized FP8 path.
        # The official implementation uses hardware FP8 fragments; we model the
        # same idea with per-tile quantize/dequantize metadata.
        k_block, k_meta = _simulate_fp8_block(k_block)
        v_block, v_meta = _simulate_fp8_block(v_block)
        fp8_meta = {"k_scale": k_meta["scale"], "k_amax": k_meta["amax"], "v_scale": v_meta["scale"], "v_amax": v_meta["amax"]}
    return TileBuffer(
        buffer_id=buffer_id,
        tile_id=tile_id,
        tile_slice=tile_slice,
        k_block=k_block,
        v_block=v_block,
        fp8_meta=fp8_meta,
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
    # The buffers stand in for ping-pong shared-memory stages. FA3's practical
    # gain comes from overlapping movement and compute; we preserve that mental
    # model without introducing actual asynchronous execution here.
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

        q_block = q[:, :, q_slice, :]
        q_fp8_meta = None
        if config.fp8:
            # We simulate FA3's block-quantized FP8 forward path by quantizing
            # each resident Q tile independently, which is also how we expose the
            # "incoherent processing" idea: every tile carries its own scale.
            q_block, q_fp8_meta = _simulate_fp8_block(q_block)

        active_buffer = _load_tile(
            buffer_id=0,
            tile_id=0,
            tile_slice=k_slices[0],
            k=k,
            v=v,
            fp8=config.fp8,
        )

        for kv_tile_id, _ in enumerate(k_slices):
            next_buffer = None
            if kv_tile_id + 1 < len(k_slices):
                # Prefetch the next logical stage into the alternate buffer.
                next_buffer = _load_tile(
                    buffer_id=(active_buffer.buffer_id + 1) % max(config.num_stages, 2),
                    tile_id=kv_tile_id + 1,
                    tile_slice=k_slices[kv_tile_id + 1],
                    k=k,
                    v=v,
                    fp8=config.fp8,
                )

            # Real FA3 overlaps producer and consumer warpgroups here. We keep the
            # stages explicit, but execute them sequentially for clarity.
            # Stage 1: consume the active K/V tile to build the score block.
            scores, valid_mask = _compute_scores_from_blocks(
                q_block=q_block,
                k_block=active_buffer.k_block,
                q_slice=q_slice,
                k_slice=active_buffer.tile_slice,
                q_len=q.shape[2],
                kv_len=k.shape[2],
                causal=causal,
                key_padding_mask=key_padding_mask,
            )
            # Stage 2: update local softmax statistics and form the unnormalized
            # tile contribution for P @ V.
            block_max, block_sum, weighted_values = compute_local_statistics(
                scores,
                valid_mask,
                active_buffer.v_block,
            )
            # Stage 3: merge the contribution into the running query-tile state.
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
                    "fp8": config.fp8,
                    "q_scale": q_fp8_meta["scale"] if q_fp8_meta is not None else None,
                    "k_scale": active_buffer.fp8_meta["k_scale"] if active_buffer.fp8_meta is not None else None,
                    "v_scale": active_buffer.fp8_meta["v_scale"] if active_buffer.fp8_meta is not None else None,
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
        saved_state={
            "pipeline_trace": pipeline_trace,
            "fp8_enabled": config.fp8,
            "fp8_format": "simulated-e4m3",
            "quantization_mode": "per-tile-block",
        }
        if config.keep_debug_state
        else {},
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
    if config.fp8:
        raise ValueError("FA3 FP8 backward is unsupported in this educational repo")
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
        # Backward keeps the same staged interpretation: consume one active tile,
        # optionally prepare the next tile, then fold the local derivatives into
        # the running dQ / dK / dV accumulators.
        dQ_block = torch.zeros_like(q_block)
        active_buffer = _load_tile(
            buffer_id=0,
            tile_id=0,
            tile_slice=k_slices[0],
            k=k,
            v=v,
            fp8=False,
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
                    fp8=False,
                )

            # In the actual Hopper kernels, the backward mainloop is heavily
            # constrained by register pressure and overlap scheduling. We keep the
            # simplified stage order visible rather than reproducing those details.
            scores, valid_mask = _compute_scores_from_blocks(
                q_block=q_block,
                k_block=active_buffer.k_block,
                q_slice=q_slice,
                k_slice=active_buffer.tile_slice,
                q_len=q.shape[2],
                kv_len=k.shape[2],
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
