from __future__ import annotations

import math
from typing import Iterable

import torch

from .masking import build_block_mask, normalize_key_padding_mask


def iter_block_slices(length: int, block_size: int) -> list[slice]:
    actual_block_size = max(1, min(block_size, length))
    return [
        slice(start, min(start + actual_block_size, length))
        for start in range(0, length, actual_block_size)
    ]


def prepare_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must have shape (batch, heads, seq, dim)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch dimension mismatch between q, k and v")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("head dimension mismatch between q, k and v")
    if k.shape[2] != v.shape[2]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[3] != k.shape[3]:
        raise ValueError("q and k must have the same hidden dimension")
    if v.shape[3] == 0:
        raise ValueError("v hidden dimension must be non-zero")

    normalized_mask = normalize_key_padding_mask(
        key_padding_mask,
        batch_size=q.shape[0],
        kv_len=k.shape[2],
        device=q.device,
    )
    if causal and normalized_mask is not None and normalized_mask.dtype != torch.bool:
        normalized_mask = normalized_mask.bool()
    return q, k, normalized_mask


def zeros_row_stats(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    row_shape = q.shape[:-1] + (1,)
    normalizers = torch.zeros(row_shape, device=q.device, dtype=torch.float32)
    row_max = torch.full(row_shape, float("-inf"), device=q.device, dtype=torch.float32)
    return normalizers, row_max


def scale_queries(q_block: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    return q_block * (1.0 / math.sqrt(hidden_dim))


def compute_scores(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("bhid,bhjd->bhij", q_block, k_block)


def block_scores_and_mask(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q_slice: slice,
    k_slice: slice,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    q_block = scale_queries(q[:, :, q_slice, :], q.shape[-1])
    k_block = k[:, :, k_slice, :]
    scores = compute_scores(q_block, k_block).to(torch.float32)
    mask = build_block_mask(
        batch_size=q.shape[0],
        q_start=q_slice.start or 0,
        q_end=q_slice.stop or q.shape[2],
        k_start=k_slice.start or 0,
        k_end=k_slice.stop or k.shape[2],
        q_len=q.shape[2],
        kv_len=k.shape[2],
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q.device,
    )
    return scores, mask


def compute_local_statistics(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    v_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if valid_mask is not None:
        masked_scores = torch.where(valid_mask, scores, torch.full_like(scores, float("-inf")))
        row_has_valid = valid_mask.any(dim=-1, keepdim=True)
    else:
        masked_scores = scores
        row_has_valid = torch.ones_like(scores[..., :1], dtype=torch.bool)

    block_max = masked_scores.max(dim=-1, keepdim=True).values
    block_max = torch.where(row_has_valid, block_max, torch.zeros_like(block_max))
    exp_scores = torch.exp(masked_scores - block_max)
    exp_scores = torch.where(row_has_valid, exp_scores, torch.zeros_like(exp_scores))

    if valid_mask is not None:
        exp_scores = torch.where(valid_mask, exp_scores, torch.zeros_like(exp_scores))

    block_sum = exp_scores.sum(dim=-1, keepdim=True)
    weighted_values = torch.einsum("bhij,bhjd->bhid", exp_scores.to(v_block.dtype), v_block)
    return block_max, block_sum, weighted_values


def merge_state_normalized(
    out_block: torch.Tensor,
    normalizer_block: torch.Tensor,
    row_max_block: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_row_max = torch.maximum(row_max_block, block_max)
    old_scale = torch.exp(row_max_block - new_row_max)
    new_scale = torch.exp(block_max - new_row_max)
    new_normalizer = old_scale * normalizer_block + new_scale * block_sum

    normalized_weighted_values = torch.where(
        new_normalizer > 0,
        (new_scale / new_normalizer).to(weighted_values.dtype) * weighted_values,
        torch.zeros_like(weighted_values),
    )
    previous_out = torch.where(
        new_normalizer > 0,
        ((old_scale * normalizer_block) / new_normalizer).to(out_block.dtype) * out_block,
        torch.zeros_like(out_block),
    )
    return previous_out + normalized_weighted_values, new_normalizer, new_row_max


def merge_state_unnormalized(
    out_acc_block: torch.Tensor,
    normalizer_block: torch.Tensor,
    row_max_block: torch.Tensor,
    block_max: torch.Tensor,
    block_sum: torch.Tensor,
    weighted_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    new_row_max = torch.maximum(row_max_block, block_max)
    old_scale = torch.exp(row_max_block - new_row_max)
    new_scale = torch.exp(block_max - new_row_max)
    out_acc_block = old_scale.to(out_acc_block.dtype) * out_acc_block + new_scale.to(
        weighted_values.dtype
    ) * weighted_values
    normalizer_block = old_scale * normalizer_block + new_scale * block_sum
    return out_acc_block, normalizer_block, new_row_max


def finalize_unnormalized(
    out_acc: torch.Tensor,
    normalizers: torch.Tensor,
) -> torch.Tensor:
    safe_normalizers = torch.where(normalizers > 0, normalizers, torch.ones_like(normalizers))
    out = out_acc / safe_normalizers.to(out_acc.dtype)
    return torch.where(normalizers > 0, out, torch.zeros_like(out))


def lse_from_state(normalizers: torch.Tensor, row_max: torch.Tensor) -> torch.Tensor:
    safe_normalizers = torch.where(normalizers > 0, normalizers, torch.ones_like(normalizers))
    lse = row_max + torch.log(safe_normalizers)
    return torch.where(normalizers > 0, lse, torch.zeros_like(lse))


def probabilities_from_lse(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    lse_block: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.exp(scores - lse_block)
    if valid_mask is not None:
        probabilities = torch.where(valid_mask, probabilities, torch.zeros_like(probabilities))
    return probabilities


def backward_step(
    *,
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    out_block: torch.Tensor,
    grad_out_block: torch.Tensor,
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
    lse_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = probabilities_from_lse(scores, valid_mask, lse_block).to(grad_out_block.dtype)
    dV = torch.einsum("bhij,bhid->bhjd", probabilities, grad_out_block)
    dP = torch.einsum("bhid,bhjd->bhij", grad_out_block, v_block)
    row_dot = torch.sum(grad_out_block * out_block, dim=-1, keepdim=True)
    dS = probabilities * (dP - row_dot)
    scale = 1.0 / math.sqrt(q_block.shape[-1])
    dQ = scale * torch.einsum("bhij,bhjd->bhid", dS, k_block)
    dK = scale * torch.einsum("bhij,bhid->bhjd", dS, q_block)
    return dQ, dK, dV


def clone_debug_list(debug_enabled: bool) -> list[dict[str, int | bool | float]]:
    return [] if debug_enabled else []
