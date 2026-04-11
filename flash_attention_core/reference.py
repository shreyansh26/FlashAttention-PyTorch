from __future__ import annotations

import math

import torch

from .masking import build_full_mask, normalize_key_padding_mask


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    key_padding_mask = normalize_key_padding_mask(
        key_padding_mask,
        batch_size=q.shape[0],
        kv_len=k.shape[2],
        device=q.device,
    )

    scores = torch.einsum("bhid,bhjd->bhij", q * (1.0 / math.sqrt(q.shape[-1])), k).to(torch.float32)
    full_mask = build_full_mask(
        batch_size=q.shape[0],
        q_len=q.shape[2],
        kv_len=k.shape[2],
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=q.device,
    )

    if full_mask is not None:
        scores = torch.where(full_mask, scores, torch.full_like(scores, float("-inf")))
        row_has_valid = full_mask.any(dim=-1, keepdim=True)
    else:
        row_has_valid = torch.ones_like(scores[..., :1], dtype=torch.bool)

    row_max = scores.max(dim=-1, keepdim=True).values
    row_max = torch.where(row_has_valid, row_max, torch.zeros_like(row_max))
    exp_scores = torch.exp(scores - row_max)
    if full_mask is not None:
        exp_scores = torch.where(full_mask, exp_scores, torch.zeros_like(exp_scores))
    exp_scores = torch.where(row_has_valid, exp_scores, torch.zeros_like(exp_scores))
    row_sum = exp_scores.sum(dim=-1, keepdim=True)
    attention = torch.where(
        row_sum > 0,
        exp_scores / row_sum,
        torch.zeros_like(exp_scores),
    ).to(v.dtype)
    return attention @ v
