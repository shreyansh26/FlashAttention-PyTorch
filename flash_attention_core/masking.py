from __future__ import annotations

import torch


def normalize_key_padding_mask(
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    kv_len: int,
    device: torch.device,
) -> torch.Tensor | None:
    if key_padding_mask is None:
        return None
    if key_padding_mask.shape != (batch_size, kv_len):
        raise ValueError(
            "key_padding_mask must have shape (batch_size, kv_len); "
            f"got {tuple(key_padding_mask.shape)}"
        )
    return key_padding_mask.to(device=device, dtype=torch.bool)


def build_block_mask(
    *,
    batch_size: int,
    q_start: int,
    q_end: int,
    k_start: int,
    k_end: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    block_mask = None

    if key_padding_mask is not None:
        block_mask = key_padding_mask[:, None, None, k_start:k_end]

    if causal:
        q_positions = torch.arange(q_start, q_end, device=device) + (kv_len - q_len)
        k_positions = torch.arange(k_start, k_end, device=device)
        causal_mask = (q_positions[:, None] >= k_positions[None, :])[None, None, :, :]
        block_mask = causal_mask if block_mask is None else block_mask & causal_mask

    if block_mask is None:
        return None
    return block_mask.expand(batch_size, 1, q_end - q_start, k_end - k_start)


def build_full_mask(
    *,
    batch_size: int,
    q_len: int,
    kv_len: int,
    causal: bool,
    key_padding_mask: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    return build_block_mask(
        batch_size=batch_size,
        q_start=0,
        q_end=q_len,
        k_start=0,
        k_end=kv_len,
        q_len=q_len,
        kv_len=kv_len,
        causal=causal,
        key_padding_mask=key_padding_mask,
        device=device,
    )
