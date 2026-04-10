from __future__ import annotations

import argparse

import torch

from .config import FlashAttentionConfig


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_type: bool = False,
    include_profile: bool = False,
    include_dump_state: bool = False,
) -> None:
    parser.add_argument("--version", type=str, default="fa1", help="fa1/fa2/fa3/fa4")
    parser.add_argument("--causal", action="store_true", help="Use causal masking")
    parser.add_argument("--b", type=int, default=1, help="Batch size")
    parser.add_argument("--h", type=int, default=2, help="Number of heads")
    parser.add_argument("--q_len", type=int, default=512, help="Query sequence length")
    parser.add_argument("--kv_len", type=int, default=512, help="Key/value sequence length")
    parser.add_argument("--d", type=int, default=64, help="Head dimension")
    parser.add_argument("--block-size", type=int, default=128, help="Tile size used by the simplified kernels")
    parser.add_argument("--num-stages", type=int, default=2, help="Logical pipeline stages for FA3/FA4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    if include_type:
        parser.add_argument("--type", type=str, default="flash", help="flash/normal")
    if include_profile:
        parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    if include_dump_state:
        parser.add_argument("--dump-state", action="store_true", help="Print saved educational state")


def config_from_args(args: argparse.Namespace) -> FlashAttentionConfig:
    return FlashAttentionConfig(
        block_size_q=args.block_size,
        block_size_kv=args.block_size,
        num_stages=args.num_stages,
        keep_debug_state=True,
    )


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_inputs(args: argparse.Namespace, *, device: torch.device):
    torch.manual_seed(args.seed)
    q = torch.randn(args.b, args.h, args.q_len, args.d, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(args.b, args.h, args.kv_len, args.d, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(args.b, args.h, args.kv_len, args.d, device=device, dtype=torch.float32, requires_grad=True)

    key_padding_mask = None
    if not args.causal:
        key_padding_mask = (torch.rand(args.b, args.kv_len, device=device) > 0.2)
        key_padding_mask[:, 0] = True

    return q, k, v, key_padding_mask
