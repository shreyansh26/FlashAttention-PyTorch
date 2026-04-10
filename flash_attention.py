import argparse
import pprint

import torch

from flash_attention_core import get_version_module, reference_attention
from flash_attention_core.script_utils import (
    add_common_arguments,
    choose_device,
    config_from_args,
    random_inputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simplified FlashAttention forward pass")
    add_common_arguments(parser, include_dump_state=True)
    args = parser.parse_args()

    device = choose_device()
    config = config_from_args(args)
    version = get_version_module(args.version)
    q, k, v, key_padding_mask = random_inputs(args, device=device)

    forward_result = version.forward(
        q,
        k,
        v,
        causal=args.causal,
        key_padding_mask=key_padding_mask,
        config=config,
    )
    reference_out = reference_attention(
        q,
        k,
        v,
        causal=args.causal,
        key_padding_mask=key_padding_mask,
    )

    max_abs_diff = (forward_result.out - reference_out).abs().max().item()
    print(f"device={device} version={args.version} causal={args.causal}")
    print(f"forward_close={torch.allclose(forward_result.out, reference_out, atol=1e-5, rtol=1e-4)}")
    print(f"max_abs_diff={max_abs_diff:.6e}")

    if args.dump_state:
        pprint.pprint(forward_result.saved_state)


if __name__ == "__main__":
    main()
