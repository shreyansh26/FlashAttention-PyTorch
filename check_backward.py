import argparse

import torch

from flash_attention_core import get_version_module, reference_attention
from flash_attention_core.script_utils import (
    add_common_arguments,
    choose_device,
    config_from_args,
    random_inputs,
    validate_fp8_support,
)


def clone_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check forward and backward correctness")
    add_common_arguments(parser)
    args = parser.parse_args()

    device = choose_device()
    config = config_from_args(args)
    validate_fp8_support(version=args.version, fp8=args.fp8, script_name="check_backward")
    version = get_version_module(args.version)
    q, k, v, key_padding_mask = random_inputs(args, device=device)

    q_flash, k_flash, v_flash = clone_inputs(q, k, v)
    flash_out = version.attention(
        q_flash,
        k_flash,
        v_flash,
        causal=args.causal,
        key_padding_mask=key_padding_mask,
        config=config,
    )
    flash_grads = torch.autograd.grad(flash_out.sum(), (q_flash, k_flash, v_flash))

    q_ref, k_ref, v_ref = clone_inputs(q, k, v)
    ref_out = reference_attention(
        q_ref,
        k_ref,
        v_ref,
        causal=args.causal,
        key_padding_mask=key_padding_mask,
    )
    ref_grads = torch.autograd.grad(ref_out.sum(), (q_ref, k_ref, v_ref))

    forward_result = version.forward(
        q.detach(),
        k.detach(),
        v.detach(),
        causal=args.causal,
        key_padding_mask=key_padding_mask,
        config=config,
    )
    manual = version.backward(
        q.detach(),
        k.detach(),
        v.detach(),
        torch.ones_like(forward_result.out),
        forward_result,
        causal=args.causal,
        key_padding_mask=key_padding_mask,
        config=config,
    )

    print(f"device={device} version={args.version} causal={args.causal}")
    print(f"forward_close={torch.allclose(flash_out, ref_out, atol=1e-5, rtol=1e-4)}")
    print(f"dq_flash_vs_ref={torch.allclose(flash_grads[0], ref_grads[0], atol=1e-5, rtol=1e-4)}")
    print(f"dk_flash_vs_ref={torch.allclose(flash_grads[1], ref_grads[1], atol=1e-5, rtol=1e-4)}")
    print(f"dv_flash_vs_ref={torch.allclose(flash_grads[2], ref_grads[2], atol=1e-5, rtol=1e-4)}")
    print(f"dq_manual_vs_ref={torch.allclose(manual.dQ, ref_grads[0], atol=1e-5, rtol=1e-4)}")
    print(f"dk_manual_vs_ref={torch.allclose(manual.dK, ref_grads[1], atol=1e-5, rtol=1e-4)}")
    print(f"dv_manual_vs_ref={torch.allclose(manual.dV, ref_grads[2], atol=1e-5, rtol=1e-4)}")


if __name__ == "__main__":
    main()
