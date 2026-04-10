import argparse
import time

import torch
import triton.testing

from flash_attention_core import get_version_module, reference_attention
from flash_attention_core.script_utils import (
    add_common_arguments,
    choose_device,
    config_from_args,
    random_inputs,
)


def benchmark(fn, *, warmup: int, rep: int) -> float:
    if torch.cuda.is_available():
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="mean")

    for _ in range(warmup):
        fn()

    start = time.perf_counter_ns()
    for _ in range(rep):
        fn()
    end = time.perf_counter_ns()
    return ((end - start) / rep) / 1_000_000


def clone_triplet(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return (
        q.detach().clone().requires_grad_(True),
        k.detach().clone().requires_grad_(True),
        v.detach().clone().requires_grad_(True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simplified FlashAttention implementations")
    add_common_arguments(parser, include_type=True, include_profile=True)
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for Triton benchmarking")
    parser.add_argument("--rep", type=int, default=100, help="Measured iterations for Triton benchmarking")
    args = parser.parse_args()

    device = choose_device()
    config = config_from_args(args)
    version = get_version_module(args.version)
    q, k, v, key_padding_mask = random_inputs(args, device=device)

    if args.type == "flash":
        forward_run = lambda: version.attention(
            q,
            k,
            v,
            causal=args.causal,
            key_padding_mask=key_padding_mask,
            config=config,
        )

        def backward_run():
            q_b = q.detach().clone()
            k_b = k.detach().clone()
            v_b = v.detach().clone()
            forward_result = version.forward(
                q_b,
                k_b,
                v_b,
                causal=args.causal,
                key_padding_mask=key_padding_mask,
                config=config,
            )
            version.backward(
                q_b,
                k_b,
                v_b,
                torch.ones_like(forward_result.out),
                forward_result,
                causal=args.causal,
                key_padding_mask=key_padding_mask,
                config=config,
            )
    else:
        forward_run = lambda: reference_attention(
            q,
            k,
            v,
            causal=args.causal,
            key_padding_mask=key_padding_mask,
        )

        def backward_run():
            q_b, k_b, v_b = clone_triplet(q, k, v)
            out = reference_attention(
                q_b,
                k_b,
                v_b,
                causal=args.causal,
                key_padding_mask=key_padding_mask,
            )
            torch.autograd.grad(out.sum(), (q_b, k_b, v_b))

    forward_ms = benchmark(forward_run, warmup=args.warmup, rep=args.rep)
    backward_ms = benchmark(backward_run, warmup=args.warmup, rep=args.rep)
    print(f"type={args.type} version={args.version} causal={args.causal} device={device}")
    print(f"forward_ms={forward_ms:.3f}")
    print(f"backward_ms={backward_ms:.3f}")

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        for pass_name, run in (("forward", forward_run), ("backward", backward_run)):
            with torch.profiler.profile(
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./profiler_logs/bench_log_{pass_name}"),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=True,
                with_modules=False,
            ) as prof:
                run()
            print(f"[{pass_name}]")
            print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


if __name__ == "__main__":
    main()
