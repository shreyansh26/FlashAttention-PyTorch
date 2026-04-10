import argparse
import time

import torch

from flash_attention_core import get_version_module, reference_attention
from flash_attention_core.script_utils import (
    add_common_arguments,
    choose_device,
    config_from_args,
    random_inputs,
)


def benchmark(fn, *, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter_ns()
    return (end - start) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simplified FlashAttention implementations")
    add_common_arguments(parser, include_type=True, include_profile=True)
    args = parser.parse_args()

    device = choose_device()
    config = config_from_args(args)
    version = get_version_module(args.version)
    q, k, v, key_padding_mask = random_inputs(args, device=device)

    if args.type == "flash":
        run = lambda: version.attention(
            q,
            k,
            v,
            causal=args.causal,
            key_padding_mask=key_padding_mask,
            config=config,
        )
    else:
        run = lambda: reference_attention(
            q,
            k,
            v,
            causal=args.causal,
            key_padding_mask=key_padding_mask,
        )

    elapsed_ms = benchmark(run)
    print(f"type={args.type} version={args.version} causal={args.causal} device={device}")
    print(f"{elapsed_ms:.3f}ms")

    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs/bench_log"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            with_modules=False,
        ) as prof:
            run()
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


if __name__ == "__main__":
    main()
