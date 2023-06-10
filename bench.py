import time
import torch
from flash_attention import flash_attention, normal_attention
import argparse

# flash_attention = torch.compile(flash_attention)
# normal_attention = torch.compile(normal_attention)

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True, help="flash/normal")
parser.add_argument('--q_len', type=int, required=False, default=4096, help="Length/first dimension of Q matrix")
parser.add_argument('--kv_len', type=int, required=False, default=4096, help="Length/first dimension of K/V matrix")
parser.add_argument('--b', type=int, required=False, default=2, help="Batch size")
parser.add_argument('--profile', action='store_true', help="For Pytorch profiling")

args = parser.parse_args()

Q = torch.randn(1, args.b, args.q_len, 512).to(device='cuda')
K = torch.randn(1, args.b, args.kv_len, 512).to(device='cuda')
V = torch.randn(1, args.b, args.kv_len, 512).to(device='cuda')
mask = torch.randint(0, 2, (args.b, args.kv_len)).to(device='cuda')

if args.type == "flash":
    start = time.time_ns()
    flash_attention(Q, K, V, mask)
    end = time.time_ns()

    t = (end - start) / 1000000
    print(f'{t}ms')
else:
    start = time.time_ns()
    normal_attention(Q, K, V, mask)
    end = time.time_ns()

    t = (end - start) / 1000000
    print(f'{t}ms')

if args.profile:
    if args.type == "flash":
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs/bench_log_flash'),
            record_shapes=True,
            profile_memory=True,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:
            flash_attention(Q, K, V, mask)
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    else:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs/bench_log_normal'),
            record_shapes=True,
            profile_memory=True,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
        ) as prof:
            normal_attention(Q, K, V, mask)
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))