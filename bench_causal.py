import time
import torch
from flash_attention_causal import flash_attention_causal, normal_attention_causal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True, help="flash/normal")
parser.add_argument('--b', type=int, required=False, default=1, help="Batch size")
parser.add_argument('--h', type=int, required=False, default=2, help="Number of heads")
parser.add_argument('--q_len', type=int, required=False, default=4096, help="Length/first dimension of Q matrix")
parser.add_argument('--kv_len', type=int, required=False, default=4096, help="Length/first dimension of K/V matrix")
parser.add_argument('--d', type=int, required=False, default=512, help="Dimension of vector")
parser.add_argument('--profile', action='store_true', help="For Pytorch profiling")

args = parser.parse_args()

Q = torch.randn(args.b, args.h, args.q_len, args.d, requires_grad=True).to(device='cuda')
K = torch.randn(args.b, args.h, args.kv_len, args.d, requires_grad=True).to(device='cuda')
V = torch.randn(args.b, args.h, args.kv_len, args.d, requires_grad=True).to(device='cuda')

if args.type == "flash":
    for _ in range(10):
        flash_attention_causal(Q, K, V)
        
    start = time.time_ns()
    flash_attention_causal(Q, K, V)
    end = time.time_ns()

    t = (end - start) / 1000000
    print(f'{t}ms')
else:
    for _ in range(10):
        normal_attention_causal(Q, K, V)
        
    start = time.time_ns()
    normal_attention_causal(Q, K, V)
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
            flash_attention_causal(Q, K, V)
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
            normal_attention_causal(Q, K, V)
        print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
