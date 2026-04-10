# FlashAttention in PyTorch

This repository now contains simplified educational implementations of FlashAttention versions 1 through 4. The goal is correctness and clarity, not CUDA-level performance. Each version keeps the same exact attention math while changing the orchestration so the algorithmic differences are visible in plain PyTorch.

## Requirements
* torch==2.0.1

## Layout
* [flash_attention_core](flash_attention_core) - Shared package with reference attention, masking helpers, config/types, and versioned implementations.
* [flash_attention.py](flash_attention.py) - Unified forward demo for `fa1` through `fa4`.
* [bench.py](bench.py) - Unified benchmark entry point with `--version` and `--causal`.
* [check_backward.py](check_backward.py) - Unified forward and backward correctness check.
* [tests](tests) - Small regression suite covering all versions.

## Supported Modes
* Non-causal attention with an optional key-padding mask of shape `(batch, kv_len)`.
* Causal attention via `--causal`.

## Versions
* `fa1` - Baseline tiled online-softmax FlashAttention.
* `fa2` - Sequence-parallel / split-Q ownership with deferred normalization and LSE-centered state.
* `fa3` - Explicit staged pipeline with ping-pong tile buffers.
* `fa4` - Explicit scheduler, main/softmax/correction phases, and conditional rescaling.

Where the simplified code leaves out real CUDA behavior such as TMA, WGMMA, TMEM, FP8 paths, or multi-CTA coordination, the version modules call that out in comments.

## Usage

### Forward Demo

```bash
python flash_attention.py
python flash_attention.py --version fa3 --causal --dump-state
```

### Benchmark

```bash
python bench.py --type flash --version fa2 --b 1 --h 2 --q_len 4096 --kv_len 4096 --d 128
python bench.py --type normal --causal --b 1 --h 2 --q_len 4096 --kv_len 4096 --d 128
```

`bench.py` uses Triton's benchmark helper when running on CUDA and reports both
`forward_ms` and `backward_ms`. For `--type flash`, the backward timing measures
the simplified manual backward implementation. For `--type normal`, the
backward timing measures PyTorch autograd on the reference attention path.

Add `--profile` to capture separate PyTorch profiler traces for the forward and
backward benchmark paths.

### Forward and Backward Correctness

```bash
python check_backward.py
python check_backward.py --version fa4 --causal --q_len 256 --kv_len 256 --d 64
```

### Tests

```bash
python -m unittest discover -s tests
python -m unittest tests.test_flash_attention_long
```

`python -m unittest discover -s tests` tells Python's built-in `unittest` runner to scan the [tests](tests) directory, find test files automatically, and run everything it discovers there. That is the command to use for the regular test suite.

`python -m unittest tests.test_flash_attention_long` runs one specific test module directly. The long-sequence test is called out separately because it is heavier than the rest of the suite and is mainly useful when you explicitly want the `kv_len=8192` coverage on GPU.
