# FlashAttention in PyTorch

A simplified implementation of FlashAttention in PyTorch. I have implemented the forward pass and backward pass algorithms from the paper, and also shown that it is equivalent to the normal attention formulation in Transformers. I also include some code for benchmarking. 

Note that this is for educational purposes only as I haven't implemented any of the CUDA and SRAM memory tricks as described in the paper.

## Requirements
* einops==0.6.1
* torch==2.0.1

## Files
* [flash_attention.py](flash_attention.py) - Implementation of the general formulation of FlashAttention which takes in Q, K, V and a mask. The code includes both the forward and backward algorithms and a simple test of equivalence of the forward pass with normal attention as well.
* [flash_attention_causal.py](flash_attention_causal.py) - The causal version of FlashAttention which takes in Q, K and V. The mask is caluclated in a causal fashion which is typcially used in autoregressive models. This code also includes the forward and backward algorithms and a simple test of equivalence of the forward pass with normal attention (causal) as well.
* [bench.py](bench.py), [bench_causal.py](bench_causal.py) - Benchmarking code for both general and causal versions of FlashAttention.
* [check_backward.py](check_backward.py), [check_backward_causal.py](check_backward_causal.py) - This script verifies two things - 1. whether the calculated value of gradients (using PyTorch's `jacrev`) of Q, K and V match for the normal version of attention and FlashAttention, and 2. whether these results match the implementation of backward pass given in the paper. The loss function is simply assumed to be a sum of the final output tensor. 

## To run

### Forward pass

**Causal mask**     
```python flash_attention_causal.py```

**Random mask**    
```python flash_attention.py```

### Benchmarking - Causal mask

**FlashAttention**    
```python bench_causal.py --b 1 --h 2 --q_len 16384 --kv_len 16384 --d 512 --type flash```

**Normal attention**    
```python bench_causal.py --b 1 --h 2 --q_len 16384 --kv_len 16384 --d 512 --type normal```

Add `--profile` to log additional details using PyTorch Profiler.

### Benchmarking - Random mask

**FlashAttention**    
```python bench.py --b 1 --h 2 --q_len 16384 --kv_len 16384 --d 512 --type flash```

**Normal attention**    
```python bench.py --b 1 --h 2 --q_len 16384 --kv_len 16384 --d 512 --type normal```

Add `--profile` to log additional details using PyTorch Profiler.

### Backward Pass

**Causal mask**     
```python check_backward_causal.py```

**Random mask**    
```python check_backward.py```
