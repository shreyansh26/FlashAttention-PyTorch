import time
import torch
from flash_attention import flash_attention, normal_attention, flash_attention_backward, flash_attention_forward
from torch.func import jacrev

Q = torch.randn(1, 1, 2048, 512, requires_grad=True).to(device='cuda')
K = torch.randn(1, 1, 2048, 512, requires_grad=True).to(device='cuda')
V = torch.randn(1, 1, 2048, 512, requires_grad=True).to(device='cuda')
mask = torch.randint(0, 2, (1, 2048)).to(device='cuda')

def loss_fn(fn, *args):
    return torch.sum(fn(*args))

args = (Q, K, V, mask)

dq_flash, dk_flash, dv_flash = jacrev(loss_fn, argnums=(1,2,3))(flash_attention, *args)
dq_normal, dk_normal, dv_normal = jacrev(loss_fn, argnums=(1,2,3))(normal_attention, *args)

print(torch.allclose(dq_flash, dq_normal, atol=1e-5))
print(torch.allclose(dk_flash, dk_normal, atol=1e-5))
print(torch.allclose(dv_flash, dv_normal, atol=1e-5))

O, l, m = flash_attention_forward(Q, K, V, mask)
dO = torch.ones_like(O)     # Since "loss" here is the sum of the elements of the output matrix
dq_flash_manual, dk_flash_manual, dv_flash_manual = flash_attention_backward(Q, K, V, mask, O, l, m, dO)

print(torch.allclose(dq_flash, dq_flash_manual, atol=1e-5))
print(torch.allclose(dk_flash, dk_flash_manual, atol=1e-5))
print(torch.allclose(dv_flash, dv_flash_manual, atol=1e-5))
