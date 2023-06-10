import torch
import torch.nn as nn
import numpy as np
import sys
import time
from einops import rearrange

BLOCK_SIZE = 1024
NEG_INF = -1e10 # -infinity
EPSILON = 1e-10

def normal_attention(Q, K, V, mask=None):
    scale = 1 / np.sqrt(Q.shape[-1])
    Q = Q * scale
    QKt = torch.einsum('... i d, ... j d -> ... i j', Q, K)

    key_mask = rearrange(mask, 'b j -> b 1 1 j')
    QKt = torch.where(key_mask > 0, QKt, NEG_INF)

    attn = nn.functional.softmax(QKt, dim=-1)
    return attn @ V

def flash_attention_forward(Q, K, V, mask=None):
    O = torch.zeros_like(Q, requires_grad=True)
    l = torch.zeros(Q.shape[:-1])[...,None]
    m = torch.ones(Q.shape[:-1])[...,None] * NEG_INF

    O = O.to(device='cuda')
    l = l.to(device='cuda')
    m = m.to(device='cuda')

    Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
    KV_BLOCK_SIZE = BLOCK_SIZE

    # Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
    # K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
    # V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
    # mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

    Tr = Q.shape[2] // Q_BLOCK_SIZE
    Tc = K.shape[2] // KV_BLOCK_SIZE

    O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
    # l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
    # m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

    for j in range(Tc):
        j_index = torch.tensor(range(j*KV_BLOCK_SIZE,(j+1)*KV_BLOCK_SIZE)).to(device='cuda')
        Kj = torch.index_select(K, dim=2, index=j_index)
        Vj = torch.index_select(V, dim=2, index=j_index)
        maskj = torch.index_select(mask, dim=1, index=j_index)

        for i in range(Tr):
            i_index = torch.tensor(range(i*Q_BLOCK_SIZE,(i+1)*Q_BLOCK_SIZE)).to(device='cuda')
            Qi = torch.index_select(Q, dim=2, index=i_index)
            Oi = O_BLOCKS[i]
            li = torch.index_select(l, dim=2, index=i_index)
            mi = torch.index_select(m, dim=2, index=i_index)

            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled  = Qi * scale

            S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
            
            # Masking
            maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
            S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

            m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            P_block_ij = torch.exp(S_ij - m_block_ij)
            # Masking
            P_block_ij = torch.where(maskj_temp > 0, P_block_ij, 0.)

            l_block_ij = torch.sum(P_block_ij, dim=-1, keepdims=True) + EPSILON

            P_block_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_block_ij, Vj)

            mi_new = torch.maximum(m_block_ij, mi)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

            O_BLOCKS[i] = (li/li_new) * torch.exp(mi - mi_new) * Oi + (torch.exp(m_block_ij - mi_new) / li_new) * P_block_ij_Vj
            l[:,:,i*Q_BLOCK_SIZE:(i+1)*Q_BLOCK_SIZE,:] = li_new
            m[:,:,i*Q_BLOCK_SIZE:(i+1)*Q_BLOCK_SIZE,:] = mi_new
        
    O = torch.cat(O_BLOCKS, dim=2)
    return O

def flash_attention(Q, K, V, mask=None):
    out = flash_attention_forward(Q, K, V, mask)
    return out

if __name__ == "__main__":
    Q = torch.randn(1, 2, 4096, 1024, requires_grad=True).to(device='cuda')
    K = torch.randn(1, 2, 4096, 1024, requires_grad=True).to(device='cuda')
    V = torch.randn(1, 2, 4096, 1024, requires_grad=True).to(device='cuda')
    mask = torch.randint(0, 2, (1, 4096)).to(device='cuda')

    for i in range(10):
        start1 = time.time_ns()
        out1 = flash_attention(Q, K, V, mask)
        end1 = time.time_ns()

        start2 = time.time_ns()
        out2 = normal_attention(Q, K, V, mask)
        end2 = time.time_ns()

        t1 = (end1 - start1) / 1000000
        t2 = (end2 - start2) / 1000000

        print(f'{t1}ms, {t2}ms')
        print(torch.allclose(out1, out2, atol=1e-5))