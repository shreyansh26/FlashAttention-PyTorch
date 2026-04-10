import unittest

import torch

from flash_attention_core import FlashAttentionConfig, get_version_module, reference_attention
from flash_attention_core.versions.fa4 import _correction_merge, _fa4_rescale_threshold


class FlashAttentionVersionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = FlashAttentionConfig(block_size_q=16, block_size_kv=16, num_stages=2)
        self.versions = ("fa1", "fa2", "fa3", "fa4")

    def _inputs(self, *, causal: bool):
        torch.manual_seed(0)
        q = torch.randn(1, 8, 24, 64, dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 8, 24, 64, dtype=torch.float32, requires_grad=True)
        v = torch.randn(1, 8, 24, 64, dtype=torch.float32, requires_grad=True)
        key_padding_mask = None
        if not causal:
            key_padding_mask = torch.tensor(
                [[1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]],
                dtype=torch.bool,
            )
        return q, k, v, key_padding_mask

    def _clone_triplet(self, q, k, v):
        return (
            q.detach().clone().requires_grad_(True),
            k.detach().clone().requires_grad_(True),
            v.detach().clone().requires_grad_(True),
        )

    def test_forward_matches_reference(self) -> None:
        for version_name in self.versions:
            version = get_version_module(version_name)
            for causal in (False, True):
                q, k, v, key_padding_mask = self._inputs(causal=causal)
                out = version.attention(
                    q,
                    k,
                    v,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                    config=self.config,
                )
                reference_out = reference_attention(
                    q,
                    k,
                    v,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                )
                self.assertTrue(
                    torch.allclose(out, reference_out, atol=1e-5, rtol=1e-4),
                    msg=f"{version_name} forward mismatch for causal={causal}",
                )

    def test_manual_backward_matches_reference_gradients(self) -> None:
        for version_name in self.versions:
            version = get_version_module(version_name)
            for causal in (False, True):
                q, k, v, key_padding_mask = self._inputs(causal=causal)

                q_flash, k_flash, v_flash = self._clone_triplet(q, k, v)
                flash_out = version.attention(
                    q_flash,
                    k_flash,
                    v_flash,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                    config=self.config,
                )
                flash_grads = torch.autograd.grad(flash_out.sum(), (q_flash, k_flash, v_flash))

                q_ref, k_ref, v_ref = self._clone_triplet(q, k, v)
                ref_out = reference_attention(
                    q_ref,
                    k_ref,
                    v_ref,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                )
                ref_grads = torch.autograd.grad(ref_out.sum(), (q_ref, k_ref, v_ref))

                forward_result = version.forward(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                    config=self.config,
                )
                manual = version.backward(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    torch.ones_like(forward_result.out),
                    forward_result,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                    config=self.config,
                )

                for computed, expected, grad_name in (
                    (flash_grads[0], ref_grads[0], "dQ"),
                    (flash_grads[1], ref_grads[1], "dK"),
                    (flash_grads[2], ref_grads[2], "dV"),
                    (manual.dQ, ref_grads[0], "manual dQ"),
                    (manual.dK, ref_grads[1], "manual dK"),
                    (manual.dV, ref_grads[2], "manual dV"),
                ):
                    self.assertTrue(
                        torch.allclose(computed, expected, atol=1e-5, rtol=1e-4),
                        msg=f"{version_name} {grad_name} mismatch for causal={causal}",
                    )

    def test_fa4_thresholded_selective_rescaling_rule(self) -> None:
        self.assertEqual(_fa4_rescale_threshold(torch.float16), 8.0)
        self.assertEqual(_fa4_rescale_threshold(torch.bfloat16), 8.0)
        self.assertEqual(_fa4_rescale_threshold(torch.float32), 0.0)

        out_acc = torch.tensor([[[[2.0, 4.0]]]], dtype=torch.float32)
        normalizer = torch.tensor([[[[3.0]]]], dtype=torch.float32)
        row_max = torch.tensor([[[[10.0]]]], dtype=torch.float32)
        block_sum = torch.tensor([[[[5.0]]]], dtype=torch.float32)
        weighted_values = torch.tensor([[[[7.0, 11.0]]]], dtype=torch.float32)

        # Small enough max increase: official FA4 keeps the old row max and skips
        # the full rescale when the exponent-domain delta stays above -threshold.
        merged = _correction_merge(
            out_acc_block=out_acc,
            normalizer_block=normalizer,
            row_max_block=row_max,
            block_max=torch.tensor([[[[11.0]]]], dtype=torch.float32),
            block_sum=block_sum,
            weighted_values=weighted_values,
            scale_log2=1.0 / torch.log(torch.tensor(2.0)).item(),
            rescale_threshold=8.0,
        )
        self.assertFalse(bool(merged[3].any().item()))
        self.assertTrue(torch.allclose(merged[2], row_max))

        # Large enough max increase: the threshold is exceeded and the merge
        # falls back to the fully rescaled path.
        merged = _correction_merge(
            out_acc_block=out_acc,
            normalizer_block=normalizer,
            row_max_block=row_max,
            block_max=torch.tensor([[[[20.0]]]], dtype=torch.float32),
            block_sum=block_sum,
            weighted_values=weighted_values,
            scale_log2=1.0 / torch.log(torch.tensor(2.0)).item(),
            rescale_threshold=8.0,
        )
        self.assertTrue(bool(merged[3].any().item()))
        self.assertTrue(torch.all(merged[2] >= row_max))


if __name__ == "__main__":
    unittest.main()
