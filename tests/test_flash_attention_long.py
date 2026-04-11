import unittest

import torch

from flash_attention_core import FlashAttentionConfig, get_version_module, reference_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA long tests require a GPU")
class FlashAttentionLongTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda")
        self.config = FlashAttentionConfig(block_size_q=128, block_size_kv=128, num_stages=2)
        self.versions = ("fa1", "fa2", "fa3", "fa4")

    def _inputs(self, *, causal: bool):
        torch.manual_seed(7)
        q = torch.randn(1, 8, 64, 64, device=self.device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(1, 8, 8192, 64, device=self.device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(1, 8, 8192, 64, device=self.device, dtype=torch.float32, requires_grad=True)

        key_padding_mask = None
        if not causal:
            key_padding_mask = (torch.rand(1, 8192, device=self.device) > 0.1)
            key_padding_mask[:, 0] = True

        return q, k, v, key_padding_mask

    def _clone_triplet(self, q, k, v):
        return (
            q.detach().clone().requires_grad_(True),
            k.detach().clone().requires_grad_(True),
            v.detach().clone().requires_grad_(True),
        )

    def test_long_kv_forward_and_backward(self) -> None:
        for version_name in self.versions:
            version = get_version_module(version_name)
            for causal in (False, True):
                q, k, v, key_padding_mask = self._inputs(causal=causal)

                forward_result = version.forward(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                    config=self.config,
                )
                reference_out = reference_attention(
                    q.detach(),
                    k.detach(),
                    v.detach(),
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                )

                self.assertEqual(forward_result.out.shape, q.shape)
                self.assertTrue(torch.isfinite(forward_result.out).all(), msg=f"{version_name} produced non-finite forward output")
                self.assertTrue(
                    torch.allclose(forward_result.out, reference_out, atol=1e-5, rtol=1e-4),
                    msg=f"{version_name} long-sequence forward mismatch for causal={causal}",
                )

                q_auto, k_auto, v_auto = self._clone_triplet(q, k, v)
                auto_out = version.attention(
                    q_auto,
                    k_auto,
                    v_auto,
                    causal=causal,
                    key_padding_mask=key_padding_mask,
                    config=self.config,
                )
                auto_grads = torch.autograd.grad(auto_out.sum(), (q_auto, k_auto, v_auto))
                for grad, grad_name in zip(auto_grads, ("dQ", "dK", "dV")):
                    self.assertEqual(grad.shape, (q_auto.shape if grad_name == "dQ" else k_auto.shape if grad_name == "dK" else v_auto.shape))
                    self.assertTrue(torch.isfinite(grad).all(), msg=f"{version_name} {grad_name} autograd long-test run produced non-finite values")

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
                self.assertEqual(manual.dQ.shape, q.shape)
                self.assertEqual(manual.dK.shape, k.shape)
                self.assertEqual(manual.dV.shape, v.shape)
                self.assertTrue(torch.isfinite(manual.dQ).all(), msg=f"{version_name} manual dQ produced non-finite values")
                self.assertTrue(torch.isfinite(manual.dK).all(), msg=f"{version_name} manual dK produced non-finite values")
                self.assertTrue(torch.isfinite(manual.dV).all(), msg=f"{version_name} manual dV produced non-finite values")

    def test_fa3_fp8_long_forward(self) -> None:
        version = get_version_module("fa3")
        q, k, v, key_padding_mask = self._inputs(causal=False)
        fp8_config = FlashAttentionConfig(block_size_q=128, block_size_kv=128, num_stages=2, fp8=True)

        forward_result = version.forward(
            q.detach(),
            k.detach(),
            v.detach(),
            causal=False,
            key_padding_mask=key_padding_mask,
            config=fp8_config,
        )
        reference_out = reference_attention(
            q.detach(),
            k.detach(),
            v.detach(),
            causal=False,
            key_padding_mask=key_padding_mask,
        )

        self.assertEqual(forward_result.out.shape, q.shape)
        self.assertTrue(torch.isfinite(forward_result.out).all())
        self.assertTrue(torch.allclose(forward_result.out, reference_out, atol=2e-1, rtol=2e-1))
        self.assertTrue(forward_result.saved_state["fp8_enabled"])


if __name__ == "__main__":
    unittest.main()
