"""Regression tests for the device-agnostic inference paths (mocked, plus one XPU parity test)."""

import os
import unittest
from unittest import mock

import torch

# `import prismatic` reads this variable at import time (dataset configs), the tests never touch data.
os.environ.setdefault("PRISMATIC_DATA_ROOT", os.devnull)

from prismatic.util import device as dev  # noqa: E402


def fake_host(xpu: bool = False, cuda: bool = False):
    """Patch backend availability so detection can be tested on any machine."""
    return mock.patch.object(dev, "is_available", side_effect=lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu"))


class TestDetection(unittest.TestCase):
    def test_xpu_beats_cuda(self):
        with fake_host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device(), torch.device("xpu", 0))

    def test_cuda_default_is_unchanged(self):
        with fake_host(cuda=True):
            self.assertEqual(dev.default_device(), torch.device("cuda:0"))

    def test_cpu_fallback(self):
        with fake_host():
            self.assertEqual(dev.default_device(), torch.device("cpu"))
            self.assertEqual(dev.device_type(None), "cpu")

    def test_explicit_device_wins(self):
        with fake_host(xpu=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.device_type(torch.device("cpu")), "cpu")


class TestAttention(unittest.TestCase):
    def test_cuda_keeps_flash_attention(self):
        self.assertEqual(dev.attn_implementation("cuda:0"), "flash_attention_2")

    def test_other_backends_fall_back_to_sdpa(self):
        self.assertEqual(dev.attn_implementation("xpu:0"), "sdpa")
        self.assertEqual(dev.attn_implementation("cpu"), "sdpa")

    def test_other_requests_pass_through(self):
        self.assertEqual(dev.attn_implementation("xpu:0", "eager"), "eager")
        self.assertIsNone(dev.attn_implementation("xpu:0", None))

    def test_fallback_logged_once(self):
        with mock.patch.object(dev, "_FALLBACK_LOGGED", False), self.assertLogs(dev._LOGGER, "WARNING") as logs:
            dev.attn_implementation("xpu:0")
            dev.attn_implementation("xpu:0")
        self.assertEqual(len(logs.records), 1)


class TestSeeding(unittest.TestCase):
    def test_seeds_only_available_backends(self):
        fake_xpu = mock.MagicMock()
        with fake_host(xpu=True), mock.patch.object(torch, "xpu", fake_xpu, create=True):
            with mock.patch.object(torch.cuda, "manual_seed_all") as cuda_seed:
                dev.manual_seed_all(7)
        fake_xpu.manual_seed_all.assert_called_once_with(7)
        cuda_seed.assert_not_called()

    def test_no_accelerator_is_a_noop(self):
        with fake_host():
            dev.manual_seed_all(7)


class TestAutocastSite(unittest.TestCase):
    def test_autocast_site_uses_model_device(self):
        """`OpenVLA.predict_action` and `PrismaticVLM.generate` hand `device_type(self.device)` to autocast."""
        self.assertEqual(dev.device_type(torch.device("xpu", 0)), "xpu")
        self.assertEqual(dev.device_type("cuda"), "cuda")


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an Intel XPU")
class TestXpuParity(unittest.TestCase):
    def test_tiny_llama_logits_match_cpu(self):
        """Random-weight Llama in bf16 with sdpa attention: CPU and XPU logits agree."""
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4
        )
        model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
        impl = dev.attn_implementation("xpu:0")
        self.assertEqual(impl, "sdpa")
        input_ids = torch.randint(0, 128, (1, 16))
        with torch.no_grad():
            expected = model(input_ids).logits.float()
            actual = model.to("xpu:0")(input_ids.to("xpu:0")).logits.float().cpu()
        self.assertTrue(torch.allclose(expected, actual, atol=5e-2, rtol=5e-2))
        self.assertEqual(expected.argmax(-1).eq(actual.argmax(-1)).float().mean().item() > 0.9, True)


if __name__ == "__main__":
    unittest.main()
