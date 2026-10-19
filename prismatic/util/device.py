"""Device-agnostic helpers for the XPU / CUDA / CPU backends (inference paths).

Detection order is XPU, then CUDA, then CPU. An explicit device always wins over detection.
FlashAttention-2 is a CUDA-only kernel, so other backends load the model with PyTorch SDPA.
"""

import logging
from typing import Optional, Union

import torch

DeviceLike = Union[str, torch.device, None]
_ACCELERATORS = ("xpu", "cuda")
_LOGGER = logging.getLogger(__name__)
_FALLBACK_LOGGED = False


def is_available(device_type: str) -> bool:
    """Whether `device_type` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: `xpu` > `cuda` > `cpu`."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def default_device(index: int = 0) -> torch.device:
    """Device of the best available backend (`cuda:0` on a CUDA host, as before)."""
    device_type = default_device_type()
    return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, index)


def resolve_device(device: DeviceLike = None) -> torch.device:
    """`None` auto-detects; an explicit value is honored verbatim."""
    if device is None:
        return default_device()
    return device if isinstance(device, torch.device) else torch.device(device)


def device_type(device: DeviceLike = None) -> str:
    """Backend type (`xpu`, `cuda`, `cpu`) of a device, device string, or `None` (detected)."""
    return resolve_device(device).type


def manual_seed_all(seed: int) -> None:
    """Seed every accelerator that is present (a no-op for backends that are unavailable)."""
    for backend in _ACCELERATORS:
        if is_available(backend):
            getattr(torch, backend).manual_seed_all(seed)


def attn_implementation(device: DeviceLike = None, requested: Optional[str] = "flash_attention_2") -> Optional[str]:
    """Attention implementation for `from_pretrained`.

    A request for `flash_attention_2` is kept on CUDA and replaced by `sdpa` on any other backend, because
    the flash-attn package only ships CUDA kernels. The fallback is logged once.
    """
    global _FALLBACK_LOGGED
    if requested != "flash_attention_2" or device_type(device) == "cuda":
        return requested
    if not _FALLBACK_LOGGED:
        _LOGGER.warning("flash_attention_2 is CUDA-only; using sdpa attention on %s.", device_type(device))
        _FALLBACK_LOGGED = True
    return "sdpa"
