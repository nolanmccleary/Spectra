from __future__ import annotations

from typing import Callable, Dict
import torch

__all__ = [
    'rgb_to_grayscale',
    'rgb_to_luma',
    'no_conversion',
    'generate_conversion',
    'inverse_delta',
    'inverse_delta_local',
    'inverse_luma',
    'no_inversion',
    'generate_inversion',
]

EPS = 1e-6

# ────────────────── conversion (RGB → single-channel) ────────────────────

def rgb_to_grayscale(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """Channel-mean grayscale (C,H,W → 1,H,W)."""
    return torch.mean(rgb_tensor, dim=0, keepdim=True)


def rgb_to_luma(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """BT.601 luma."""
    r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.unsqueeze(0)


def no_conversion(tensor: torch.Tensor) -> torch.Tensor:
    """Identity conversion (kept for completeness)."""
    return tensor


_CONVERSION_TABLE: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'grayscale': rgb_to_grayscale,
    'grayscale_local': rgb_to_grayscale,
    'luma': rgb_to_luma,
    'noinvert': no_conversion,
}


def generate_conversion(kind: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the conversion function requested by *kind*."""
    if kind not in _CONVERSION_TABLE:
        raise ValueError(
            f"'{kind}' not in set of valid conversion handles: {_CONVERSION_TABLE.keys()}"
        )
    return _CONVERSION_TABLE[kind]


def inverse_delta(tensor: torch.Tensor, delta: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Global-mean luminance inversion used by original code."""
    C, H, W = tensor.shape
    if delta.shape == (C, H, W):
        return delta
    rgb_mean = tensor.mean()
    gd = delta.unsqueeze(0)
    result = torch.where(
        gd <= 0,
        gd * tensor / (rgb_mean + eps),
        gd * (1 - tensor) / ((1 - rgb_mean) + eps),
    )
    return result.view(C, H, W)


# ─────────────────── inversion (single-channel → RGB) ───────────────────
def inverse_delta_local(tensor: torch.Tensor, delta: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Local-mean variant of *inverse_delta*."""
    C, H, W = tensor.shape
    if delta.shape == (C, H, W):
        return delta
    rgb_mean = tensor.mean(dim=0, keepdim=True)
    gd = delta.unsqueeze(0)
    result = torch.where(
        gd <= 0,
        gd * tensor / (rgb_mean + eps),
        gd * (1 - tensor) / ((1 - rgb_mean) + eps),
    )
    return result.view(C, H, W)


def inverse_luma(tensor: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Inverse of *rgb_to_luma* preserving chroma ratio."""
    if delta.dim() == 2:
        delta = delta.unsqueeze(0)
    r, g, b = tensor[0], tensor[1], tensor[2]
    luma = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(0)
    new_luma = torch.clamp(luma + delta, 0.0, 1.0)
    ratio = (new_luma + EPS) / (luma + EPS)
    perturbed = tensor * ratio
    return (perturbed - tensor)


def no_inversion(_: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:  # noqa: D401
    """Identity inversion."""
    return delta


_INVERSION_TABLE: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    'grayscale': inverse_delta,
    'grayscale_local': inverse_delta_local,
    'luma': inverse_luma,
    'noinvert': no_inversion,
}


def generate_inversion(kind: str):
    """Return inversion function requested by *kind*."""
    if kind not in _INVERSION_TABLE:
        raise ValueError(
            f"'{kind}' not in set of valid inversion handles: {_INVERSION_TABLE.keys()}"
        )
    return _INVERSION_TABLE[kind] 