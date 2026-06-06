"""Utilities for exporting SpeakerBeamSS to ONNX."""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from model.s4d import S4D, S4DKernel


def encoder_latent_length(wave_samples: int, kernel: int = 320, stride: int = 160) -> int:
    return (wave_samples - kernel) // stride + 1


def _c_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


def _c_div(ar, ai, br, bi, eps=1e-8):
    denom = br * br + bi * bi + eps
    return (ar * br + ai * bi) / denom, (ai * br - ar * bi) / denom


def _c_exp(ar, ai):
    e = torch.exp(ar)
    return e * torch.cos(ai), e * torch.sin(ai)


def s4d_kernel_real_onnx(kernel: S4DKernel, L: int) -> torch.Tensor:
    """
    ONNX-safe S4DKernel: real-valued ops only, L may be symbolic.
    Returns K shape (H, L).
    """
    dt = torch.exp(kernel.log_dt)
    C_r, C_i = kernel.C[..., 0], kernel.C[..., 1]
    A_r = -torch.exp(kernel.log_A_real)
    A_i = kernel.A_imag

    dtA_r, dtA_i = _c_mul(A_r, A_i, dt.unsqueeze(-1), torch.zeros_like(A_i))

    exp_dtA_r, exp_dtA_i = _c_exp(dtA_r, dtA_i)
    one_r = exp_dtA_r - 1.0
    one_i = exp_dtA_i
    C_adj_r, C_adj_i = _c_mul(C_r, C_i, *_c_div(one_r, one_i, A_r, A_i))

    t = torch.arange(L, device=dt.device, dtype=dt.dtype).view(1, 1, L)
    exp_t_r = torch.exp(dtA_r.unsqueeze(-1) * t) * torch.cos(dtA_i.unsqueeze(-1) * t)
    exp_t_i = torch.exp(dtA_r.unsqueeze(-1) * t) * torch.sin(dtA_i.unsqueeze(-1) * t)

    prod_r, prod_i = _c_mul(
        C_adj_r.unsqueeze(-1), C_adj_i.unsqueeze(-1), exp_t_r, exp_t_i
    )
    return 2.0 * prod_r.sum(dim=1)


def s4d_forward_dynamic(self: S4D, u: torch.Tensor, **kwargs) -> tuple[torch.Tensor, None]:
    """ONNX-safe S4D with length-dependent kernel (no FFT / complex)."""
    if not self.transposed:
        u = u.transpose(-1, -2)
    L = u.size(-1)
    k = s4d_kernel_real_onnx(self.kernel, L)
    k_conv = torch.flip(k, dims=[-1]).unsqueeze(1)
    y = F.conv1d(u, k_conv, padding=L - 1, groups=self.h)[..., :L]
    y = y + u * self.D.unsqueeze(-1)
    y = self.dropout(self.activation(y))
    y = self.output_linear(y)
    if not self.transposed:
        y = y.transpose(-1, -2)
    return y, None


def patch_s4d_for_onnx(model: torch.nn.Module) -> list[tuple[S4D, types.MethodType]]:
    """Replace S4D FFT forward with dynamic real conv1d (any latent length L)."""
    backups: list[tuple[S4D, types.MethodType]] = []
    for module in model.modules():
        if isinstance(module, S4D):
            backups.append((module, module.forward))
            module.forward = types.MethodType(s4d_forward_dynamic, module)
    return backups


def restore_s4d(backups: list[tuple[S4D, types.MethodType]]) -> None:
    for module, forward in backups:
        module.forward = forward
