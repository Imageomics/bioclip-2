"""
Implementation of common operations for the Lorentz model of hyperbolic geometry.

This module represents hyperbolic points by their spatial coordinates on the
upper sheet of a two-sheeted hyperboloid.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def pairwise_inner(x: Tensor, y: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    x_time = torch.sqrt(1 / curv + torch.sum(x ** 2, dim=-1, keepdim=True))
    y_time = torch.sqrt(1 / curv + torch.sum(y ** 2, dim=-1, keepdim=True))
    return x @ y.T - x_time @ y_time.T


def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    c_xyl = -curv * pairwise_inner(x, y, curv)
    distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return distance / curv**0.5


def exp_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    sinh_input = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
    return torch.sinh(sinh_input) * x / torch.clamp(rc_xnorm, min=eps)


def log_map0(x: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    rc_x_time = torch.sqrt(1 + curv * torch.sum(x**2, dim=-1, keepdim=True))
    distance0 = torch.acosh(torch.clamp(rc_x_time, min=1 + eps))
    rc_xnorm = curv**0.5 * torch.norm(x, dim=-1, keepdim=True)
    return distance0 * x / torch.clamp(rc_xnorm, min=eps)


def half_aperture(
    x: Tensor, curv: float | Tensor = 1.0, min_radius: float = 0.1, eps: float = 1e-8
) -> Tensor:
    asin_input = 2 * min_radius / (torch.norm(x, dim=-1) * curv**0.5 + eps)
    return torch.asin(torch.clamp(asin_input, min=-1 + eps, max=1 - eps))


def oxy_angle(x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8) -> Tensor:
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    c_xyl = curv * (torch.sum(x * y, dim=-1) - x_time * y_time)
    acos_numer = y_time + c_xyl * x_time
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1) * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))


def pairwise_oxy_angle(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1))
    y_time = torch.sqrt(1 / curv + torch.sum(y**2, dim=-1))
    c_xyl = curv * pairwise_inner(x, y, curv)
    acos_numer = y_time[None, :] + c_xyl * x_time[:, None]
    acos_denom = torch.sqrt(torch.clamp(c_xyl**2 - 1, min=eps))
    acos_input = acos_numer / (torch.norm(x, dim=-1)[:, None] * acos_denom + eps)
    return torch.acos(torch.clamp(acos_input, min=-1 + eps, max=1 - eps))
