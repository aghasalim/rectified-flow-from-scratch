"""2D target distributions.

These are the whole point of the repo's first half. They train in seconds on a
CPU and you can plot the learned field directly, so a bug shows up as a picture
that looks wrong rather than as a loss curve that is slightly off.
"""
from __future__ import annotations

import math

import torch


def eight_gaussians(n: int, g: torch.Generator, scale: float = 4.0, std: float = 0.25):
    centres = torch.tensor(
        [(math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)) for i in range(8)]
    ) * scale
    idx = torch.randint(0, 8, (n,), generator=g)
    return centres[idx] + std * torch.randn(n, 2, generator=g)


def two_moons(n: int, g: torch.Generator, noise: float = 0.1):
    half = n // 2
    a = math.pi * torch.rand(half, generator=g)
    outer = torch.stack([torch.cos(a) * 3, torch.sin(a) * 3], dim=1)
    b = math.pi * torch.rand(n - half, generator=g)
    inner = torch.stack([1.5 - torch.cos(b) * 3, -torch.sin(b) * 3 + 1.5], dim=1)
    return torch.cat([outer, inner]) + noise * torch.randn(n, 2, generator=g) * 3


def spiral(n: int, g: torch.Generator, noise: float = 0.08):
    t = torch.rand(n, generator=g) ** 0.5 * 3.5 * math.pi
    r = t * 0.45
    pts = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1)
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).view(-1, 1)
    return pts * sign + noise * torch.randn(n, 2, generator=g) * 3


def checkerboard(n: int, g: torch.Generator):
    x = torch.rand(n, generator=g) * 8 - 4
    y = torch.rand(n, generator=g) - torch.randint(0, 2, (n,), generator=g).float() * 2
    y = y + torch.floor(x) % 2
    return torch.stack([x, y * 2], dim=1)


DATASETS = {
    "8gaussians": eight_gaussians,
    "moons": two_moons,
    "spiral": spiral,
    "checkerboard": checkerboard,
}


def sample_data(name: str, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return DATASETS[name](n, g).float()


def sample_noise(n: int, dim: int = 2, seed: int | None = None) -> torch.Tensor:
    if seed is None:
        return torch.randn(n, dim)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dim, generator=g)
