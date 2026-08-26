"""Velocity networks.

Small on purpose. The 2D experiments are meant to run on a laptop CPU in
seconds, and the point of them is that you can look at the field and the
trajectories, not that the model is large.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """Sinusoidal features for t, then a small MLP.

    Feeding raw scalar t works but converges slower: the network has to build
    its own high frequency features to represent a field that changes quickly
    near the endpoints.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        ang = t.view(-1, 1) * freqs.view(1, -1) * 1000.0
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.mlp(emb)


class VelocityMLP(nn.Module):
    """v_theta(x, t) for low dimensional data."""

    def __init__(self, data_dim: int = 2, hidden: int = 256, depth: int = 4, t_dim: int = 64):
        super().__init__()
        self.time = TimeEmbedding(t_dim)
        layers: list[nn.Module] = [nn.Linear(data_dim + t_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, data_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        return self.net(torch.cat([x, self.time(t)], dim=-1))
