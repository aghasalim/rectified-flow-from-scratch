"""The straightness metric, and the curvature it implies.

    S = E_{x0} integral_0^1 || (x1 - x0) - v(x_t, t) ||^2 dt

x1 here is the endpoint the ODE actually reaches from x0, not a data sample
drawn independently. That matters: S measures whether the model's own
trajectory is a straight line, which is what decides whether one Euler step
reproduces the full integration. Reading it as a distance to the data would
make it a sample quality metric, which it is not.

S = 0 exactly when every trajectory is straight, since then the velocity along
the path is constant and equal to the displacement.
"""
from __future__ import annotations

import torch

from .samplers import euler


@torch.no_grad()
def straightness(model, x0: torch.Tensor, steps: int = 100) -> dict:
    """Return S and a couple of related descriptive numbers."""
    traj = euler(model, x0.clone(), steps=steps, keep_path=True).xs   # (S+1, B, D)
    x1 = traj[-1]
    disp = x1 - x0                                                    # (B, D)

    total = torch.zeros(x0.shape[0])
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x0.shape[0],), i * dt)
        v = model(traj[i], t)
        total += ((disp - v) ** 2).sum(dim=-1) * dt
    s = total.mean().item()

    # Path length against straight line distance. 1.0 means perfectly straight.
    seg = (traj[1:] - traj[:-1]).norm(dim=-1).sum(dim=0)
    chord = disp.norm(dim=-1).clamp_min(1e-12)
    ratio = (seg / chord)
    return {
        "straightness_S": s,
        "path_length_ratio_mean": ratio.mean().item(),
        "path_length_ratio_p90": ratio.quantile(0.9).item(),
        "steps": steps,
    }
