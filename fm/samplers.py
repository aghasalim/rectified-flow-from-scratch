"""ODE samplers.

Sampling a flow model means integrating dx/dt = v(x, t) from t=0 (noise) to
t=1 (data). Number of function evaluations (NFE) is the cost, so the whole
question is how few steps you can take before the samples degrade. That is
exactly what straightness buys: a straight path is integrated exactly by one
Euler step, a curved one is not.

Every sampler here reports its own NFE rather than assuming steps == NFE,
because Heun and RK4 evaluate the field more than once per step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Trajectory:
    xs: torch.Tensor      # (steps+1, B, D), the full path, for plotting
    nfe: int


@torch.no_grad()
def euler(model, x: torch.Tensor, steps: int, keep_path: bool = False) -> Trajectory:
    dt = 1.0 / steps
    path = [x.clone()]
    nfe = 0
    for i in range(steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * model(x, t)
        nfe += 1
        if keep_path:
            path.append(x.clone())
    return Trajectory(torch.stack(path) if keep_path else x.unsqueeze(0), nfe)


@torch.no_grad()
def heun(model, x: torch.Tensor, steps: int, keep_path: bool = False) -> Trajectory:
    """Second order. Two evaluations per step, so NFE is 2*steps."""
    dt = 1.0 / steps
    path = [x.clone()]
    nfe = 0
    for i in range(steps):
        t0 = torch.full((x.shape[0],), i * dt, device=x.device)
        t1 = torch.full((x.shape[0],), (i + 1) * dt, device=x.device)
        v0 = model(x, t0)
        x_pred = x + dt * v0
        v1 = model(x_pred, t1)
        x = x + dt * 0.5 * (v0 + v1)
        nfe += 2
        if keep_path:
            path.append(x.clone())
    return Trajectory(torch.stack(path) if keep_path else x.unsqueeze(0), nfe)


@torch.no_grad()
def rk4(model, x: torch.Tensor, steps: int, keep_path: bool = False) -> Trajectory:
    dt = 1.0 / steps
    path = [x.clone()]
    nfe = 0
    for i in range(steps):
        b = x.shape[0]
        t0 = torch.full((b,), i * dt, device=x.device)
        th = torch.full((b,), (i + 0.5) * dt, device=x.device)
        t1 = torch.full((b,), (i + 1) * dt, device=x.device)
        k1 = model(x, t0)
        k2 = model(x + 0.5 * dt * k1, th)
        k3 = model(x + 0.5 * dt * k2, th)
        k4 = model(x + dt * k3, t1)
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        nfe += 4
        if keep_path:
            path.append(x.clone())
    return Trajectory(torch.stack(path) if keep_path else x.unsqueeze(0), nfe)


SAMPLERS = {"euler": euler, "heun": heun, "rk4": rk4}
