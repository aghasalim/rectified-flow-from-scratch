"""Reflow: retrain on the model's own noise-to-sample coupling.

Take the trained model, integrate it accurately from many noise samples, and
keep the pairs (x0, x1) it produced. Those pairs define a coupling in which the
trajectories do not cross, because the map x0 -> x1 is a function. Training a
fresh model on that coupling with the same linear path gives a field whose
marginal trajectories are straight, which is what makes few-step sampling work.

Reflow does not improve sample quality on its own. It trades a little quality
for a lot of straightness, and the straightness is what you spend at low NFE.
That trade is the thing to measure, so `results/` records both.
"""
from __future__ import annotations

import torch

from .samplers import rk4
from .toys import sample_noise


@torch.no_grad()
def build_coupling(model, n: int = 60_000, steps: int = 100, batch: int = 4096, seed: int = 7):
    """Integrate the model accurately and return (N, 2, D) pairs."""
    out = []
    remaining = n
    torch.manual_seed(seed)
    while remaining > 0:
        b = min(batch, remaining)
        x0 = sample_noise(b, 2)
        x1 = rk4(model, x0.clone(), steps=steps).xs[-1]
        out.append(torch.stack([x0, x1], dim=1))
        remaining -= b
    return torch.cat(out)
