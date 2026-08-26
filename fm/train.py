"""Conditional flow matching training.

The objective is a plain regression:

    L = E_{t, x0, x1} || v_theta(x_t, t) - u_t(x0, x1) ||^2

with (x_t, u_t) given by the path. There is no ELBO, no score, no noise
schedule to tune. The subtlety is entirely in what pairs (x0, x1) you draw:

  * independent coupling  -> conditional flow matching. Straight conditional
    paths, but paths for different pairs cross, and the model learns the
    average velocity at a crossing point, so the marginal trajectory bends.
  * reflow coupling       -> pairs are (noise, the point the model maps it to).
    Those paths do not cross, so averaging removes nothing, and the marginal
    trajectory stays straight. That is the whole reflow idea.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from torch import nn

from .paths import PATHS
from .toys import sample_data, sample_noise


@dataclass
class TrainConfig:
    dataset: str = "8gaussians"
    path: str = "linear"
    steps: int = 6000
    batch: int = 512
    lr: float = 2e-3
    seed: int = 0
    log_every: int = 500
    ema_decay: float = 0.999
    history: list = field(default_factory=list)


def train(model: nn.Module, cfg: TrainConfig, pairs: torch.Tensor | None = None, quiet=False):
    """Train v_theta. If `pairs` is given it is an (N, 2, D) tensor of coupled
    (x0, x1); otherwise pairs are drawn independently each step."""
    torch.manual_seed(cfg.seed)
    path = PATHS[cfg.path]()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.steps)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    data = None if pairs is not None else sample_data(cfg.dataset, 60_000, seed=cfg.seed + 1)
    started = time.perf_counter()

    for step in range(cfg.steps):
        if pairs is None:
            idx = torch.randint(0, data.shape[0], (cfg.batch,))
            x1 = data[idx]
            x0 = sample_noise(cfg.batch, x1.shape[1])
        else:
            idx = torch.randint(0, pairs.shape[0], (cfg.batch,))
            x0, x1 = pairs[idx, 0], pairs[idx, 1]

        t = torch.rand(cfg.batch)
        x_t, target = path.sample(x0, x1, t)
        loss = ((model(x_t, t) - target) ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                ema[k].mul_(cfg.ema_decay).add_(v.detach(), alpha=1 - cfg.ema_decay)

        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            cfg.history.append({"step": step, "loss": loss.item()})
            if not quiet:
                print(f"    step {step:6d}  loss {loss.item():.5f}")

    model.load_state_dict(ema)
    cfg.history.append({"step": "wall_clock_s", "loss": time.perf_counter() - started})
    return model
