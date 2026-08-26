"""Probability paths: the interpolant a velocity field is regressed against.

A conditional flow matching path is defined by where a sample sits at time t
given its endpoints. Given noise x0 and data x1:

    linear (rectified flow):  x_t = (1-t) x0 + t x1,  target velocity = x1 - x0

The target velocity is constant along each pair, which is the whole reason
rectified flow produces straight trajectories. The velocity a trained model
learns is the conditional expectation of that target given (x_t, t), and the
averaging over crossing pairs is what puts curvature back in.

The VP (variance preserving) path is here for the comparison in task 06. It is
the path a DDPM implicitly uses, and its target velocity is not constant, which
is the point being demonstrated.
"""
from __future__ import annotations

import math

import torch


class LinearPath:
    """Straight interpolant. Rectified flow and Gaussian CFM both use this."""

    name = "linear"

    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        """Return (x_t, target velocity) for a batch of pairs.

        t has shape (B,) and is broadcast over feature dims.
        """
        shape = (-1,) + (1,) * (x0.dim() - 1)
        tt = t.view(shape)
        x_t = (1.0 - tt) * x0 + tt * x1
        return x_t, x1 - x0

    def straight_target(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0


class VPPath:
    """Variance preserving path, the one diffusion uses.

    x_t = alpha(t) x1 + sigma(t) x0 with alpha^2 + sigma^2 = 1, using the cosine
    schedule. The velocity target follows from differentiating that, and unlike
    the linear path it depends on t, so the trajectories are curved by
    construction rather than by the averaging.

    Time runs the same way as LinearPath: t=0 is noise, t=1 is data. Diffusion
    papers usually write it the other way round, and the first version of this
    class copied that convention while the samplers integrated 0 -> 1. The model
    then learned a field pointing from data to noise and was integrated
    backwards. It did not crash: it produced a plausible looking blob whose
    sliced W2 got *worse* with more NFE, 2.43 at one step against 2.84 at 128,
    which is what gave it away. Endpoint conventions are now pinned by a test.
    """

    name = "vp"

    @staticmethod
    def _alpha_sigma(t: torch.Tensor):
        """alpha multiplies data, sigma multiplies noise."""
        a = torch.sin(0.5 * math.pi * t)      # 0 at t=0, 1 at t=1
        s = torch.cos(0.5 * math.pi * t)      # 1 at t=0, 0 at t=1
        return a, s

    def sample(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        shape = (-1,) + (1,) * (x0.dim() - 1)
        tt = t.view(shape)
        a, s = self._alpha_sigma(tt)
        da = 0.5 * math.pi * torch.cos(0.5 * math.pi * tt)
        ds = -0.5 * math.pi * torch.sin(0.5 * math.pi * tt)
        x_t = a * x1 + s * x0
        return x_t, da * x1 + ds * x0


PATHS = {"linear": LinearPath, "vp": VPPath}
