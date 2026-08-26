"""Correctness tests. Written against properties, not against saved outputs."""

import pytest
import torch

from fm.paths import LinearPath, VPPath
from fm.samplers import euler, heun, rk4
from fm.straightness import straightness
from fm.toys import DATASETS, sample_data, sample_noise


# --- paths -----------------------------------------------------------------
def test_linear_path_endpoints():
    x0, x1 = torch.randn(64, 2), torch.randn(64, 2)
    p = LinearPath()
    at0, _ = p.sample(x0, x1, torch.zeros(64))
    at1, _ = p.sample(x0, x1, torch.ones(64))
    assert torch.allclose(at0, x0, atol=1e-6)
    assert torch.allclose(at1, x1, atol=1e-6)


def test_linear_target_is_constant_in_t():
    """The defining property: the conditional velocity does not depend on t."""
    x0, x1 = torch.randn(32, 2), torch.randn(32, 2)
    p = LinearPath()
    _, v_a = p.sample(x0, x1, torch.full((32,), 0.1))
    _, v_b = p.sample(x0, x1, torch.full((32,), 0.9))
    assert torch.allclose(v_a, v_b, atol=1e-6)


@pytest.mark.parametrize("path", [LinearPath, VPPath])
def test_every_path_runs_noise_to_data(path):
    """t=0 must be x0 (noise) and t=1 must be x1 (data), for every path.

    This is the test that was missing. VPPath was written with the diffusion
    convention (t=0 data) while the samplers integrate 0 -> 1, so the model was
    trained to point from data to noise and then integrated the wrong way. No
    crash, no NaN, just a quietly wrong control arm.
    """
    x0 = torch.full((8, 2), -3.0)
    x1 = torch.full((8, 2), 7.0)
    p = path()
    at0, _ = p.sample(x0, x1, torch.zeros(8))
    at1, _ = p.sample(x0, x1, torch.ones(8))
    assert torch.allclose(at0, x0, atol=1e-5), f"{path.name}: t=0 is not the noise end"
    assert torch.allclose(at1, x1, atol=1e-5), f"{path.name}: t=1 is not the data end"


def test_vp_target_does_depend_on_t():
    """The contrast: the diffusion path is curved by construction."""
    x0, x1 = torch.randn(32, 2), torch.randn(32, 2)
    p = VPPath()
    _, v_a = p.sample(x0, x1, torch.full((32,), 0.1))
    _, v_b = p.sample(x0, x1, torch.full((32,), 0.9))
    assert not torch.allclose(v_a, v_b, atol=1e-3)


def test_vp_preserves_variance():
    a, s = VPPath._alpha_sigma(torch.linspace(0, 1, 11))
    assert torch.allclose(a**2 + s**2, torch.ones(11), atol=1e-6)


# --- samplers --------------------------------------------------------------
class ConstantField(torch.nn.Module):
    """v(x,t) = c. Trajectories are exact straight lines, so every sampler
    must land on x0 + c regardless of step count."""

    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, x, t):
        return self.c.expand_as(x)


@pytest.mark.parametrize("name,fn", [("euler", euler), ("heun", heun), ("rk4", rk4)])
@pytest.mark.parametrize("steps", [1, 2, 7, 50])
def test_samplers_exact_on_constant_field(name, fn, steps):
    c = torch.tensor([0.7, -1.3])
    x0 = torch.randn(16, 2)
    got = fn(ConstantField(c), x0.clone(), steps=steps).xs[-1]
    assert torch.allclose(got, x0 + c, atol=1e-5), f"{name} wrong at {steps} steps"


class LinearInT(torch.nn.Module):
    """v(x,t) = t. Exact solution from x0 is x0 + 1/2. Euler is first order so
    it carries O(dt) error; Heun and RK4 integrate this exactly."""

    def forward(self, x, t):
        return t.view(-1, 1).expand_as(x)


def test_higher_order_beats_euler_on_curved_field():
    x0 = torch.zeros(8, 2)
    exact = x0 + 0.5
    e = (euler(LinearInT(), x0.clone(), 4).xs[-1] - exact).abs().max()
    h = (heun(LinearInT(), x0.clone(), 4).xs[-1] - exact).abs().max()
    r = (rk4(LinearInT(), x0.clone(), 4).xs[-1] - exact).abs().max()
    assert h < e and r < e
    assert h < 1e-6 and r < 1e-6


@pytest.mark.parametrize("fn,per_step", [(euler, 1), (heun, 2), (rk4, 4)])
def test_reported_nfe_matches_evaluations(fn, per_step):
    calls = {"n": 0}

    class Counting(torch.nn.Module):
        def forward(self, x, t):
            calls["n"] += 1
            return torch.zeros_like(x)

    tr = fn(Counting(), torch.randn(4, 2), steps=5)
    assert tr.nfe == 5 * per_step == calls["n"]


# --- straightness ----------------------------------------------------------
def test_straightness_zero_on_constant_field():
    m = ConstantField(torch.tensor([1.0, -0.5]))
    out = straightness(m, torch.randn(64, 2), steps=50)
    assert out["straightness_S"] < 1e-8
    assert abs(out["path_length_ratio_mean"] - 1.0) < 1e-4


def test_straightness_positive_on_curved_field():
    """The negative test. A field that genuinely curves must score above zero,
    otherwise the metric is measuring nothing."""

    class Curved(torch.nn.Module):
        def forward(self, x, t):
            ang = 2.5 * t.view(-1, 1)
            return torch.cat([torch.cos(ang), torch.sin(ang)], dim=-1).expand_as(x)

    out = straightness(Curved(), torch.randn(64, 2), steps=50)
    assert out["straightness_S"] > 0.05
    assert out["path_length_ratio_mean"] > 1.05


# --- data ------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(DATASETS))
def test_datasets_shape_and_finite(name):
    d = sample_data(name, 1000, seed=3)
    assert d.shape == (1000, 2) and torch.isfinite(d).all()


def test_data_is_seeded():
    assert torch.equal(sample_data("moons", 100, seed=5), sample_data("moons", 100, seed=5))
    assert not torch.equal(sample_data("moons", 100, seed=5), sample_data("moons", 100, seed=6))


def test_noise_is_seeded():
    assert torch.equal(sample_noise(50, 2, seed=1), sample_noise(50, 2, seed=1))
