"""Figures and the trajectory animation, drawn from committed results.

Everything here reads results/*.csv and the saved model checkpoints. Nothing
re-runs an experiment, so a figure cannot disagree with the numbers the README
quotes. That rule exists because the first version of a figure in a sibling
repo re-derived its own data, got a parameter wrong, and drew two identical
curves asserting a gap of 1x where the real one was 3297x.

    .venv/bin/python -m bench.figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from fm.models import VelocityMLP
from fm.samplers import euler
from fm.toys import sample_data, sample_noise

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

COLOURS = {"1-rectified": "#2166ac", "2-rectified": "#1a9850", "diffusion-vp": "#b2182b"}
LABELS = {"1-rectified": "1-rectified (CFM)", "2-rectified": "2-rectified (reflow)",
          "diffusion-vp": "diffusion (VP path)"}


def load_models(dataset: str, seed: int = 0):
    path = RESULTS / f"models-{dataset}-seed{seed}.pt"
    if not path.exists():
        return {}
    blobs = torch.load(path, map_location="cpu")
    out = {}
    for name, sd in blobs.items():
        m = VelocityMLP()
        m.load_state_dict(sd)
        m.eval()
        out[name] = m
    return out


def fig_nfe_quality(out: Path) -> Path:
    """Sample quality against compute budget, averaged over seeds with spread."""
    table = pd.read_csv(RESULTS / "nfe-quality.csv")
    table = table[table["sampler"] == "euler"]
    datasets = sorted(table["dataset"].unique())

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 5), squeeze=False)
    for axis, ds in zip(axes[0], datasets):
        sub = table[table["dataset"] == ds]
        for model in ["diffusion-vp", "1-rectified", "2-rectified"]:
            g = sub[sub["model"] == model].groupby("nfe")["sliced_w2"]
            med, lo, hi = g.median(), g.min(), g.max()
            axis.plot(med.index, med.values, marker="o", color=COLOURS[model],
                      label=LABELS[model], linewidth=1.9, markersize=6)
            axis.fill_between(med.index, lo.values, hi.values,
                              color=COLOURS[model], alpha=0.16, linewidth=0)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("NFE (number of velocity evaluations)")
        axis.set_ylabel("sliced $W_2$ to target  (lower is better)")
        axis.set_title(ds)
        axis.grid(alpha=0.3, which="both")
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Sample quality against compute budget, Euler sampler\n"
                 "shaded band is min to max over 3 seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_straightness(out: Path) -> Path:
    """The metric that explains the NFE curve."""
    table = pd.read_csv(RESULTS / "straightness.csv")
    datasets = sorted(table["dataset"].unique())
    models = ["diffusion-vp", "1-rectified", "2-rectified"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.26
    for i, model in enumerate(models):
        for axis, col, ylabel in ((left, "straightness_S", "straightness  $S$"),
                                  (right, "path_length_ratio_mean", "path length / straight-line distance")):
            vals = [table[(table["dataset"] == d) & (table["model"] == model)][col] for d in datasets]
            med = [float(v.median()) for v in vals]
            err = [[float(v.median() - v.min()) for v in vals],
                   [float(v.max() - v.median()) for v in vals]]
            pos = [j + i * width for j in range(len(datasets))]
            axis.bar(pos, med, width, yerr=err, capsize=4,
                     color=COLOURS[model], label=LABELS[model])
            axis.set_ylabel(ylabel)
    for axis, log in ((left, True), (right, False)):
        axis.set_xticks([j + width for j in range(len(datasets))])
        axis.set_xticklabels(datasets)
        axis.grid(alpha=0.3, axis="y")
        if log:
            axis.set_yscale("log")
    right.axhline(1.0, color="#333333", linestyle="--", linewidth=1.1)
    right.text(-0.35, 1.002, "1.0 = perfectly straight", fontsize=8.5, color="#333333")
    left.set_title("Straightness  $S = \\mathbb{E}\\int_0^1\\|(x_1-x_0)-v(x_t,t)\\|^2 dt$\nlower is straighter, log scale")
    right.set_title("Path length ratio")
    left.legend(frameon=False, fontsize=9)
    fig.suptitle("Reflow buys straightness, which is what few-step sampling spends", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def fig_trajectories(dataset: str, out: Path, seed: int = 0, n: int = 400) -> Path:
    """Actual integration paths, one panel per model. This is the picture that
    makes 'straight' mean something."""
    models = load_models(dataset, seed)
    if not models:
        return out
    order = ["diffusion-vp", "1-rectified", "2-rectified"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), squeeze=False)
    target = sample_data(dataset, 3000, seed=seed + 100)

    for axis, model in zip(axes[0], order):
        x0 = sample_noise(n, 2, seed=seed + 300)
        traj = euler(models[model], x0.clone(), steps=60, keep_path=True).xs.numpy()
        axis.scatter(target[:, 0], target[:, 1], s=3, c="#cccccc", alpha=0.55,
                     label="target", zorder=1)
        for i in range(n):
            axis.plot(traj[:, i, 0], traj[:, i, 1], linewidth=0.45,
                      color=COLOURS[model], alpha=0.32, zorder=2)
        axis.scatter(traj[0, :, 0], traj[0, :, 1], s=6, c="#333333", zorder=3, label="$x_0$ (noise)")
        axis.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, c=COLOURS[model], zorder=4, label="$x_1$")
        axis.set_title(LABELS[model])
        axis.set_aspect("equal")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"Integration paths, {dataset}: noise at t=0 to samples at t=1\n"
                 "straighter lines mean fewer Euler steps are needed", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def fig_velocity_field(dataset: str, out: Path, seed: int = 0) -> Path:
    """The learned field at several times, as a quiver plot."""
    models = load_models(dataset, seed)
    if not models:
        return out
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    order = ["1-rectified", "2-rectified"]
    fig, axes = plt.subplots(len(order), len(times),
                             figsize=(3.05 * len(times), 3.15 * len(order)), squeeze=False)
    lim = 6.0
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 19), np.linspace(-lim, lim, 19))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    target = sample_data(dataset, 2000, seed=seed + 100)

    for r, model in enumerate(order):
        for c, t in enumerate(times):
            axis = axes[r][c]
            v = models[model](grid, torch.full((grid.shape[0],), t)).numpy()
            mag = np.linalg.norm(v, axis=1)
            axis.scatter(target[:, 0], target[:, 1], s=1.5, c="#dddddd", zorder=1)
            axis.quiver(gx, gy, v[:, 0].reshape(gx.shape), v[:, 1].reshape(gx.shape),
                        mag.reshape(gx.shape), cmap="viridis", scale=90, width=0.005, zorder=2)
            axis.set_xlim(-lim, lim); axis.set_ylim(-lim, lim)
            axis.set_aspect("equal"); axis.set_xticks([]); axis.set_yticks([])
            if r == 0:
                axis.set_title(f"t = {t:g}", fontsize=11)
            if c == 0:
                axis.set_ylabel(LABELS[model], fontsize=9.5)
    fig.suptitle(f"Learned velocity field $v_\\theta(x,t)$, {dataset}\n"
                 "colour is speed; the 2-rectified field varies far less with t, which is what straight means",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_training_curves(out: Path) -> Path:
    """Loss histories, if the experiment recorded them."""
    path = RESULTS / "training-curves.csv"
    if not path.exists():
        return out
    table = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
    for axis, ds in zip(axes[0], sorted(table["dataset"].unique())):
        sub = table[table["dataset"] == ds]
        for model in ["diffusion-vp", "1-rectified", "2-rectified"]:
            g = sub[sub["model"] == model].groupby("step")["loss"]
            med, lo, hi = g.median(), g.min(), g.max()
            axis.plot(med.index, med.values, color=COLOURS[model], label=LABELS[model], linewidth=1.8)
            axis.fill_between(med.index, lo.values, hi.values, color=COLOURS[model], alpha=0.16, linewidth=0)
        axis.set_yscale("log"); axis.set_xlabel("training step"); axis.set_ylabel("CFM loss")
        axis.set_title(ds); axis.grid(alpha=0.3, which="both"); axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Training loss, median and range over 3 seeds\n"
                 "the objective is a plain regression, so the curves are dull, which is the point", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


@torch.no_grad()
def anim_trajectories(dataset: str, out: Path, seed: int = 0, n: int = 700, frames: int = 60) -> Path:
    """Animated GIF: particles carried from noise to the target distribution."""
    models = load_models(dataset, seed)
    if not models:
        return out
    order = ["diffusion-vp", "1-rectified", "2-rectified"]
    paths = {}
    for model in order:
        x0 = sample_noise(n, 2, seed=seed + 300)
        paths[model] = euler(models[model], x0.clone(), steps=frames, keep_path=True).xs.numpy()
    target = sample_data(dataset, 3000, seed=seed + 100)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))
    arts = []
    for axis, model in zip(axes, order):
        axis.scatter(target[:, 0], target[:, 1], s=3, c="#dddddd", alpha=0.6, zorder=1)
        sc = axis.scatter([], [], s=7, c=COLOURS[model], zorder=3)
        tail, = axis.plot([], [], linewidth=0, zorder=2)
        axis.set_xlim(-7, 7); axis.set_ylim(-7, 7); axis.set_aspect("equal")
        axis.set_title(LABELS[model], fontsize=11); axis.grid(alpha=0.22)
        arts.append((sc, tail))
    caption = fig.suptitle("", fontsize=13)

    def update(f):
        for (sc, _), model in zip(arts, order):
            sc.set_offsets(paths[model][f])
        caption.set_text(f"{dataset}: integrating $dx/dt = v_\\theta(x,t)$    "
                         f"t = {f / frames:.2f}")
        return [a for pair in arts for a in pair] + [caption]

    anim = FuncAnimation(fig, update, frames=frames + 1, interval=70, blit=False)
    anim.save(out, writer=PillowWriter(fps=14))
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    made = [
        fig_nfe_quality(RESULTS / "nfe-quality.png"),
        fig_straightness(RESULTS / "straightness.png"),
        fig_training_curves(RESULTS / "training-curves.png"),
    ]
    for ds in ("8gaussians", "moons"):
        made += [
            fig_trajectories(ds, RESULTS / f"trajectories-{ds}.png"),
            fig_velocity_field(ds, RESULTS / f"velocity-field-{ds}.png"),
            anim_trajectories(ds, RESULTS / f"animation-{ds}.gif"),
        ]
    for p in made:
        if p.exists():
            print(f"-> {p.relative_to(REPO)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
