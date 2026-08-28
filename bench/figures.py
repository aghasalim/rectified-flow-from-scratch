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
from matplotlib.collections import LineCollection

from bench.style import PALETTE, titled
from fm.models import VelocityMLP
from fm.samplers import euler
from fm.toys import sample_data, sample_noise

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# Red for the diffusion control and green for the reflowed model, because the
# README already reads that way in prose. Blue for plain flow matching is
# arbitrary, so it comes straight off the shared palette.
COLOURS = {"1-rectified": PALETTE[0], "2-rectified": PALETTE[2], "diffusion-vp": PALETTE[1]}
LABELS = {"1-rectified": "1-rectified (CFM)", "2-rectified": "2-rectified (reflow)",
          "diffusion-vp": "diffusion (VP path)"}
ORDER = ["diffusion-vp", "1-rectified", "2-rectified"]
PRETTY = {"8gaussians": "8 gaussians", "moons": "two moons"}
GREY = "#c9c9c9"



def _shrink_gif(path: Path, colours: int = 64) -> None:
    """Rewrite the GIF on one shared palette.

    PillowWriter gives every frame its own full palette, which is most of the file
    size and is wasted here: consecutive frames differ only slightly, so one palette
    taken from a middle frame covers all of them and lets the encoder store just the
    changes. Colour count is high enough that the antialiased text does not band.
    """
    from PIL import Image

    source = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(source.convert("RGB"))
            durations.append(source.info.get("duration", 62))
            source.seek(source.tell() + 1)
    except EOFError:
        pass
    shared = frames[len(frames) // 2].quantize(colours, method=Image.Quantize.MEDIANCUT)
    quantised = [f.quantize(palette=shared, dither=Image.Dither.NONE) for f in frames]
    # No disposal method: leaving it unset lets the encoder store only the region
    # that changed between frames. Setting disposal=2 forces a full redraw each
    # frame and made the file larger than the one it replaced.
    quantised[0].save(path, save_all=True, append_images=quantised[1:], loop=0,
                      duration=durations, optimize=True)

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
    """Sample quality against compute budget.

    Three seeds is too few for a min to max band, it just draws a fat rectangle
    that hides the median. Each seed is plotted faintly instead and the median
    on top of them, so the reader sees the actual spread.
    """
    table = pd.read_csv(RESULTS / "nfe-quality.csv")
    table = table[table["sampler"] == "euler"]
    datasets = [d for d in ("8gaussians", "moons") if d in set(table["dataset"])]

    claims = {"8gaussians": ("One step after reflow matches 128 steps without it",
                             "8 gaussians, Euler sampler, thin lines are the 3 seeds"),
              "moons": ("Same shape on a second dataset",
                        "two moons, Euler sampler, thin lines are the 3 seeds")}

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.1 * len(datasets), 4.5), squeeze=False)
    for axis, ds in zip(axes[0], datasets):
        sub = table[table["dataset"] == ds]
        for model in ORDER:
            rows = sub[sub["model"] == model]
            for _, seed_rows in rows.groupby("seed"):
                seed_rows = seed_rows.sort_values("nfe")
                axis.plot(seed_rows["nfe"], seed_rows["sliced_w2"], color=COLOURS[model],
                          linewidth=0.8, alpha=0.35, zorder=2)
            med = rows.groupby("nfe")["sliced_w2"].median()
            axis.plot(med.index, med.values, marker="o", color=COLOURS[model],
                      label=LABELS[model], linewidth=2.0, markersize=5, zorder=3)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("NFE (velocity calls per sample)")
        axis.set_ylabel("sliced $W_2$ to target (data units)")
        axis.grid(alpha=0.5, which="both")
        titled(axis, *claims[ds])
    axes[0][0].legend(loc="upper right", borderaxespad=0.6)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_straightness(out: Path) -> Path:
    """The metric that explains the NFE curve.

    Dots on a log axis, not bars. The values span more than three orders of
    magnitude, so a linear axis cannot show the small ones at all, and a bar on
    a log axis is worse: its length is measured from wherever the bottom of the
    axis happens to fall, so the length means nothing and the eye reads it
    anyway. A dot carries its value in its position, which is the part a log
    axis gets right. The sibling schrodinger-bridge repo dropped log bars for
    the same reason.

    Every number in the titles is read out of the CSV here. A number typed into
    a string drifts the moment the data is regenerated, and check_numbers.py
    only reads prose, so it would never catch it.
    """
    table = pd.read_csv(RESULTS / "straightness.csv")
    datasets = [d for d in ("8gaussians", "moons") if d in set(table["dataset"])]

    def median_of(dataset: str, model: str, col: str) -> float:
        rows = table[(table["dataset"] == dataset) & (table["model"] == model)]
        return float(rows[col].median())

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.7))
    offset = 0.25
    panels = ((left, "straightness_S", lambda v: v, "straightness $S$ (squared data units per unit $t$)"),
              (right, "path_length_ratio_mean", lambda v: (v - 1.0) * 100.0,
               "path length above the straight line (%)"))
    for i, model in enumerate(ORDER):
        pos = [j + (i - 1) * offset for j in range(len(datasets))]
        for axis, col, transform, ylabel in panels:
            vals = [table[(table["dataset"] == d) & (table["model"] == model)][col]
                    for d in datasets]
            med = [transform(float(v.median())) for v in vals]
            err = [[m - transform(float(v.min())) for m, v in zip(med, vals)],
                   [transform(float(v.max())) - m for m, v in zip(med, vals)]]
            axis.errorbar(pos, med, yerr=err, fmt="o", markersize=8, capsize=4,
                          linestyle="none", color=COLOURS[model], label=LABELS[model],
                          zorder=3)
            axis.set_ylabel(ylabel)
            for x, m in zip(pos, med):
                axis.text(x, m * 1.55, f"{m:.3g}", ha="center", va="bottom",
                          fontsize=8, color="#444444", zorder=4)

    for axis, col, transform, _ in panels:
        seen = [transform(float(v)) for v in table[col]]
        axis.set_yscale("log")
        axis.set_xticks(range(len(datasets)))
        axis.set_xticklabels([PRETTY[d] for d in datasets])
        axis.grid(False, axis="x")
        axis.grid(alpha=0.5, axis="y", which="major")
        axis.set_xlim(-0.6, len(datasets) - 0.4)
        axis.set_ylim(min(seen) / 3, max(seen) * 8)

    # The smallest drop across the datasets, so "or more" is true of both.
    factor = min(median_of(d, "1-rectified", "straightness_S")
                 / median_of(d, "2-rectified", "straightness_S") for d in datasets)
    quoted = "8gaussians" if "8gaussians" in datasets else datasets[0]
    excess = {m: (median_of(quoted, m, "path_length_ratio_mean") - 1.0) * 100.0
              for m in ("2-rectified", "diffusion-vp")}

    titled(left, f"Reflow cuts the curvature metric by {factor:.0f}x or more",
           "$S=\\mathbb{E}\\int_0^1\\|(x_1-x_0)-v(x_t,t)\\|^2dt$, zero means every path is a line")
    titled(right, "The reflowed paths are straight to five decimals",
           f"on {PRETTY[quoted]}, {excess['2-rectified']:.3g}% longer than the straight line, "
           f"against {excess['diffusion-vp']:.3g}% for the diffusion control")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.09))
    fig.text(0.0, -0.115, "dots are the median of 3 seeds, whiskers span min to max. "
             "Log axis, so a dot's height is the value and no length on the panel is.",
             fontsize=9.3, color="#5a5a5a")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


@torch.no_grad()
def fig_trajectories(dataset: str, out: Path, seed: int = 0, n: int = 400) -> Path:
    """Actual integration paths, one panel per model. This is the picture that
    makes 'straight' mean something."""
    models = load_models(dataset, seed)
    if not models:
        return out
    claims = {"diffusion-vp": ("Diffusion control: curved",
                              "the VP path bends, so one step cuts the corner"),
              "1-rectified": ("Flow matching: still curved",
                              "paths cross, the field learns their average"),
              "2-rectified": ("After one reflow: straight",
                              "a line, so one Euler step lands on it")}

    target = sample_data(dataset, 3000, seed=seed + 100)
    x0 = sample_noise(n, 2, seed=seed + 300)
    trajs = {m: euler(models[m], x0.clone(), steps=60, keep_path=True).xs.numpy()
             for m in ORDER}

    # Equal aspect on wide data leaves a dead band under the panels unless the
    # figure is as tall as the data actually is. Same limits on all three so
    # the panels can be compared by eye.
    pts = np.concatenate([t.reshape(-1, 2) for t in trajs.values()] + [target.numpy()])
    lo, hi = pts.min(0) - 0.4, pts.max(0) + 0.4
    span = hi - lo
    width, gap, edges, head, foot = 3.62, 0.5, 0.74, 0.85, 0.95
    height = width * span[1] / span[0]
    fig_w = 3 * width + 2 * gap + edges
    fig_h = height + head + foot
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(left=0.62 / fig_w, right=1 - 0.12 / fig_w,
                        top=1 - head / fig_h, bottom=foot / fig_h, wspace=gap / width)

    for axis, model in zip(axes[0], ORDER):
        traj = trajs[model]
        axis.scatter(target[:, 0], target[:, 1], s=3, c=GREY, alpha=0.55, zorder=1)
        axis.add_collection(LineCollection(
            [traj[:, i, :] for i in range(n)], linewidths=0.45,
            colors=COLOURS[model], alpha=0.3, zorder=2))
        axis.scatter(traj[0, :, 0], traj[0, :, 1], s=6, c="#333333", zorder=3)
        axis.scatter(traj[-1, :, 0], traj[-1, :, 1], s=6, c=COLOURS[model], zorder=4)
        axis.set_xlim(lo[0], hi[0])
        axis.set_ylim(lo[1], hi[1])
        axis.set_aspect("equal")
        axis.set_xlabel("$x_1$ (data units)")
        axis.grid(alpha=0.4)
        titled(axis, *claims[model])
    axes[0][0].set_ylabel("$x_2$ (data units)")
    fig.text(0.62 / fig_w, 0.16 / fig_h,
             f"{PRETTY[dataset]}, seed {seed}, {n} trajectories integrated with 60 Euler steps. "
             "Grey is the target sample, black dots are $x_0$ at $t=0$, coloured dots are $x_1$ at $t=1$.",
             fontsize=9.3, color="#5a5a5a")
    fig.savefig(out)
    plt.close(fig)
    return out


@torch.no_grad()
def fig_velocity_field(dataset: str, out: Path, seed: int = 0) -> Path:
    """The learned field at several times, as a quiver plot.

    One colour scale across all ten panels. Normalising each panel on its own
    would make a slow field and a fast one look identical, which is the whole
    thing being compared here.
    """
    models = load_models(dataset, seed)
    if not models:
        return out
    times = [0.0, 0.25, 0.5, 0.75, 1.0]
    order = ["1-rectified", "2-rectified"]
    lim = 6.0
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 19), np.linspace(-lim, lim, 19))
    grid = torch.tensor(np.stack([gx.ravel(), gy.ravel()], 1), dtype=torch.float32)
    target = sample_data(dataset, 2000, seed=seed + 100)

    fields = {(m, t): models[m](grid, torch.full((grid.shape[0],), t)).numpy()
              for m in order for t in times}
    top = max(float(np.linalg.norm(v, axis=1).max()) for v in fields.values())

    fig, axes = plt.subplots(len(order), len(times),
                             figsize=(2.5 * len(times), 2.75 * len(order)), squeeze=False)
    for r, model in enumerate(order):
        for c, t in enumerate(times):
            axis = axes[r][c]
            v = fields[(model, t)]
            mag = np.linalg.norm(v, axis=1)
            axis.scatter(target[:, 0], target[:, 1], s=1.2, c="#dddddd", zorder=1)
            q = axis.quiver(gx, gy, v[:, 0].reshape(gx.shape), v[:, 1].reshape(gx.shape),
                            mag.reshape(gx.shape), cmap="viridis", clim=(0.0, top),
                            scale=90, width=0.005, zorder=2)
            axis.set_xlim(-lim, lim)
            axis.set_ylim(-lim, lim)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(True)
            if r == 0:
                axis.set_title(f"$t$ = {t:g}", fontsize=10.5, loc="center", pad=6)
            if c == 0:
                axis.set_ylabel(LABELS[model], fontsize=9.5)

    fig.subplots_adjust(left=0.055, right=0.885, top=0.845, bottom=0.02,
                        wspace=0.07, hspace=0.07)
    fig.text(0.055, 0.965, "The reflowed field keeps pointing the same way as t advances",
             fontsize=12.5, fontweight="bold")
    fig.text(0.055, 0.925,
             f"{PRETTY[dataset]}, seed {seed}. A field that does not change with t is exactly one "
             "whose trajectories are straight lines.",
             fontsize=9.3, color="#5a5a5a")
    bar = fig.colorbar(q, cax=fig.add_axes([0.905, 0.05, 0.013, 0.72]))
    bar.set_label("speed $\\|v_\\theta(x,t)\\|$ (data units per unit $t$)", fontsize=9.5)
    bar.outline.set_visible(False)
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_training_curves(out: Path) -> Path:
    """Loss histories, if the experiment recorded them."""
    path = RESULTS / "training-curves.csv"
    if not path.exists():
        return out
    table = pd.read_csv(path)
    datasets = [d for d in ("8gaussians", "moons") if d in set(table["dataset"])]
    claims = {"8gaussians": ("Nothing interesting happens during training",
                             "8 gaussians, thin lines are the 3 seeds"),
              "moons": ("The same dull curve on two moons",
                        "two moons, thin lines are the 3 seeds")}

    fig, axes = plt.subplots(1, len(datasets), figsize=(6.1 * len(datasets), 4.4), squeeze=False)
    for axis, ds in zip(axes[0], datasets):
        sub = table[table["dataset"] == ds]
        for model in ORDER:
            rows = sub[sub["model"] == model]
            for _, seed_rows in rows.groupby("seed"):
                seed_rows = seed_rows.sort_values("step")
                axis.plot(seed_rows["step"], seed_rows["loss"], color=COLOURS[model],
                          linewidth=0.8, alpha=0.35, zorder=2)
            med = rows.groupby("step")["loss"].median()
            axis.plot(med.index, med.values, color=COLOURS[model], label=LABELS[model],
                      linewidth=2.0, zorder=3)
        axis.set_yscale("log")
        axis.set_xlabel("training step")
        axis.set_ylabel("regression loss (squared data units per unit $t$)")
        axis.grid(alpha=0.5, which="both")
        titled(axis, *claims[ds])
    axes[0][0].legend(loc="center right", borderaxespad=0.8)
    fig.text(0.0, -0.02,
             "Each model regresses against its own coupling, so the reflow loss starts low: "
             "its targets are already the first model's own paths. The levels are not "
             "comparable between models, only the shapes are.",
             fontsize=9.3, color="#5a5a5a")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


@torch.no_grad()
def anim_straightening(dataset: str, out: Path, seed: int = 0, n: int = 260,
                       steps: int = 56, hold: int = 14) -> Path:
    """Particles carried by the learned field, before and after reflow.

    Same noise sample, same integrator, same number of steps in both panels, so
    the only thing that differs on screen is the field doing the carrying.
    """
    models = load_models(dataset, seed)
    if not models:
        return out
    order = ["1-rectified", "2-rectified"]
    verdict = {"1-rectified": "paths bend", "2-rectified": "paths are straight"}
    x0 = sample_noise(n, 2, seed=seed + 300)
    paths = {m: euler(models[m], x0.clone(), steps=steps, keep_path=True).xs.numpy()
             for m in order}
    target = sample_data(dataset, 2500, seed=seed + 100)

    st = pd.read_csv(RESULTS / "straightness.csv")
    measured = {m: float(st[(st["dataset"] == dataset) & (st["model"] == m)
                            & (st["seed"] == seed)]["straightness_S"].iloc[0]) for m in order}

    lim = 6.2
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.15))
    heads, trails = [], []
    for axis, model in zip(axes, order):
        axis.scatter(target[:, 0], target[:, 1], s=2.5, c="#e0e0e0", zorder=1)
        trail = LineCollection([], linewidths=0.55, colors=COLOURS[model], alpha=0.3, zorder=2)
        axis.add_collection(trail)
        head = axis.scatter([], [], s=8, c=COLOURS[model], zorder=3)
        axis.set_xlim(-lim, lim)
        axis.set_ylim(-lim, lim)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        axis.set_title(f"{LABELS[model]}: {verdict[model]}", fontsize=10.5, loc="center")
        axis.text(0.03, 0.03, f"measured $S$ = {measured[model]:.4g}", transform=axis.transAxes,
                  fontsize=8.5, color="#5a5a5a")
        heads.append(head)
        trails.append(trail)
    clock = fig.text(0.5, 0.95, "", ha="center", fontsize=11.5, fontweight="bold")
    fig.text(0.5, 0.02, f"{PRETTY[dataset]}, seed {seed}. Same starting noise and the same "
             f"{steps} Euler steps in both panels.", ha="center", fontsize=8.5, color="#5a5a5a")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.08, wspace=0.06)

    def update(f):
        k = min(f, steps)
        for head, trail, model in zip(heads, trails, order):
            p = paths[model]
            head.set_offsets(p[k])
            trail.set_segments([p[:k + 1, i, :] for i in range(n)] if k else [])
        clock.set_text(f"$t$ = {k / steps:.2f}")
        return heads + trails + [clock]

    anim = FuncAnimation(fig, update, frames=steps + 1 + hold, interval=62, blit=False)
    anim.save(out, writer=PillowWriter(fps=16), dpi=130)
    _shrink_gif(out)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    # Only the figures some markdown in the repo actually shows. The moons
    # versions of the trajectory and field panels made the same point a second
    # time, nothing linked them, and they cost 1.3 MB in the tree.
    made = [
        fig_nfe_quality(RESULTS / "nfe-quality.png"),
        fig_straightness(RESULTS / "straightness.png"),
        fig_training_curves(RESULTS / "training-curves.png"),
        anim_straightening("8gaussians", RESULTS / "animation-8gaussians.gif"),
        fig_trajectories("8gaussians", RESULTS / "trajectories-8gaussians.png"),
        fig_velocity_field("8gaussians", RESULTS / "velocity-field-8gaussians.png"),
    ]
    for p in made:
        if p.exists():
            print(f"-> {p.relative_to(REPO)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
