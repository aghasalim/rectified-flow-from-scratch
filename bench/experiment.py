"""The main experiment: does reflow buy few-step sampling, and what does it cost?

Trains three models per dataset:
  1-rectified   linear path, independent coupling  (plain conditional flow matching)
  2-rectified   linear path, retrained on model 1's own coupling
  diffusion     VP path, independent coupling      (the curved-path control)

Then measures, for each, sample quality against NFE and the straightness metric.

Quality here is 2-Wasserstein against a held out sample of the target, computed
exactly in 1D projections and averaged over random directions (sliced W2). FID
needs an Inception network and 2D point clouds have no meaningful features for
one, so a distributional distance is the honest choice at this scale.

    .venv/bin/python -m bench.experiment            # writes results/*.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from fm.models import VelocityMLP
from fm.reflow import build_coupling
from fm.samplers import SAMPLERS
from fm.straightness import straightness
from fm.toys import sample_data, sample_noise
from fm.train import TrainConfig, train

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
NFE_GRID = [1, 2, 4, 8, 16, 32, 64, 128]


def sliced_w2(a: torch.Tensor, b: torch.Tensor, n_proj: int = 256, seed: int = 0) -> float:
    """Sliced 2-Wasserstein. Exact per projection, averaged over directions."""
    g = torch.Generator().manual_seed(seed)
    d = a.shape[1]
    dirs = torch.randn(n_proj, d, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    pa = (a @ dirs.T).sort(dim=0).values
    pb = (b @ dirs.T).sort(dim=0).values
    n = min(pa.shape[0], pb.shape[0])
    return ((pa[:n] - pb[:n]) ** 2).mean().sqrt().item()


def evaluate(model, dataset: str, seed: int, n: int = 8192) -> list[dict]:
    ref = sample_data(dataset, n, seed=seed + 100)
    rows = []
    for sampler_name in ("euler", "heun", "rk4"):
        fn = SAMPLERS[sampler_name]
        per_step = {"euler": 1, "heun": 2, "rk4": 4}[sampler_name]
        for nfe in NFE_GRID:
            steps = nfe // per_step
            if steps < 1:
                continue
            x0 = sample_noise(n, 2, seed=seed + 200)
            t0 = time.perf_counter()
            out = fn(model, x0.clone(), steps=steps).xs[-1]
            wall = time.perf_counter() - t0
            rows.append({
                "sampler": sampler_name, "steps": steps, "nfe": steps * per_step,
                "sliced_w2": sliced_w2(out, ref, seed=seed),
                "wall_s": wall,
            })
    return rows


def run(dataset: str, seed: int, steps: int, quiet: bool = False) -> dict:
    out: dict = {"dataset": dataset, "seed": seed}

    # Seed BEFORE constructing the network, not only inside train().
    #
    # train() calls torch.manual_seed at its start, which is too late: the
    # weights have already been drawn. PyTorch seeds its global RNG
    # nondeterministically per process, so the FIRST model built in a run got
    # different initial weights every time while the later two, built after a
    # previous train() had already seeded, reproduced exactly. That is precisely
    # what a re-run showed: diffusion-vp identical to the digit, 1-rectified
    # drifting from step 0.
    torch.manual_seed(seed)
    print(f"  [{dataset} seed={seed}] 1-rectified (linear path, independent coupling)")
    m1 = VelocityMLP()
    cfg1 = TrainConfig(dataset=dataset, path="linear", steps=steps, seed=seed)
    train(m1, cfg1, quiet=quiet)

    print(f"  [{dataset} seed={seed}] building reflow coupling from model 1")
    pairs = build_coupling(m1, n=60_000, steps=100, seed=seed + 50)

    torch.manual_seed(seed + 1)
    print(f"  [{dataset} seed={seed}] 2-rectified (retrained on that coupling)")
    m2 = VelocityMLP()
    cfg2 = TrainConfig(dataset=dataset, path="linear", steps=steps, seed=seed)
    train(m2, cfg2, pairs=pairs, quiet=quiet)

    torch.manual_seed(seed + 2)
    print(f"  [{dataset} seed={seed}] diffusion control (VP path)")
    md = VelocityMLP()
    cfgd = TrainConfig(dataset=dataset, path="vp", steps=steps, seed=seed)
    train(md, cfgd, quiet=quiet)

    models = {"1-rectified": m1, "2-rectified": m2, "diffusion-vp": md}
    out["curves"] = {
        "1-rectified": cfg1.history,
        "2-rectified": cfg2.history,
        "diffusion-vp": cfgd.history,
    }
    x0 = sample_noise(4096, 2, seed=seed + 300)
    out["models"] = {}
    for name, m in models.items():
        s = straightness(m, x0.clone(), steps=100)
        ev = evaluate(m, dataset, seed)
        out["models"][name] = {"straightness": s, "nfe": ev}
        print(f"    {name:13} S={s['straightness_S']:.4f}  "
              f"len/chord={s['path_length_ratio_mean']:.4f}  "
              f"W2@1={next(r for r in ev if r['nfe'] == 1)['sliced_w2']:.4f}  "
              f"W2@128={next(r for r in ev if r['nfe'] == 128)['sliced_w2']:.4f}")
    torch.save({k: m.state_dict() for k, m in models.items()},
               RESULTS / f"models-{dataset}-seed{seed}.pt")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["8gaussians", "moons"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    started = time.perf_counter()
    everything = []
    for ds in args.datasets:
        for seed in args.seeds:
            everything.append(run(ds, seed, args.steps, quiet=True))

    nfe_rows, str_rows, curve_rows = [], [], []
    for e in everything:
        for model, hist in e.get("curves", {}).items():
            for h in hist:
                if h["step"] == "wall_clock_s":
                    continue
                curve_rows.append({"dataset": e["dataset"], "seed": e["seed"],
                                   "model": model, "step": h["step"], "loss": h["loss"]})
        for model, payload in e["models"].items():
            for r in payload["nfe"]:
                nfe_rows.append({"dataset": e["dataset"], "seed": e["seed"],
                                 "model": model, **r})
            str_rows.append({"dataset": e["dataset"], "seed": e["seed"],
                             "model": model, **payload["straightness"]})

    for name, rows in (("nfe-quality", nfe_rows), ("straightness", str_rows),
                       ("training-curves", curve_rows)):
        if not rows:
            continue
        p = RESULTS / f"{name}.csv"
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p.relative_to(REPO)}  ({len(rows)} rows)")

    (RESULTS / "run-meta.json").write_text(json.dumps({
        "datasets": args.datasets, "seeds": args.seeds, "train_steps": args.steps,
        "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu",
    }, indent=1))
    print(f"total {time.perf_counter() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
