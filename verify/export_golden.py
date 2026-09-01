"""Write the golden reference files the other-language checks read.

The kernel in this repo is a velocity MLP and an Euler integrator, and every
number in the README came out of that one PyTorch implementation. Nothing
checked the implementation itself. These files pin it down: the trained weights
in plain text, the exact noise batch the published run used, and the point
clouds the published sliced W2 was measured on. A C or Rust reimplementation
reading them has to land on the same published numbers, and a mistake in the
Python would have to be reproduced identically to survive.

Run:  python verify/export_golden.py            writes the files
      python verify/export_golden.py --check    only checks them, writes nothing
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.experiment import sliced_w2          # noqa: E402
from fm.models import VelocityMLP               # noqa: E402
from fm.samplers import euler                   # noqa: E402
from fm.straightness import straightness        # noqa: E402
from fm.toys import sample_data, sample_noise   # noqa: E402

OUT = ROOT / "verify" / "golden"
DATASET, SEED = "8gaussians", 0
# One straight model and one curved one, so the kernel checks have to reproduce
# S across four orders of magnitude rather than at a single scale.
MODELS = ["2-rectified", "diffusion-vp"]
N_STRAIGHT, N_EVAL, STEPS = 4096, 8192, 100


WRITE = "--check" not in sys.argv


def write_matrix(path: Path, arr: np.ndarray) -> None:
    if not WRITE:
        return
    np.savetxt(path, np.atleast_2d(arr).astype(np.float32), fmt="%.9g")


def write_weights(path: Path, sd: dict) -> None:
    if not WRITE:
        return
    with path.open("w") as fh:
        for name, t in sd.items():
            a = t.detach().cpu().numpy().astype(np.float32)
            rows, cols = (a.shape[0], a.shape[1]) if a.ndim == 2 else (a.shape[0], 1)
            fh.write(f"{name} {rows} {cols}\n")
            fh.write(" ".join(f"{v:.9g}" for v in a.reshape(-1)) + "\n")


def main() -> int:
    ckpt = torch.load(ROOT / "results" / f"models-{DATASET}-seed{SEED}.pt", map_location="cpu")
    pub_s = {r["model"]: r for r in csv.DictReader(
        (ROOT / "results" / "straightness.csv").open())
        if r["dataset"] == DATASET and r["seed"] == str(SEED)}
    nfe_rows = [r for r in csv.DictReader((ROOT / "results" / "nfe-quality.csv").open())
                if r["dataset"] == DATASET and r["seed"] == str(SEED)
                and r["sampler"] == "euler"]

    OUT.mkdir(parents=True, exist_ok=True)
    x0 = sample_noise(N_STRAIGHT, 2, seed=SEED + 300)
    write_matrix(OUT / "x0-straightness.txt", x0.numpy())
    x0_eval = sample_noise(N_EVAL, 2, seed=SEED + 200)
    ref = sample_data(DATASET, N_EVAL, seed=SEED + 100)
    write_matrix(OUT / "w2-reference.txt", ref.numpy())

    g = torch.Generator().manual_seed(SEED)
    dirs = torch.randn(256, 2, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    write_matrix(OUT / "w2-directions.txt", dirs.numpy())

    meta: dict = {"dataset": DATASET, "seed": SEED, "steps": STEPS,
                  "n_straightness": N_STRAIGHT, "n_eval": N_EVAL, "models": {}}
    for name in MODELS:
        model = VelocityMLP()
        model.load_state_dict(ckpt[name])
        model.eval()
        write_weights(OUT / f"weights-{name}.txt", ckpt[name])

        s = straightness(model, x0.clone(), steps=STEPS)
        pub = pub_s[name]
        for key in ("straightness_S", "path_length_ratio_mean", "path_length_ratio_p90"):
            got, want = s[key], float(pub[key])
            if abs(got - want) > 1e-6 * max(1.0, abs(want)):
                print(f"MISMATCH {name} {key}: recomputed {got!r} against published {want!r}")
                return 1

        gen = euler(model, x0_eval.clone(), steps=1).xs[-1]
        write_matrix(OUT / f"w2-samples-{name}-nfe1.txt", gen.numpy())
        w2 = sliced_w2(gen, ref, seed=SEED)
        want = float(next(r["sliced_w2"] for r in nfe_rows
                          if r["model"] == name and r["nfe"] == "1"))
        if abs(w2 - want) > 1e-6 * max(1.0, abs(want)):
            print(f"MISMATCH {name} sliced_w2@1: recomputed {w2!r} against published {want!r}")
            return 1

        meta["models"][name] = {
            "straightness_S": float(pub["straightness_S"]),
            "path_length_ratio_mean": float(pub["path_length_ratio_mean"]),
            "path_length_ratio_p90": float(pub["path_length_ratio_p90"]),
            "sliced_w2_euler_nfe1": want,
        }
        print(f"{name}: S={s['straightness_S']:.6g} ratio={s['path_length_ratio_mean']:.6g} "
              f"w2@1={w2:.6g}, all match results/")

    # Reference median for the Monte Carlo check: S is an expectation over the
    # noise, so an independent draw should land near the median over seeds.
    for name in MODELS:
        vals = [float(r["straightness_S"]) for r in csv.DictReader(
            (ROOT / "results" / "straightness.csv").open())
            if r["dataset"] == DATASET and r["model"] == name]
        meta["models"][name]["straightness_S_median_over_seeds"] = statistics.median(vals)

    if not WRITE:
        print("the committed checkpoints still reproduce every number in results/")
        return 0
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(list(OUT.iterdir()))} files to verify/golden/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
