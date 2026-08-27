"""Fail if a number quoted in README.md no longer matches results/.

Same reason as the sibling repo: prose goes stale when the data is regenerated,
not when the prose is edited, so this runs in CI rather than on doc changes.
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["diffusion-vp", "1-rectified", "2-rectified"]


def main() -> int:
    nfe = list(csv.DictReader((ROOT / "results" / "nfe-quality.csv").open()))
    st = list(csv.DictReader((ROOT / "results" / "straightness.csv").open()))
    body = (ROOT / "README.md").read_text()
    # Detail moved out of the README lives in notes/METHODS.md. A figure quoted
    # there is still a quoted figure and still has to match its source.
    _methods = ROOT / "notes" / "METHODS.md"
    if _methods.exists():
        body += "\n" + _methods.read_text()

    claims, failures = [], []
    for ds in ("8gaussians", "moons"):
        for m in MODELS:
            for n in (1, 2, 4, 8, 128):
                vals = [float(r["sliced_w2"]) for r in nfe
                        if r["dataset"] == ds and r["model"] == m
                        and r["nfe"] == str(n) and r["sampler"] == "euler"]
                if vals:
                    claims.append((f"W2 {ds}/{m}@{n}", statistics.median(vals), 3))
            s = [float(r["straightness_S"]) for r in st
                 if r["dataset"] == ds and r["model"] == m]
            ratio = [float(r["path_length_ratio_mean"]) for r in st
                     if r["dataset"] == ds and r["model"] == m]
            if s:
                med = statistics.median(s)
                claims.append((f"S {ds}/{m}", med, 5 if med < 0.01 else 3))
            if ratio:
                med = statistics.median(ratio)
                claims.append((f"ratio {ds}/{m}", med, 5 if med < 1.01 else 3))

    for label, value, places in claims:
        text = f"{value:.{places}f}"
        # boundary anchored so 0.108 does not match inside 0.1084
        if not re.search(r"(?<![\d.])" + re.escape(text) + r"(?!\d)", body):
            failures.append(f"{label} should read {text}, not found in README.md")

    print(f"checked {len(claims)} quoted figures against results/")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("no drift")
    return 0


if __name__ == "__main__":
    sys.exit(main())
