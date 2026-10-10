"""Fail if a number in the README's result tables no longer matches results/.

Same reason as the sibling repo: prose goes stale when the data is regenerated,
not when the prose is edited, so this runs in CI rather than on doc changes.

Each table is parsed and compared cell by cell: the W2 table under each dataset
heading (rows are NFE, columns are models) and the straightness table (one row
per dataset and model). A number that is right but sits in the wrong cell fails.
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["diffusion-vp", "1-rectified", "2-rectified"]
NAMES = {"diffusion (VP)": "diffusion-vp", "1-rectified": "1-rectified", "2-rectified": "2-rectified"}
HEADINGS = {"8 gaussians": "8gaussians", "two moons": "moons"}


def tables(text: str) -> list[tuple[str, list[list[str]]]]:
    """Every markdown table as (nearest bold heading above it, rows of cells), separator dropped."""
    out, heading, rows = [], "", []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
            continue
        if rows:
            out.append((heading, rows))
            rows = []
        m = re.match(r"\*\*(.+?)\*\*", line)
        if m:
            heading = m.group(1)
    return out


def expected(nfe: list[dict], st: list[dict]) -> tuple[dict, dict]:
    w2, straight = {}, {}
    for ds in ("8gaussians", "moons"):
        for m in MODELS:
            for n in (1, 2, 4, 8, 128):
                vals = [float(r["sliced_w2"]) for r in nfe
                        if r["dataset"] == ds and r["model"] == m
                        and r["nfe"] == str(n) and r["sampler"] == "euler"]
                if vals:
                    w2[(ds, m, str(n))] = f"{statistics.median(vals):.3f}"
            s = [float(r["straightness_S"]) for r in st if r["dataset"] == ds and r["model"] == m]
            ratio = [float(r["path_length_ratio_mean"]) for r in st
                     if r["dataset"] == ds and r["model"] == m]
            if s and ratio:
                s_med, r_med = statistics.median(s), statistics.median(ratio)
                straight[(ds, m)] = (f"{s_med:.{5 if s_med < 0.01 else 3}f}",
                                     f"{r_med:.{5 if r_med < 1.01 else 3}f}")
    return w2, straight


def check(readme: str, nfe: list[dict], st: list[dict]) -> tuple[int, list[str]]:
    w2, straight = expected(nfe, st)
    seen_w2, seen_st, failures = set(), set(), []
    for heading, rows in tables(readme):
        header, body = rows[0], rows[1:]
        if header[0] == "NFE" and heading in HEADINGS:
            ds = HEADINGS[heading]
            models = [NAMES.get(re.sub(r" \(.*\)$", "", h), NAMES.get(h)) for h in header[1:]]
            for row in body:
                for m, cell in zip(models, row[1:]):
                    key = (ds, m, row[0])
                    seen_w2.add(key)
                    if w2.get(key) != cell:
                        failures.append(f"W2 {ds}/{m}@{row[0]}: README says {cell}, results say {w2.get(key)}")
        elif header[:3] == ["dataset", "model", "S"]:
            for row in body:
                key = (row[0], NAMES.get(row[1]))
                seen_st.add(key)
                if straight.get(key) != (row[2], row[3]):
                    failures.append(f"S, ratio {key[0]}/{key[1]}: README says {row[2]}, {row[3]}, "
                                    f"results say {straight.get(key)}")
    for key in sorted(set(w2) - seen_w2):
        failures.append(f"W2 {'/'.join(key)} is in results/ but in no README table")
    for key in sorted(set(straight) - seen_st):
        failures.append(f"S {'/'.join(key)} is in results/ but in no README table")
    return len(seen_w2) + 2 * len(seen_st), failures


def main() -> int:
    nfe = list(csv.DictReader((ROOT / "results" / "nfe-quality.csv").open()))
    st = list(csv.DictReader((ROOT / "results" / "straightness.csv").open()))
    n, failures = check((ROOT / "README.md").read_text(), nfe, st)
    print(f"checked {n} table cells in README.md against results/")
    if failures:
        print("\nDRIFT DETECTED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("no drift")
    # It does not check claims written in words (ratios, multiples, ranges),
    # and it does not read notes/.
    print("this checks the README tables against results/, not claims written in words")
    return 0


if __name__ == "__main__":
    sys.exit(main())
