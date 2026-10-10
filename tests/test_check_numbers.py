"""The README number check has to fail on a changed cell, not just a missing number."""

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("check_numbers", ROOT / "scripts" / "check_numbers.py")
check_numbers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_numbers)


def _inputs():
    nfe = list(csv.DictReader((ROOT / "results" / "nfe-quality.csv").open()))
    st = list(csv.DictReader((ROOT / "results" / "straightness.csv").open()))
    return (ROOT / "README.md").read_text(), nfe, st


def test_committed_readme_matches_results():
    readme, nfe, st = _inputs()
    n, failures = check_numbers.check(readme, nfe, st)
    assert n == 42 and failures == []


def test_swapped_cells_are_caught():
    # Both numbers still appear elsewhere in the README, so a search for each
    # value would pass. Only a cell by cell comparison sees the swap.
    readme, nfe, st = _inputs()
    row = "| 128 | 0.110 | 0.117 | 0.112 |"
    assert row in readme
    _, failures = check_numbers.check(readme.replace(row, "| 128 | 0.117 | 0.110 | 0.112 |"), nfe, st)
    assert len(failures) == 2


def test_changed_straightness_cell_is_caught():
    readme, nfe, st = _inputs()
    row = "| moons | 1-rectified | 1.661 | 1.433 |"
    assert row in readme
    _, failures = check_numbers.check(readme.replace(row, "| moons | 1-rectified | 1.661 | 1.245 |"), nfe, st)
    assert len(failures) == 1
