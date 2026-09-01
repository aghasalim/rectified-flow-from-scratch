#!/usr/bin/env bash
# Recompute what this repo publishes in other languages and require agreement.
#
# Every number in the README came out of one PyTorch implementation, and every
# figure reads the same CSVs the tables do, so nothing here has ever been
# checked against anything but itself. These are independent implementations of
# the parts that carry a number: the velocity network and the Euler integrator,
# the straightness metric, the sliced Wasserstein distance, the medians over
# seeds, and the inference behind the claims written in words. A mistake in the
# Python would have to be reproduced identically in C, Rust, Java, Go, SQL and
# JavaScript to survive.
#
# Each check is skipped with a clear message if its toolchain is absent, so this
# runs on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

pass=0 fail=0 skip=0

# VERIFY_ONLY runs just the checks whose name contains it, which is what the CI
# self test uses to corrupt a file and require one check to reject it.
selected () {
    case "$1" in *"${VERIFY_ONLY:-}"*) return 0 ;; *) return 1 ;; esac
}

run () {
    local name="$1" tool="$2"; shift 2
    selected "$name" || return
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# --- the median tables, three times ----------------------------------------
# SQL, Go and JavaScript each derive the 42 published table cells from the per
# seed CSVs. sqlite3 reads stdin, which inside a script is the script itself,
# so its input is closed explicitly. Its CSV output is CRLF, so the carriage
# returns come off before anything is compared.
check_medians () {
    sqlite3 -init verify/medians.sql :memory: "" < /dev/null | tr -d '\r' \
        | LC_ALL=C sort > "$tmp/sql.csv" || return 1
    if [ ! -s "$tmp/sql.csv" ]; then
        echo "SQL produced nothing"
        return 1
    fi
    local n
    n=$(wc -l < "$tmp/sql.csv" | tr -d ' ')
    if [ "$n" -ne 42 ]; then
        echo "SQL produced $n rows, expected 42"
        return 1
    fi
    echo "SQL recomputed $n median cells from results/nfe-quality.csv and results/straightness.csv"

    local ok=0
    if command -v go >/dev/null 2>&1; then
        ( cd verify/gocheck && go run . -root "$root" -out "$tmp/go.raw" ) || return 1
        LC_ALL=C sort "$tmp/go.raw" > "$tmp/go.csv"
        if ! diff "$tmp/sql.csv" "$tmp/go.csv" > "$tmp/sql-go.diff"; then
            echo "SQL and Go disagree:"; head -20 "$tmp/sql-go.diff"; return 1
        fi
        echo "Go agrees with SQL on all $n cells, exactly at the 1e-10 they are printed to"
        ok=1
    else
        echo "note: go is not installed, so the Go recomputation did not run"
    fi
    if command -v node >/dev/null 2>&1; then
        node verify/readme_tables.js "$root" "$tmp/js.raw" || return 1
        LC_ALL=C sort "$tmp/js.raw" > "$tmp/js.csv"
        if ! diff "$tmp/sql.csv" "$tmp/js.csv" > "$tmp/sql-js.diff"; then
            echo "SQL and JavaScript disagree:"; head -20 "$tmp/sql-js.diff"; return 1
        fi
        echo "JavaScript agrees with SQL on all $n cells, and every cell sits where the README puts it"
        ok=1
    else
        echo "note: node is not installed, so the README table check did not run"
    fi
    [ "$ok" -eq 1 ]
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o "$tmp/kernel" verify/kernel.c -lm || return 1
    "$tmp/kernel" "$root" 2-rectified || return 1
    "$tmp/kernel" "$root" diffusion-vp
}

check_rust () { ( cd verify/mcstraight && cargo run --release --quiet -- "$root" ); }

check_java () { java verify/SlicedW2.java "$root"; }

# The golden files are exports of the committed checkpoints, and the checkpoints
# are the only copy of the trained models. Nothing checked that loading one back
# still gives the published straightness and sliced W2, which is the claim the C,
# Rust and Java checks all rest on. --check recomputes them and writes nothing.
check_golden () {
    local py
    for py in "${PYTHON:-}" .venv/bin/python python3 python; do
        [ -n "$py" ] || continue
        if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import torch, numpy' 2>/dev/null; then
            "$py" verify/export_golden.py --check || return 1
            return 0
        fi
    done
    echo "skipped: no python with torch available"
    return 2
}

run_golden () {
    local name="Python, the checkpoints still produce results/"
    selected "$name" || return
    printf '\n=== %s ===\n' "$name"
    check_golden
    case $? in
        0) pass=$((pass + 1)) ;;
        2) skip=$((skip + 1)) ;;
        *) fail=$((fail + 1)) ;;
    esac
}

run "SQL, Go and JavaScript, the published median tables" sqlite3 check_medians
run "Go, structural validation of results/"               go      check_go
run "C, velocity network, Euler sampler and straightness" cc      check_c
run "Java, sliced Wasserstein-2"                          java    check_java
run "R, the claims written in words"                      Rscript Rscript verify/verify.R "$root"
run "Rust, Monte Carlo error bar on the published S"      cargo   check_rust
run_golden

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
