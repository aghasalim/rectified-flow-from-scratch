// Check every cell of the two README tables against results/, by position.
//
// scripts/check_numbers.py recomputes the same medians and then searches the
// README text for each one. That catches a stale number but not a misplaced
// one: swap two cells in a row and both strings are still present, so the
// search still finds them. This parses the markdown tables, maps each column
// header to a model and each row to an NFE or a dataset, and requires the cell
// at that position to be the median for that cell. It also writes the medians
// in the shared format verify/verify.sh diffs against SQL and Go.
//
//   node verify/readme_tables.js <repo-root> [medians-out]

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const out = process.argv[3];

function readCSV(p) {
  const lines = fs.readFileSync(p, "utf8").trim().split("\n");
  const header = lines[0].trim().split(",");
  return lines.slice(1).map((line) => {
    const cells = line.trim().split(",");
    if (cells.length !== header.length) {
      throw new Error(`${p}: row has ${cells.length} fields, header has ${header.length}`);
    }
    return Object.fromEntries(header.map((h, i) => [h, cells[i]]));
  });
}

function median(v) {
  const s = [...v].sort((a, b) => a - b);
  const n = s.length;
  if (n === 0) throw new Error("median of nothing");
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

const nfe = readCSV(path.join(root, "results", "nfe-quality.csv"));
const straight = readCSV(path.join(root, "results", "straightness.csv"));

const DATASETS = ["8gaussians", "moons"];
const MODELS = ["1-rectified", "2-rectified", "diffusion-vp"];
const NFES = [1, 2, 4, 8, 128];

const w2 = (ds, m, n) =>
  median(nfe.filter((r) => r.dataset === ds && r.model === m && r.sampler === "euler" &&
                           Number(r.nfe) === n).map((r) => Number(r.sliced_w2)));
const col = (ds, m, c) =>
  median(straight.filter((r) => r.dataset === ds && r.model === m).map((r) => Number(r[c])));

// The shared median table.
const rows = [];
for (const ds of DATASETS) {
  for (const m of MODELS) {
    for (const n of NFES) {
      rows.push(`w2,${ds},${m},${String(n).padStart(3, "0")},${w2(ds, m, n).toFixed(10)}`);
    }
    for (const c of ["straightness_S", "path_length_ratio_mean"]) {
      rows.push(`${c},${ds},${m},median,${col(ds, m, c).toFixed(10)}`);
    }
  }
}
if (out) fs.writeFileSync(out, rows.join("\n") + "\n");

// The README tables.
const readme = fs.readFileSync(path.join(root, "README.md"), "utf8").split("\n");
const clean = (s) => s.replace(/\*/g, "").trim();
const cells = (line) => line.split("|").slice(1, -1).map(clean);

const LABELS = {
  "diffusion (VP)": "diffusion-vp",
  "1-rectified (CFM)": "1-rectified",
  "2-rectified (reflow)": "2-rectified",
  "1-rectified": "1-rectified",
  "2-rectified": "2-rectified",
};

const failures = [];
let checked = 0;

function check(label, cell, want) {
  checked++;
  const decimals = (cell.split(".")[1] || "").length;
  const shown = want.toFixed(decimals);
  if (cell !== shown) failures.push(`${label}: README says ${cell}, results/ gives ${shown}`);
}

// Each NFE table is introduced by a bold dataset name above it.
let dataset = null;
let header = null;
let tablesSeen = 0;
for (const line of readme) {
  const flat = clean(line).toLowerCase();
  if (flat.startsWith("8 gaussians")) dataset = "8gaussians";
  else if (flat.startsWith("two moons")) dataset = "moons";
  if (!line.trim().startsWith("|")) {
    header = null; // a table ends at the first line that is not one of its rows
    continue;
  }
  const c = cells(line);
  if (c[0] === "NFE") { header = c; tablesSeen++; continue; }
  if (c[0] === "dataset" && c[1] === "model") { header = c; tablesSeen++; continue; }
  if (!header || c.every((x) => /^-*:?-*$/.test(x))) continue;

  if (header[0] === "NFE") {
    const n = Number(c[0]);
    if (!NFES.includes(n)) continue;
    for (let i = 1; i < header.length; i++) {
      const model = LABELS[header[i]];
      if (!model) { failures.push(`unknown column header ${header[i]}`); continue; }
      check(`W2 ${dataset}/${model}@${n}`, c[i], w2(dataset, model, n));
    }
  } else if (header[0] === "dataset") {
    const ds = c[0];
    const model = LABELS[c[1]];
    if (!DATASETS.includes(ds) || !model) { failures.push(`unknown row ${c[0]} ${c[1]}`); continue; }
    check(`S ${ds}/${model}`, c[2], col(ds, model, "straightness_S"));
    check(`ratio ${ds}/${model}`, c[3], col(ds, model, "path_length_ratio_mean"));
  }
}

if (tablesSeen !== 3) {
  failures.push(`found ${tablesSeen} tables in README.md, expected 3`);
}
if (checked !== 42) {
  failures.push(`checked ${checked} cells, expected 42`);
}

console.log(`JavaScript: ${checked} README table cells checked in place against results/`);
if (failures.length) {
  console.log("DISAGREEMENT:");
  for (const f of failures) console.log("  - " + f);
  process.exit(1);
}
process.exit(0);
