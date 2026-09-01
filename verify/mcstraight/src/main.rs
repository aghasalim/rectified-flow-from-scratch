//! An error bar on the published straightness, from noise this repo has never seen.
//!
//! S is an expectation over the starting noise. results/straightness.csv holds
//! one estimate of it, from one batch of 4096 points drawn with one seed, and
//! nothing anywhere says how much that estimate moves if you draw a different
//! batch. So a published S could be off by more than the digits it is quoted to
//! and nothing in the repo would notice.
//!
//! This reimplements the velocity network and the Euler integrator a third
//! time, draws its own noise from a xorshift generator with Box-Muller (no
//! crates, and deliberately not PyTorch's generator, so the two batches are
//! independent), and reports the Monte Carlo standard error. The published
//! value has to sit inside it.
//!
//!   cd verify/mcstraight && cargo run --release -- <repo-root>

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::exit;

const STEPS: usize = 100;
const TDIM: usize = 64;
const N: usize = 4096;
const MODELS: [&str; 2] = ["2-rectified", "diffusion-vp"];

struct Tensor {
    rows: usize,
    cols: usize,
    v: Vec<f64>,
}

type Weights = HashMap<String, Tensor>;

fn load_weights(path: &str) -> Weights {
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("mcstraight: {}: {}", path, e);
        exit(1);
    });
    let mut w = Weights::new();
    let mut it = text.split_ascii_whitespace();
    while let Some(name) = it.next() {
        let rows: usize = it.next().expect("rows").parse().expect("rows");
        let cols: usize = it.next().expect("cols").parse().expect("cols");
        let v: Vec<f64> = (0..rows * cols)
            .map(|_| it.next().expect("value").parse::<f32>().expect("value") as f64)
            .collect();
        w.insert(name.to_string(), Tensor { rows, cols, v });
    }
    w
}

/// Weights are looked up by name, so a reordered export fails here.
fn get<'a>(w: &'a Weights, name: &str) -> &'a Tensor {
    w.get(name).unwrap_or_else(|| {
        eprintln!("mcstraight: no tensor named {} in the exported weights", name);
        exit(1);
    })
}

fn affine(w: &Weights, wname: &str, bname: &str, x: &[f64], y: &mut [f64]) {
    let m = get(w, wname);
    let b = get(w, bname);
    assert_eq!(m.cols, x.len(), "{} expects {} inputs", wname, m.cols);
    for o in 0..m.rows {
        let row = &m.v[o * m.cols..(o + 1) * m.cols];
        let mut acc = b.v[o];
        for i in 0..m.cols {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
}

fn silu(z: f64) -> f64 {
    z / (1.0 + (-z).exp())
}

fn time_embedding(w: &Weights, t: f64) -> [f64; TDIM] {
    let half = TDIM / 2;
    let mut raw = [0.0; TDIM];
    for j in 0..half {
        let freq = (-(10000f64.ln()) * j as f64 / half as f64).exp();
        let ang = t * freq * 1000.0;
        raw[j] = ang.sin();
        raw[half + j] = ang.cos();
    }
    let mut h = [0.0; TDIM];
    affine(w, "time.mlp.0.weight", "time.mlp.0.bias", &raw, &mut h);
    for v in h.iter_mut() {
        *v = silu(*v);
    }
    let mut out = [0.0; TDIM];
    affine(w, "time.mlp.2.weight", "time.mlp.2.bias", &h, &mut out);
    out
}

fn velocity(w: &Weights, x: [f64; 2], temb: &[f64; TDIM]) -> [f64; 2] {
    let mut input = [0.0; 2 + TDIM];
    input[0] = x[0];
    input[1] = x[1];
    input[2..].copy_from_slice(temb);

    let mut a = vec![0.0; 256];
    let mut b = vec![0.0; 256];
    affine(w, "net.0.weight", "net.0.bias", &input, &mut a);
    for v in a.iter_mut() {
        *v = silu(*v);
    }
    for layer in ["net.2", "net.4", "net.6"] {
        affine(
            w,
            &format!("{}.weight", layer),
            &format!("{}.bias", layer),
            &a,
            &mut b,
        );
        for i in 0..a.len() {
            a[i] = silu(b[i]);
        }
    }
    let mut out = [0.0; 2];
    affine(w, "net.8.weight", "net.8.bias", &a, &mut out);
    out
}

/// xorshift64*, then Box-Muller. Small, deterministic, and nothing to install.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        // (0, 1), open at zero so the logarithm below is finite.
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0) + 1e-12
    }
    fn normal_pair(&mut self) -> (f64, f64) {
        let (u1, u2) = (self.unit(), self.unit());
        let r = (-2.0 * u1.ln()).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        (r * th.cos(), r * th.sin())
    }
}

fn published_s(root: &str, model: &str) -> f64 {
    let path = format!("{}/results/straightness.csv", root);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("mcstraight: {}: {}", path, e);
        exit(1);
    });
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or("").trim_end().split(',').collect();
    let idx = |name: &str| {
        header.iter().position(|h| *h == name).unwrap_or_else(|| {
            eprintln!("mcstraight: straightness.csv has no column {}", name);
            exit(1);
        })
    };
    let (ids, iseed, imodel, is) = (idx("dataset"), idx("seed"), idx("model"), idx("straightness_S"));
    let mut vals = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.trim_end().split(',').collect();
        if f.len() != header.len() {
            eprintln!(
                "mcstraight: straightness.csv row has {} fields, header has {}",
                f.len(),
                header.len()
            );
            exit(1);
        }
        if f[ids] == "8gaussians" && f[imodel] == model {
            vals.push((f[iseed].to_string(), f[is].parse::<f64>().unwrap_or(f64::NAN)));
        }
    }
    match vals.iter().find(|(seed, _)| seed == "0") {
        Some((_, v)) if v.is_finite() => *v,
        _ => {
            eprintln!("mcstraight: no usable 8gaussians/seed 0/{} row", model);
            exit(1)
        }
    }
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let mut bad = false;

    for model in MODELS {
        let w = load_weights(&format!("{}/verify/golden/weights-{}.txt", root, model));
        let published = published_s(&root, model);

        let dt = 1.0 / STEPS as f64;
        let tembs: Vec<[f64; TDIM]> = (0..STEPS).map(|i| time_embedding(&w, i as f64 * dt)).collect();

        // Seeded so the run is reproducible, but from a generator PyTorch does
        // not have, so the batch is independent of the published one.
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut per_traj = Vec::with_capacity(N);
        for _ in 0..N {
            let (a, b) = rng.normal_pair();
            let x0 = [a, b];
            let mut x = x0;
            let (mut sv0, mut sv1, mut sv2) = (0.0, 0.0, 0.0);
            for temb in tembs.iter() {
                let v = velocity(&w, x, temb);
                sv0 += v[0] * dt;
                sv1 += v[1] * dt;
                sv2 += (v[0] * v[0] + v[1] * v[1]) * dt;
                x[0] += dt * v[0];
                x[1] += dt * v[1];
            }
            let d = [x[0] - x0[0], x[1] - x0[1]];
            per_traj.push(d[0] * d[0] + d[1] * d[1] - 2.0 * (d[0] * sv0 + d[1] * sv1) + sv2);
        }

        let n = per_traj.len() as f64;
        let mean = per_traj.iter().sum::<f64>() / n;
        let var = per_traj.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let se = (var / n).sqrt();
        // The published value is a mean over its own 4096 draws, so its
        // standard error is about the same size. Comparing two means adds them.
        let se_diff = se * 2f64.sqrt();
        let z = (mean - published).abs() / se_diff;
        println!(
            "  {:<13} independent MC S = {:.6} +- {:.6} (1 se, {} draws), published {:.6}, z = {:.2}",
            model, mean, se, N, published, z
        );
        if !(z <= 4.0) {
            println!("    DISAGREES, the published value is {:.1} standard errors away", z);
            bad = true;
        }
    }

    if bad {
        exit(1);
    }
}
