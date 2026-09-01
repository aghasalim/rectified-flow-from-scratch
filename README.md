# rectified-flow-from-scratch

[![ci](https://github.com/aghasalim/rectified-flow-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/rectified-flow-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![results](https://img.shields.io/badge/results-reproducible-1a9850.svg)](results/)

Conditional flow matching, rectified flow, and reflow, built from the papers. The
straightness metric is measured rather than asserted, and it is the number that
explains everything else in here.

Everything below ran on a laptop CPU (Apple M4). Total compute for the whole
results table is 766 s, about 13 minutes, recorded in
[results/run-meta.json](results/run-meta.json).

## The one result

All three models reach roughly the same sample quality if you give them enough
compute. Only the reflowed one still works when you give it a single step.

**8 gaussians**, sliced Wasserstein-2 to the target (lower is better), median of
3 seeds, Euler sampler:

| NFE | diffusion (VP) | 1-rectified (CFM) | 2-rectified (reflow) |
|---:|---:|---:|---:|
| 1 | 1.796 | 2.639 | **0.117** |
| 2 | 0.324 | 0.601 | **0.113** |
| 4 | 0.164 | 0.252 | **0.111** |
| 8 | 0.116 | 0.159 | **0.110** |
| 128 | 0.110 | 0.117 | 0.112 |

**two moons**, same protocol:

| NFE | diffusion (VP) | 1-rectified (CFM) | 2-rectified (reflow) |
|---:|---:|---:|---:|
| 1 | 0.988 | 1.821 | **0.032** |
| 2 | 0.412 | 0.757 | **0.032** |
| 4 | 0.184 | 0.343 | **0.032** |
| 8 | 0.088 | 0.172 | **0.032** |
| 128 | 0.032 | 0.031 | 0.031 |

The reflowed model at 1 NFE is as good as itself at 128 NFE. On moons that is a
128x reduction in sampling cost for no measurable loss, and the flat green line
below is what that looks like.

![sample quality against NFE](results/nfe-quality.png)

## Why it works, measured

The reason is straightness. If a trajectory is a straight line then one Euler
step integrates it exactly, because the velocity never changes along the path.
If it curves, one step cuts the corner and you land somewhere wrong.

The metric is

    S = E integral from 0 to 1 of || (x1 - x0) - v(x_t, t) ||^2 dt

where x1 is the endpoint the model's own ODE actually reaches, not an
independent data sample. S is zero exactly when every trajectory is straight.

| dataset | model | S | path length / straight line |
|---|---|---:|---:|
| 8gaussians | diffusion (VP) | 2.934 | 1.048 |
| 8gaussians | 1-rectified | 2.927 | 1.066 |
| 8gaussians | 2-rectified | **0.00099** | **1.00006** |
| moons | diffusion (VP) | 1.012 | 1.245 |
| moons | 1-rectified | 1.661 | 1.433 |
| moons | 2-rectified | **0.00011** | **1.00006** |

Reflow drops S by about 3000x on 8 gaussians and about 15000x on moons. The path
length ratio goes to 1.00006, which means the trajectories are straight to five
decimal places.

![straightness](results/straightness.png)

You can see the same thing without any metric at all, just by plotting the paths:

![integration paths](results/trajectories-8gaussians.png)

Watching the same paths get drawn makes it plainer still:

![particles carried from noise to the target](results/animation-8gaussians.gif)

*Both panels start from the same noise sample and take the same 56 Euler steps
on the same target, so the only thing that differs is the field doing the
carrying: plain flow matching on the left, the same model after one round of
reflow on the right.*

## What reflow actually does

Plain conditional flow matching draws x0 from noise and x1 from data
independently. Each pair gets a straight conditional path, but paths from
different pairs cross, and at a crossing the model can only learn one velocity,
so it learns the average of the two. Reflow fixes the coupling instead of the
model. Integrate the trained model from many noise samples, keep the (x0, x1)
pairs it produced, retrain on those, and because x0 to x1 is now a function
nothing crosses and nothing gets averaged away, which is what takes S on 8
gaussians from 2.927 to 0.00099.

![learned velocity fields](results/velocity-field-8gaussians.png)

Full detail in [notes/METHODS.md](notes/METHODS.md#what-reflow-actually-does).

## The cost

Reflow is not free and the tables above show the price if you look at the last
row. At 128 NFE on 8 gaussians the reflowed model scores 0.112 against 0.117 for
the model it was distilled from, both medians of 3 seeds. The reflowed number is
the lower one, but the seeds run 0.107 to 0.176 for it and 0.110 to 0.181 for
its teacher, so the two are level rather than one winning. Trained on its own
teacher's outputs it cannot really pull ahead, and reflow ends up trading a
ceiling for a floor.

Full detail in [notes/METHODS.md](notes/METHODS.md#the-cost).

## Training

The objective is a plain regression against the interpolant slope. No ELBO, no
score function, no noise schedule. The loss curves are dull, and that is the
point:

![training curves](results/training-curves.png)

## What I got wrong

**The diffusion control ran backwards for a whole experiment.** I wrote `VPPath`
using the convention from the diffusion papers, where t=0 is data and t=1 is
noise, while `LinearPath` and every sampler in the repo use t=0 for noise. So the
model learned a field pointing from data to noise and then got integrated the
wrong way. Nothing crashed and nothing went NaN, it produced a plausible looking
blob, and what gave it away was the sliced W2 getting worse with more compute,
2.43 at 1 NFE against 2.84 at 128 in that broken run. The test I was missing was
the obvious one, that t=0 is the noise end, and it now runs over every path
class.

Full detail in [notes/METHODS.md](notes/METHODS.md#what-i-got-wrong).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

```bash
python -m bench.experiment --datasets 8gaussians moons --seeds 0 1 2 --steps 6000
```

```bash
python -m bench.figures
```

The experiment takes about 13 minutes on an M4 CPU and writes `results/*.csv` plus
the model checkpoints. `bench.figures` reads those files and never re-runs an
experiment, so a figure cannot disagree with a number in this README.

## Everything here is recomputed in another language

Every number above came out of one PyTorch implementation. The tables come from
`results/*.csv`, the figures read the same CSVs, and `scripts/check_numbers.py`
recomputes the medians the same way `bench/` did, so nothing in the repo has
ever been checked against anything but itself. If the time embedding used the
wrong frequency, or the straightness metric compared against a data sample
instead of the trajectory endpoint, everything downstream would agree with the
mistake.

`verify/verify.sh` recomputes what this repo publishes in languages that share
no code with it. A mistake in the Python now has to be reproduced identically in
C, Rust, Java, Go, SQL and JavaScript to survive.

| language | file | what it recomputes, from what | measured agreement |
|---|---|---|---|
| C | `verify/kernel.c` | the velocity network, the Euler sampler and S, from the exported weights and the same 4096 point noise batch, against `results/straightness.csv` | worst relative difference 1.3e-07 over 6 published numbers |
| Rust | `verify/mcstraight/` | S again, from 4096 trajectories of its own noise (xorshift and Box-Muller, no crates), which puts the first error bar on it | published S is 1.21 and 0.11 standard errors from the independent estimate |
| Java | `verify/SlicedW2.java` | sliced Wasserstein-2 from the exported point clouds and projection directions, against `results/nfe-quality.csv` | relative 1.3e-10 and 6.2e-08 |
| SQL | `verify/medians.sql` | the 42 published table cells, as medians over seeds from the per seed CSVs | identical to Go and JavaScript at the 1e-10 they are printed to |
| Go | `verify/gocheck/` | the same 42 cells, plus structural validation of all 630 rows in `results/` | same 42 cells, no ragged rows, no NaN or Inf, 3 seeds behind every cell |
| JavaScript | `verify/readme_tables.js` | the same 42 cells, and where each one sits in the README tables | all 42 cells match the README in place |
| R | `verify/verify.R` | the claims written in words, not as figures: the quoted per seed ranges and whether the two models can be told apart | ranges match to 3 decimals, exact sign flip test gives p = 0.250 and p = 0.500 |
| Python | `verify/export_golden.py --check` | loads the committed checkpoints and re-derives S and sliced W2, which is what the C, Rust and Java checks rest on | matches `results/` to better than 1e-6 |
| shell | `verify/verify.sh` | the driver, skipping absent toolchains | prints N passed, M failed, K skipped |

The whole suite takes 138 s on an M4 laptop, and CI runs it, then corrupts a
results file and requires the checks to reject it.

Two things came out of writing it. The first is that `scripts/check_numbers.py`
searches the README text for each recomputed figure, which means it cannot see a
misplaced one: swapping the two middle cells of a row in the 8 gaussians table
leaves both strings present and it still passes. The JavaScript check parses the
tables and maps each column to a model, so it rejects that swap. The second is
that the published S had never had an error bar. It is an average over one batch
of noise, and the Rust run says the Monte Carlo standard error on 4096 draws is
0.000048 for the reflowed model and 0.008950 for the diffusion control, so the
fourth decimal of 0.00099 is real and the fourth of 2.934 is not.

Each check was tested by corrupting what it reads and confirming it fails:

| check | perturbation it was shown to catch |
|---|---|
| C kernel | one weight in the exported network moved by 0.05, and a shifted S in `results/straightness.csv` |
| Rust Monte Carlo | S moved from 0.00121 to 0.0016, which is 7.0 standard errors |
| Java sliced W2 | one of 8192 exported sample points moved |
| Go structural | a ragged row, a NaN, and a seed removed so a cell had 2 seeds behind it |
| JavaScript README | a median changed, and two cells swapped inside one table row |
| R inference | a changed value that moved a quoted per seed range |
| Python checkpoints | a value in `results/straightness.csv` the checkpoints no longer reproduce |

There is no Ruby or Perl here. Nothing was left that they would not simply
repeat, and a file that only looks like a check is worse than no file.

## Layout

```
fm/paths.py         linear and VP interpolants, and their velocity targets
fm/models.py        MLP velocity field with sinusoidal time embedding
fm/samplers.py      Euler, Heun, RK4, each reporting its own NFE
fm/straightness.py  the S metric
fm/train.py         the CFM training loop
fm/reflow.py        builds the model's own coupling
fm/toys.py          8 gaussians, moons, spiral, checkerboard
bench/experiment.py trains everything and writes the CSVs
bench/figures.py    all figures and the animation, from committed CSVs only
tests/              30 tests
verify/             the same numbers recomputed in C, Rust, Java, Go, SQL and JavaScript
```

## Sources

The papers this is built from, and what each one is actually for:

- **Lipman, Chen, Ben-Hamu, Nickel, Le. Flow Matching for Generative Modeling. ICLR 2023.** [arXiv:2210.02747](https://arxiv.org/abs/2210.02747) The conditional flow matching objective, and the argument that you can regress against a conditional path and still get the right marginal field.
- **Liu, Gong, Liu. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. ICLR 2023.** [arXiv:2209.03003](https://arxiv.org/abs/2209.03003) Rectified flow and the reflow procedure. The crossing-paths argument in the section above is theirs.
- **Albergo, Vanden-Eijnden. Building Normalizing Flows with Stochastic Interpolants. ICLR 2023.** [arXiv:2209.15571](https://arxiv.org/abs/2209.15571) The interpolant framing that makes diffusion and flow matching two choices of the same object.
- **Tong et al. Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport. TMLR 2024.** [arXiv:2302.00482](https://arxiv.org/abs/2302.00482) Minibatch OT coupling, which attacks the same crossing problem reflow does but during training instead of after.
- **Ho, Jain, Abbeel. Denoising Diffusion Probabilistic Models. NeurIPS 2020.** [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) The VP path used as the control arm here.
- **Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. Score-Based Generative Modeling through SDEs. ICLR 2021.** [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) The probability flow ODE, which is why a diffusion model can be sampled with the same ODE solvers used here.

Sliced Wasserstein as a distance between point clouds follows **Bonneel, Rabin, Peyré, Pfister, Sliced and Radon Wasserstein Barycenters of Measures, JMIV 2015**.

## Methodology

The rules this follows are in [`METHODOLOGY.md`](METHODOLOGY.md). The ones that bit
hardest here were "a reference implementation exists before the optimized one",
"report variance not just the point estimate", and "negative results stay in".

## Author

Aghasalim Mustafazada, third year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT, see [LICENSE](LICENSE).
