# rectified-flow-from-scratch

Conditional flow matching, rectified flow, and reflow — implemented from the papers, with the straightness metric measured rather than asserted.

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why this one first

Flow matching is the simpler object. You draw a straight line from noise to data, regress a velocity field against the line's slope, and integrate an ODE at sampling time. No forward SDE, no score function, no ELBO, no variational bound. Diffusion then reads as a special case — a particular curved probability path — instead of as prerequisite machinery you have to absorb before anything makes sense.

It's also cheap to work on. The 2D experiments run on a laptop, and you can *see* the velocity field and the trajectories, which is the fastest route to actually understanding what these models do. Repo 03 (latent diffusion) is the compute-hungry one; come here first.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- Tasks 00–04 run on CPU or any GPU. Task 05 wants 12GB+.

## Results

CIFAR-10, unconditional, FID vs number of function evaluations:

| NFE | DDPM | CFM (Gaussian path) | Rectified Flow | 2-Rectified | 1-step distilled |
|---:|---:|---:|---:|---:|---:|
| 1 | — | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO | — |
| 20 | TODO | TODO | TODO | TODO | — |
| 100 | TODO | TODO | TODO | TODO | — |

Straightness `S = E ∫₀¹ ‖(x₁−x₀) − v(x_t,t)‖² dt` (lower is straighter):

| Model | S | Mean path curvature |
|---|---:|---:|
| Diffusion (VP path) | TODO | TODO |
| Rectified flow | TODO | TODO |
| 2-rectified | TODO | TODO |

Fill from `results/`. The second table is the one that explains the first.

## Waves

```
00 bootstrap + metrics                    (serial)
   ├─ 01 theory: CFM + continuity eq      ┐
   └─ 02 2D toys: velocity fields         ┘ parallel
        └─ 03 ODE samplers + straightness (serial)
             ├─ 04 reflow                 ┐
             └─ 05 images (CIFAR/latents) ┘ parallel
                  └─ 06 diffusion-vs-FM ablation + writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `fm/__init__.py`, `fm/metrics/`, `data/` | — |
| 01 | `notes/00-flow-matching.md`, `fm/ref/` | `scripts/` |
| 02 | `fm/toys/`, `results/toys/` | `scripts/`, `fm/metrics/` |
| 03 | `fm/samplers.py`, `fm/straightness.py` | `fm/toys/`, `fm/ref/` |
| 04 | `fm/reflow.py` | `fm/samplers.py`, `fm/toys/` |
| 05 | `fm/models/`, `train/` | `fm/samplers.py`, `fm/metrics/` |
| 06 | `bench/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md).

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

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

MIT — see [LICENSE](LICENSE).
