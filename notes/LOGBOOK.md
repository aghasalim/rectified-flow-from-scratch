# Logbook

## 2026-08-26, the diffusion control ran backwards and I nearly shipped it
**Tried:** trained three models per dataset (1-rectified, 2-rectified, VP-path diffusion control) and looked at the first seed before letting the full sweep run.
**Measured:** the control scored sliced W2 of 2.4312 at 1 NFE and 2.8353 at 128. Quality got *worse* with 50x more compute.
**Concluded:** that cannot happen with a correctly oriented ODE, so the bug was in the path not the model. `VPPath` used the diffusion-paper convention where t=0 is data and t=1 is noise, while `LinearPath` and every sampler use t=0 for noise. The model learned a field pointing from data to noise and was then integrated forwards. No crash, no NaN, just a plausible looking blob. After fixing the orientation the control reaches 0.1165 at 128 NFE, in line with the other two. I had a test for variance preservation and a test that the VP velocity depends on t, and both passed throughout. The missing test was the obvious one, that t=0 is the noise end, and it is now parametrised over every path class.

## 2026-08-26, straightness returned exactly zero and it was right
**Tried:** ran the straightness metric on an untrained network expecting a large positive number.
**Measured:** S = 0.0000, path length ratio 1.0000, exactly.
**Concluded:** correct, not a bug. An untrained MLP outputs a nearly constant field, and a constant field gives perfectly straight trajectories. I only convinced myself by building a field that curves on purpose and confirming S rose above 0.05. That negative test is now in the suite, because a metric that always returns zero is indistinguishable from one that works until you make it fail.

## 2026-08-26, reflow gives one-step sampling, measured over 3 seeds
**Tried:** full sweep, 2 datasets x 3 seeds x 3 models, 6000 training steps each, NFE grid 1 to 128 across Euler, Heun and RK4. 460.5 s total on an M4 CPU.
**Measured:** on moons the reflowed model scores 0.034 at 1 NFE and 0.031 at 128, so 128x less sampling compute for no measurable loss. Plain CFM needs 1.835 at 1 NFE. Straightness S drops from 1.662 to 0.00011, about 15000x, and the path length ratio goes to 1.00006. On 8 gaussians the same pattern holds with S dropping about 3200x.
**Concluded:** the straightness metric predicts the NFE curve, which is the point of measuring it rather than asserting it. The cost is a ceiling: the reflowed model is trained on its teacher's outputs so it can never beat the teacher, and at 128 NFE it is 0.106 against the teacher's 0.108 on 8 gaussians. Reflow trades the top of the curve for the bottom of it. Worth re-running on a dataset where the teacher is visibly imperfect, because these two toys are easy enough that the ceiling never binds.
