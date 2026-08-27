# Methods and detail

Long form detail moved out of the README.


## What reflow actually does


Plain conditional flow matching draws x0 from noise and x1 from data
independently. Each individual pair gets a straight conditional path, and the
target velocity x1 - x0 is constant along it. That part is fine. The problem is
that paths belonging to different pairs cross, and at a crossing point the model
can only learn one velocity, so it learns the average of the two. Averaging two
different directions gives a direction that points somewhere neither path was
going, and the marginal trajectory bends.

Reflow fixes the coupling instead of the model. Integrate the trained model
accurately from many noise samples, keep the pairs (x0, x1) it produced, and
retrain on those. Because x0 to x1 is now a function, the paths cannot cross, so
there is nothing to average away and the straight conditional paths survive into
the marginal field.

This shows up directly in the learned velocity field. The reflowed field barely
changes with t, which is what straight means:

![learned velocity fields](results/velocity-field-8gaussians.png)


## The cost


Reflow is not free and the tables above show the price if you look at the last
row. At 128 NFE on 8 gaussians the reflowed model scores 0.106 against 0.108 for
the model it was distilled from, so quality is basically unchanged here, but the
reflowed model can never be better than its teacher because it is trained on its
teacher's outputs. Any error the first model made is baked into the coupling. On
a harder dataset that gap would be visible.

The honest summary is that reflow trades a ceiling for a floor. You give up the
chance of improving with more compute, and in exchange you get most of the
quality at one step.


## What I got wrong


**The diffusion control ran backwards for a whole experiment.** I wrote `VPPath`
using the convention from the diffusion papers, where t=0 is data and t=1 is
noise, while `LinearPath` and every sampler in the repo use t=0 for noise. So the
model learned a field pointing from data to noise and then got integrated the
wrong way.

It did not crash and it did not produce NaNs. It produced a plausible looking
blob. What gave it away was that its sliced W2 got *worse* with more compute,
2.43 at one step against 2.84 at 128, which cannot happen if the ODE is oriented
correctly. I had a test asserting the VP path preserves variance and a test
asserting its velocity depends on t, and both passed the whole time. The test I
did not have was the obvious one, that t=0 is the noise end, and that is now
parametrised over every path class.

**My straightness metric returned exactly 0.0 on an untrained network and I
nearly "fixed" it.** An untrained MLP outputs a nearly constant field, and a
constant field really does give perfectly straight lines, so S=0 was correct. I
only convinced myself by writing a field that curves on purpose and checking S
went above zero. That negative test is in the suite now, because a metric that
can only return zero looks identical to one that is working.
