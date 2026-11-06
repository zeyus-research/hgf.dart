# hgf_fit

Parameter estimation for [`hgf`](../hgf) networks: priors, response models,
batched objectives with exact gradients, MAP optimisers that
fit many subjects in one batched pass, the Laplace approximation for
uncertainty and model evidence, full posteriors by NUTS, and hierarchical
(multilevel) models with group-level populations.

```dart
import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

final model = HgfModel(
  network: Network()
    ..addNodes(kind: NodeKind.binaryState)
    ..addNodes(valueChildren: [0]),
  response: const BinarySoftmaxReadout(),          // decisions ~ σ(β·logit μ̂)
  parameters: [
    Parameter.node('omega2', 1, NodeField.tonicVolatility,
        prior: TruncatedNormal(-3, 5, lower: -8, upper: 2)),
    Parameter.response('inverseTemperature',
        prior: const Uniform(0.2, 20), initial: 1.0),
  ],
);

final subjects = [
  for (final (u, r) in data) Subject(inputs: u, responses: r),
];
final fit = fitMap(model, subjects);               // L-BFGS by default
for (final r in fit.results) print(r.parameters);  // {omega2: …, inverseTemperature: …}

final lap = laplace(model, fit);                   // one more pass
for (final r in lap.results) print(r.interval());  // 95% intervals
print(lap.logEvidence);                            // for model comparison

final post = sampleNuts(model, subjects, chains: 4);  // full posteriors
print(post.results.first);                         // mean, sd, ESS, R̂
File('draws.json').writeAsStringSync(post.toJson());  // for ArviZ
```

- **Model.** `HgfModel` = a `Network` (the perceptual model), a
  `ResponseModel`, and free `Parameter`s. A parameter sets a node field
  (`tonicVolatility`, an initial `mean`, an input `precision`, …), a value or
  volatility coupling strength, or a response-model parameter, and can set
  several targets at once.
- **Priors** (`Normal`, `TruncatedNormal`, `Uniform`, `HalfNormal`,
  `LogNormal`, `GammaPrior`, `Flat`) are normalised, so log posteriors are
  comparable across models. Optimisers work on an unconstrained scale
  (identity, log or logit transforms picked from the prior's support).
- **Response models** read their parameters per batch element:
  `GaussianReadout` (continuous reports of a belief, with a fitted noise
  precision), `BinarySoftmaxReadout` (binary decisions with an inverse
  temperature), `ObservationSurprise` (fit the perceptual model to the
  observations alone, e.g. with `TotalGaussianSurprise`), or your own with
  `ResponseModel.fn`. Each also simulates responses (`HgfModel.simulate`),
  for parameter recovery.
- **Batched evaluation.** `HgfModel.negLogPosterior` evaluates any number of
  (subject, parameter vector) pairs in one `BatchNetwork` pass, with exact
  gradients: forward-mode tangents through every update of the filter (one
  direction per parameter; `BatchNetwork.tangents` in `package:hgf`), plus
  the priors' and transforms' derivatives. A parameter that only the
  response model reads (β, a response precision) costs nothing in the
  filter. A gradient costs one batch element instead of the `2k + 1` of
  central differences, and is exact (checked against Richardson finite
  differences in every update mode). Custom response models get exact
  gradients by reporting their partial derivatives
  (`ResponseModel.fn(surprisePartials:)`); without them, `BatchedObjective`
  falls back to finite differences (`Gradients.central` / `forward`).
  Subjects with different series lengths or time steps go into separate
  batches automatically.
- **Optimisers.** `Lbfgs` and `NelderMead` are written as generators that
  yield evaluation requests, so `fitMap` runs every subject's optimiser in
  lock step and answers all their requests with one pass. `restarts:` adds
  random starts drawn from the priors and keeps the best.
- **Laplace approximation.** `laplace(model, fit)` takes every subject's
  Hessian at its estimate by finite differences, all in one pass (`2k² + 1`
  elements per subject), and gives the posterior covariance, standard errors
  (delta method), intervals that respect each parameter's support, and the
  Laplace log evidence. On the recovery problem the evidence is within 0.01
  nats of grid quadrature when the posterior is well identified (0.08 when
  it isn't), the standard errors within 2% (13%), and 95% intervals cover
  the true values 92–98% of the time.

- **NUTS.** `sampleNuts(model, subjects)` draws from each subject's
  posterior with the No-U-Turn Sampler as Stan implements it (multinomial
  trajectories, the generalised no-U-turn criterion, dual-averaging step
  size, windowed adaptation of a diagonal or dense metric). Every chain of
  every subject is a generator, like the optimisers, so each batched pass
  is one leapfrog step of all of them, with exact gradients (forward
  differences where the model has none: the leapfrog integrator is
  reversible and volume preserving for any deterministic force, and the
  trajectory weights use the exact density, so the chain still targets the
  exact posterior). `PosteriorDraws` gives summaries, rank-normalised
  split R̂, bulk and tail ESS, MCSE and E-BFMI (ArviZ's estimators, checked
  against arviz-stats to 1e-9), Stan-style CSV, and JSON that ArviZ's
  `from_dict` reads (`example/posterior_sampling.dart`). Posterior moments
  match grid quadrature within Monte Carlo error. `sampleFunction` runs the
  sampler on any log density.

- **Hierarchical models.** `HierarchicalModel(model, pooled: {...},
  shared: {...})` builds a multilevel model on a subject-level `HgfModel`:
  a pooled parameter varies by subject around a `Population` (`N(μ, σ²)`
  on the parameter's unconstrained scale, so normal, log-normal or
  logit-normal across subjects, with hyperpriors on `μ` and `σ`), a shared
  one takes one value for everyone, the rest stay independent.
  `sampleHierarchical` samples the joint posterior by NUTS (non-centred by
  default, `centered: true` for well-identified subjects), one batched pass
  of all subjects of all chains per leapfrog step; `fitEmpiricalBayes`
  estimates the populations by Laplace-EM in milliseconds, with shrunken
  subject estimates (`example/hierarchical.dart`, pyhgf's multilevel
  notebook):

  ```dart
  final hier = HierarchicalModel(model, pooled: {
    'omega2': const Population(mean: Normal(-5, 5), sd: HalfNormal(10)),
    'inverseTemperature': const Population(mean: Normal(0, 2), sd: HalfNormal(2)),
  });
  final eb = fitEmpiricalBayes(hier, subjects);   // populations, quickly
  final fit = sampleHierarchical(hier, subjects); // the joint posterior
  print(fit.summary('omega2', group: 'mu'));
  File('draws.json').writeAsStringSync(fit.toJson());  // subject dimension
  ```

## Performance

pyhgf's parameter-recovery notebook (binary 2-level HGF, ω₂ and β per agent,
320 trials), MAP by L-BFGS, single thread, AOT
(`benchmark/bin/fit_benchmark.dart`) against pyhgf + JAX
(`tool/parity/fit_map_jax.py`: the notebook's log-probability `vmap`-ed over
agents, exact gradients from `jax.value_and_grad`, SciPy's L-BFGS-B):

| agents | hgf_fit L-BFGS, exact gradients | (central differences) | JAX f64 + L-BFGS-B | hgf_fit Nelder–Mead |
|---|---|---|---|---|
| 20 | 13.7 ms (15 passes) | 29 ms | 94 ms (51 grad calls) | 113 ms |
| 200 | 120 ms (17 passes) | 292 ms | 1566 ms (126 grad calls) | 1115 ms |

Both evaluate the same objective: JAX's value at hgf_fit's estimates matches
hgf_fit's to 3e-13. hgf_fit reaches an equal or lower objective for every
agent. Fitting each agent separately takes as many passes as the slowest
agent needs; the joint 2N-dimensional L-BFGS-B needs more iterations as N
grows.

NUTS on the same 20 agents, 4 chains × (1000 warmup + 1000 draws), one
core (`fit_benchmark --nuts`; `tool/parity/fit_nuts_jax.py` for PyMC, the
notebook's route via `wrap_jax`, and NumPyro, the whole sampler jitted; both
sample all agents as one 40-dimensional model, with the same priors):

| | wall time | leapfrog steps / draw | bulk ESS min / median | ESS/s min / median |
|---|---|---|---|---|
| hgf_fit (per agent, exact gradients) | 43.7 s | 4.5 per agent | 1006 / 1878 | 23.0 / 43.0 |
| hgf_fit (per agent, forward differences) | 64 s | 4.5 per agent | 830 / 1922 | 12.9 / 30.0 |
| NumPyro (joint, exact gradients) | 145 s (+6.7 s compile) | 15 | 2718 / 5806 | 18.7 / 39.9 |
| PyMC (joint, JAX op) | 214 s | 14.5 | 2798 / 5515 | 13.1 / 25.8 |

The posterior means agree (rms z of the 40 differences, in combined Monte
Carlo errors: 0.98 vs NumPyro, 1.10 vs PyMC; sds within 7%). The joint
samplers get 3× the ESS per draw (their long 40-dimensional trajectories
make draws antithetic), but hgf_fit is 3.3× faster per draw, so with exact
gradients it is ahead of NumPyro in ESS per second too (1.2× at the
minimum, 1.08× at the median), on one core. Exact gradients made each
leapfrog step 1.5× cheaper here: β is a response parameter, so the filter
carries one tangent direction (ω₂).

Hierarchical: pyhgf's multilevel notebook (10 agents, ω₂ normal and β
log-normal across agents, hyperpriors as in the notebook, centred), 4 chains
× (1000 + 1000), one core (`melos run bench:hier`;
`tool/parity/fit_hier_jax.py`):

| | wall time | bulk ESS min / median | ESS/s min / median |
|---|---|---|---|
| hgf_fit `sampleHierarchical` | 52.3 s | 2685 / 4047 | 51.4 / 77.4 |
| NumPyro (jitted) | 62.0 s (+8.1 s compile) | 2591 / 3468 | 41.8 / 55.9 |
| PyMC (the notebook's `CustomDist` + `wrap_jax`, `backend="jax"`) | 102.8 s | 2854 / 3936 | 27.8 / 38.3 |
| hgf_fit `fitEmpiricalBayes` (point estimates) | 45 ms | | |

Posterior means agree with both (rms z 1.11 and 0.83).

The cost of the tangents per element-trial, against a plain filter
(`package:hgf`'s `benchmark/bin/tangent_cost.dart`, AOT, response
included):

| model | plain | 1 direction | 2 directions | forward differences, 2 parameters |
|---|---|---|---|---|
| binary 2-level | 65 ns | 125 ns | 155 ns | 195 ns |
| continuous 3-level, standard | 133 ns | 321 ns | 401 ns | 398 ns |
| continuous 3-level, eHGF | 158 ns | 344 ns | 432 ns | 473 ns |
| continuous 3-level, unbounded | 314 ns | 613 ns | 764 ns | 942 ns |

Each direction costs about as much as the primal update's arithmetic (the
exps are shared), so exact gradients are about as fast as forward
differences when every parameter is in the network, and faster when some
are response parameters, or than the central differences MAP needs; their
real gain there is exactness.

## Status

Phase 6 of the [roadmap](../../docs/ROADMAP.md): MAP estimation, the
Laplace approximation, NUTS, exact gradients and hierarchical models work.
Planned: isolate-parallel passes.
