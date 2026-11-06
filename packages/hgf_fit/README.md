# hgf_fit

Parameter estimation for [`hgf`](../hgf) networks: priors, response models,
batched objectives with finite-difference gradients, and MAP optimisers that
fit many subjects in one batched pass.

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
  (subject, parameter vector) pairs in one `BatchNetwork` pass;
  `BatchedObjective` adds central finite-difference gradients in the same pass
  (`2k + 1` elements per gradient). Subjects with different series
  lengths or time steps go into separate batches automatically.
- **Optimisers.** `Lbfgs` and `NelderMead` are written as generators that
  yield evaluation requests, so `fitMap` runs every subject's optimiser in
  lock step and answers all their requests with one pass. `restarts:` adds
  random starts drawn from the priors and keeps the best.

## Performance

pyhgf's parameter-recovery notebook (binary 2-level HGF, ω₂ and β per agent,
320 trials), MAP by L-BFGS, single thread, AOT
(`benchmark/bin/fit_benchmark.dart`) against pyhgf + JAX
(`tool/parity/fit_map_jax.py`: the notebook's log-probability `vmap`-ed over
agents, exact gradients from `jax.value_and_grad`, SciPy's L-BFGS-B):

| agents | hgf_fit L-BFGS | JAX f64 + L-BFGS-B | hgf_fit Nelder–Mead |
|---|---|---|---|
| 20 | 29 ms (15 passes) | 94 ms (51 grad calls) | 113 ms |
| 200 | 292 ms (17 passes) | 1566 ms (126 grad calls) | 1115 ms |

Both evaluate the same objective: JAX's value at hgf_fit's estimates matches
hgf_fit's to 3e-13. hgf_fit reaches an equal or lower objective for every
agent. Fitting each agent separately takes as many passes as the slowest
agent needs; the joint 2N-dimensional L-BFGS-B needs more iterations as N
grows.

## Status

Phase 6 of the [roadmap](../../docs/ROADMAP.md): MAP estimation works.
Planned: Laplace (Hessian) uncertainty, HMC/NUTS, hierarchical priors,
isolate-parallel passes, and draw export for ArviZ.
