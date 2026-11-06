# hgf

Hierarchical Gaussian Filter (HGF) and generalised predictive-coding networks
in pure Dart. This is a port of
[pyhgf](https://github.com/ComputationalPsychiatry/pyhgf)'s nodalised engine.
It is tested against pyhgf reference trajectories (float64 JAX and rshgf) to
1e-9 relative error, on the Dart VM, dart2js and dart2wasm.

## Features

- Continuous-state, binary-state and constant-state (bias) nodes, with value
  and volatility coupling (with strengths).
- Exponential-family state nodes (`addEfState`): track the sufficient
  statistics of a univariate or multivariate normal, at a fixed learning
  rate or with the HGF-implied rate (`hgf-1/2/3` learning, Mathys & Weber
  2020).
- Categorical state nodes (`addCategoricalState`): one-hot observations,
  one binary HGF per category, and the implied Dirichlet distribution over
  the categories with its KL divergence (Bayesian surprise).
- Dirichlet process nodes (`addDpState`): sort observations into clusters,
  opening a new one when no existing cluster explains an observation (a
  Chinese restaurant process), each cluster a univariate ef-state node.
  pyhgf's random cluster proposals are reproduced, so the clustering
  matches pyhgf's.
- Non-linear value coupling: `linear`, `relu`, `sigmoid`, `tanh`,
  `leaky_relu`, `gelu`, or your own `CustomCoupling` (g, g', g'', and
  optionally g''' for exact gradients), declared on the parent.
- Relaxed (rHGF) or mean-field value-coupling updates.
- Standard, eHGF or unbounded (default) volatility updates.
- Missing observations and irregular time steps.
- `inputData` for whole series; `predict` + `step` for real-time
  trial-by-trial filtering; `append`/`reset`; and `setInitial` for parameter
  sweeps and fitting loops.
- Columnar `Trajectories` with CSV/JSON export (shaped like pyhgf's
  `node_trajectories`), and per-node / total surprise as in `to_pandas()`.
- Surprise under response models, the objective for fitting:
  pyhgf's `first_level_gaussian_surprise`, `total_gaussian_surprise`,
  `first_level_binary_surprise`, `binary_softmax` and
  `binary_softmax_inverse_temperature`, plus a Gaussian response model and
  custom ones (`ResponseFunction.fn`).
- Generative mode (`sample`): input nodes observe draws from their own
  predictive distribution; `saveState`/`restoreState` to draw several futures
  from the same beliefs.
- `NetworkSpec`: JSON network descriptions.
- `BatchNetwork`: many parameter sets (and series) filtered in lock step,
  bit-identical to `Network`, for fitting and simulation. With
  `tangents: k` it also carries forward-mode derivatives with respect to
  `k` parameters (initial values, node parameters, coupling strengths)
  through every update, so `surprise(..., gradient:)` returns exact
  gradients of the negative log-likelihood. A response function only
  reports its partial derivatives with respect to the beliefs it reads
  (`batchSurprisePartials`); the built-in ones do. Checked against finite
  differences on every batchable fixture, on the VM, dart2js and dart2wasm.

## Usage

```dart
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

// Binary 3-level HGF.
final hgf = Network(volatilityUpdates: VolatilityUpdates.eHGF)
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0], tonicVolatility: -3.0)
  ..addNodes(volatilityChildren: [1], tonicVolatility: -4.0, mean: 1.0);

final traj = hgf.inputData(Float64List.fromList(outcomes))!;
final belief = traj.series(1, NodeField.mean);        // μ2 per trial
File('traj.csv').writeAsStringSync(traj.toCsv());     // → pandas / R

// Real time: one observation per trial, from the current beliefs.
hgf.reset();
hgf.predict();                                        // predict this trial
final p = hgf.get(0, NodeField.expectedMean);         // P(u = 1), before u
hgf.step(Float64List.fromList([1.0]));                // observe u, update

// Parameter sweep without re-building the network.
for (final omega in omegas) {
  hgf.setInitial(1, NodeField.tonicVolatility, omega);
  hgf.inputData(data, record: false);                  // final state only
}

// Simulate an agent's observations from its own predictions.
final simulated = hgf.sample(200, random: math.Random(1))!;

// Fitting objective: total surprise of the participant's decisions.
final nll = hgf.surprise(
  u,
  BinarySoftmaxInverseTemperature(decisions, inverseTemperature: 2.0),
);
```

Observations for multiple inputs are row-major (`nTime × nInputColumns`),
in `inputIdxs` order: one column per input, `dimension` columns for an
ef-state input, one per category for a categorical input. `observed` flags are one per input node
(`nTime × nInputs`). Input nodes are the nodes without children.

```dart
// Track the mean and covariance of a 2-D signal; the learning rate is
// implied by one 3-level HGF per sufficient statistic.
final ef = Network()..addEfState(dimension: 2, learning: EfLearning.hgf2);
final traj = ef.inputData(xy)!;                       // xy: nTime × 2
final xis = traj.vectorAt(0, VectorField.xis, 99);    // [E x, E y, E x², E xy, E y²]

// Which of 3 options comes next? Node 0 is the categorical input; nodes
// 1-3 are its binary HGFs, 4-6 their value parents, 7 the shared
// volatility parent.
final cat = Network()..addCategoricalState(nCategories: 3);
final run = cat.inputData(
  oneHot, // nTime × 3
  fields: [...defaultRecordedFields, NodeField.klDivergence],
)!;
final p = run.vectorAt(0, VectorField.mean, t);       // P(category) after trial t
final alpha = run.vectorAt(0, VectorField.alpha, t);  // implied Dirichlet
final bayesianSurprise = run.series(0, NodeField.klDivergence);

// How many clusters, and where? Node 0 is the Dirichlet process input,
// nodes 1-3 its clusters (one ef-state node each).
final dp = Network()
  ..addDpState(batchSize: 3, alpha: 0.1)
  ..addEfState(nNodes: 3, valueChildren: [0], nus: 15.0);
dp.inputData(samples, record: false);                    // nTime × 1
final centres = dp.getVector(0, VectorField.expectedMeans);
final counts = dp.getVector(0, VectorField.n);           // per cluster
```

## Performance

The engine compiles the network into flat typed arrays and an int-coded
update schedule, and allocates nothing per time step. On an i7-7700 (Dart
AOT), a 4-node HGF with unbounded volatility updates filters a trial in
about 0.65 µs. That is faster than pyhgf's Rust backend and its JAX backend
(float64 and float32) on single series, with no compilation warm-up. Wasm
builds are about 1.6× slower than AOT. Run `melos run bench` at the repository root for
the full comparison.

Dirichlet process nodes are the exception: each trial weighs 20000 cluster
proposals, as pyhgf does, about 1.3 ms per trial on AOT (1.8 ms on Wasm),
against 3.8 ms for pyhgf's JAX backend.

## Differences from pyhgf

- Not yet ported: per-child coupling functions. `BatchNetwork` and `sample`
  don't take ef-state, categorical or dp-state nodes yet.
- Dirichlet process nodes skip missing observations (pyhgf clusters them,
  and a NaN placeholder opens a NaN cluster), may follow other nodes (pyhgf
  crashes when a continuous node comes first), and check at `build()` that
  there is one cluster per `batchSize`. Their clusters must be univariate
  ef-state nodes with a fixed learning rate. The cluster proposals match
  pyhgf's float64 draws under JAX's default PRNG mode (to about 1 ulp); with
  pyhgf in float32 or in JAX's legacy PRNG mode the clustering differs.
  See `docs/upstream/README.md`, P7 and P14–P16.
- Categorical nodes: `addCategoricalState` adds one node per call (pyhgf's
  `n_nodes > 1` builds a broken graph), may follow other nodes (pyhgf
  crashes when a continuous node comes first), and passes its parameters to
  the implied binary HGFs (pyhgf ignores `binary_parameters`). The Dirichlet
  KL divergence is computed in log space, so it stays finite where pyhgf's
  overflows to NaN (concentrations summing past ~171). On a missing
  observation, α is ±∞ as in pyhgf. See `docs/upstream/README.md`, P7 and
  P10–P13.
- An ef-state node with `dimension > 1` keeps its observation in
  `VectorField.mean`; the scalar `NodeField.mean` holds the first component.
  `addEfState(xis: ...)` sets the prior sufficient statistics (pyhgf ignores
  a `xis` parameter).
- Custom coupling functions come with their derivatives (no autodiff);
  `CustomCoupling.check` compares them with finite differences.
- `sample` draws independently for each input node (pyhgf's JAX version
  reuses one key for all inputs at a time step). Random streams differ from
  JAX's, so samples match pyhgf in distribution, not draw by draw.
- Constant nodes reject parameters (pyhgf silently ignores them): scale the
  bias with the coupling strength.
- Response functions are evaluated trial by trial while filtering, so they
  need no recorded trajectories. As a result, `TotalGaussianSurprise` is
  `+∞` from the trial where node 1's mean becomes NaN, not on every trial
  (totals agree). A NaN response marks a missed trial (no surprise).
- A coupling function belongs to the parent and applies to all its value
  children (as in rshgf).

### Opt-in divergence: `ehgfMgfCorrection`

`Network(volatilityUpdates: VolatilityUpdates.eHGF, ehgfMgfCorrection: true)`
(spec key `"ehgf_mgf_correction": true`) **changes the relaxed eHGF update
away from pyhgf 0.3.2**. It is off by default, and without it the engine
matches pyhgf.

The eHGF precision update re-predicts each volatility child's volatility
from the parent's new mean, `t·exp(κ μ_b + ω̃)`. The relaxed prediction of
that child also had the terms `Σ_p κ_p² / (2 π̂_p)` (one per volatility
parent) in the exponent, and pyhgf's re-prediction leaves them out. The
increment is then floored at zero on almost every trial, so volatility
parents never gain precision. A 3-level continuous HGF diverges as a
result: NaN from trial 44 on pyhgf's USD/CHF example, in JAX and rshgf
alike. The flag adds the missing terms. It changes nothing under mean-field
updates (whose prediction has no such terms) or for the standard and
unbounded updates.

The `*_mgf_correction` parity fixtures are generated from pyhgf's JAX
backend patched with the same change (`tool/parity/gen_fixtures.py`), and
Dart matches them to 1e-9. `dart run example/ehgf_mgf_correction.dart`
compares both versions on USD/CHF.
