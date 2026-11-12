## Unreleased

- Continuous layers (pyhgf's continuous `DeepNetwork`, `input_data`):
  `LayerKind.continuous`, `addLayer(valueChildren:, volatilityChildren:,
  valueCoupling:, volatilityCoupling:, volatilityFullyConnected:,
  tonicDrift:, autoconnectionStrength:)`, `DeepNetwork.inputData`,
  `volatilityWeights`, `isContinuous`, `ContinuousTopology`. Standard,
  eHGF and unbounded updates, mean-field, noise parents of the
  observation layer (pyhgf #477). Saved by `toJson`. 28 new pyhgf
  fixtures within 1e-9, and checked against hgf's `Network`.
  `addLayer(addConstantInput:)` and `tonicVolatilityVol:` are now
  nullable (`null` keeps the previous defaults).

- The rest of pyhgf's learning: `LearningKind.synapticUncertainty`
  (weight beliefs; `SynapticUncertainty` settings, `installWeightBelief`,
  `weightPrecisionDeltas`, saved by `toJson` and `snapshot`),
  `batchUpdate(sampleWeight:)`, `DeepNetwork(updateInputLayer:)` and
  `checkGradientHealth` (`GradientHealth`). 16 new pyhgf fixtures, all
  within 1e-9.

- `DeepNetwork.toJson` / `DeepNetwork.fromJson`: save a network (settings,
  layers, weights, per-node parameters, current beliefs) and load it back
  bit for bit; custom couplings by name.
- `Dataset.fromCsv` (and `CsvData`): datasets from CSV text, with feature
  and target columns by name, text class labels (`classNames`),
  regression with one or more targets, rows with missing values dropped.
  `Dataset.featureNames`.
- `minibatches`: an epoch of shuffled minibatches for `batchUpdate`.
- `Standardiser(mean, sd)`, `toJson` / `fromJson`, to save the input
  scaling with a model.
- `example/workflow.dart` (+ `example/data/wine.csv`, UCI wine, CC BY 4.0):
  load, split, standardise, train with early stopping (`fit` or
  minibatches), evaluate, save and reload. README: "Getting started with
  your own data".
- `Layer` (a built layer, from `DeepNetwork.debugLayers`) is exported, for
  `hgf_gpu`'s `GpuDeepNetwork`.

- Batched products (`batchUpdate`, `predict`) run on `Float64x2` lanes on
  native targets: ~2.8× faster GEMM kernel, bit-identical to the scalar
  kernel, which Wasm and JS builds keep (`Float64x2` is emulated there).
- `batchUpdate`'s weight gradient runs through the same tiled product
  (bit-identical): with the above, `batchUpdate` 1.3–1.6× faster.
- Fixed: `weightInitialisation(seed:)` gave different weights in every run
  (it seeded through `Object.hash`, which is randomised per process).
- First version: a port of pyhgf's vectorised `DeepNetwork` (2026-11-09).
  - Layers of volatile, binary and categorical (output only) nodes, dense or
    one-to-one weights with an optional bias column, per-layer coupling
    functions (`CouplingFn` from hgf, or `CustomCoupling`).
  - Standard, eHGF and unbounded volatility updates; relaxed or mean-field;
    `feedforwardUncertainty`, `carryPrecision`, `predictPrecision`,
    `tonicVolatility`, initial-belief overrides.
  - `fit` (sequential, pyhgf's scan), `batchUpdate` (pyhgf's batch step,
    returning the input-layer errors), `predict`, recorded trajectories.
  - `Sgd` and `Adam` with optax's arithmetic; Xavier, He, orthogonal and
    sparse weight initialisation.
  - Float64; matches pyhgf's JAX backend under `jax_enable_x64` to 1e-9 on
    26 fixtures (`tool/parity/gen_deep_fixtures.py`), NaN included.
