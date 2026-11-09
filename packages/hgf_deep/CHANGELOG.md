## Unreleased

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
