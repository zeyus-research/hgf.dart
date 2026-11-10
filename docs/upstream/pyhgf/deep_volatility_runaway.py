"""DeepNetwork's implied volatility levels run away on a small tanh network
with pyhgf's defaults, even with the weights frozen: an interior layer's
``mean_vol`` ratchets up (to ~10² within tens of samples), its predicted
precision collapses (Ω = e^μ_vol), one posterior mean step then jumps by
orders of magnitude, and the beliefs go NaN. The data are N(0, 1) predictors
and targets, i.e. plain unit-variance noise.

rshgf follows JAX until the first NaN and then keeps going: Rust's
``f64::max`` / ``min`` drop a NaN operand where ``jnp.maximum`` / ``clip``
propagate it, so the backends part ways silently (rshgf in float32 here).

The cause is the floor of the value posterior's precision (see P18 in
../README.md): each scheme runs again with the proposed curvature floor
(``_CurvatureFloor`` of tool/parity/gen_deep_fixtures.py, a patch of pyhgf's
``vectorised_layer_posterior_update``), which stays finite."""

import _common  # noqa: F401, I001  (first: sets float64 JAX)

import numpy as np
import warnings

import optax
import jax.numpy as jnp
from pyhgf.model import DeepNetwork

warnings.filterwarnings("ignore")

rng = np.random.default_rng(201)
weights = [rng.normal(scale=0.5, size=(2, 5)), rng.normal(scale=0.5, size=(4, 4))]
x = rng.normal(size=(60, 3))
y = rng.normal(size=(60, 2))

import dataclasses  # noqa: E402

from gen_deep_fixtures import _CurvatureFloor, _Nothing  # noqa: E402

for patch, vol in [(p, v) for p in (_Nothing, _CurvatureFloor)
                   for v in ("unbounded", "eHGF", "standard")]:
    with patch():
        net = DeepNetwork(volatility_updates=vol, coupling_fn=jnp.tanh)
        for size in (2, 4, 3):
            net = net.add_layer(size=size)
        layers = list(net.state.layers)
        for i, w in enumerate(weights, start=1):
            layers[i] = dataclasses.replace(layers[i], weights_mean=jnp.asarray(w))
        net.state = dataclasses.replace(net.state, layers=tuple(layers))
        net.fit(x, y, optimiser=optax.sgd(0.0), weight_update=False,
                record=("mean_vol",), check_gradient_health=False)
        mv = np.asarray(net.trajectories["mean_vol"][1])  # (T, 4): the hidden layer
        finite = np.isfinite(mv).all(axis=1)
        first_nan = int(np.argmin(finite)) if not finite.all() else None
        peak = np.nanmax(np.where(np.isfinite(mv), mv, np.nan), axis=1)
        print(f"{'floor' if patch is _CurvatureFloor else 'pyhgf':5s} {vol:9s} hidden mean_vol max at t=0,10,20,30: "
              + ", ".join(f"{peak[t]:.3g}" for t in (0, 10, 20, 30))
              + f"   first NaN at sample: {first_nan}")
