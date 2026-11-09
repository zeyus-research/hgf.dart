"""rshgf's ``DeepNetwork(predict_precision=False)`` panics as soon as a
volatile layer has a volatility parent (the default): the prediction leaves
``expected_precision_vol`` as ``None`` and the prediction error unwraps it
(``src/updates/vectorised/volatile/prediction_error.rs``). JAX passes the
frozen volatility level through (``expected_*_vol = *_vol``) and runs.

Fix (in ``rshgf_deep_batched_kernel.patch``): in ``layer_prediction``, when ``predict_precision`` is off, carry
``mean_vol`` / ``precision_vol`` / ``effective_precision_vol`` into the
``expected_*`` fields instead of ``None`` (as
``_predict_volatility_level``'s caller does in JAX)."""

import _common  # noqa: F401, I001

import numpy as np
import optax
from pyhgf.model import DeepNetwork
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork

rng = np.random.default_rng(0)
x, y = rng.normal(size=(5, 3)), rng.normal(size=(5, 2))

jx = DeepNetwork(predict_precision=False)
for size in (2, 4, 3):
    jx = jx.add_layer(size=size)
jx.fit(x, y, optimiser=optax.sgd(0.01), check_gradient_health=False)
print("JAX: ok, predictions finite:", bool(np.isfinite(np.asarray(jx.predictions)).all()))

rs = RsDeepNetwork(predict_precision=False)
for size in (2, 4, 3):
    rs = rs.add_layer(size=size)
try:
    rs.fit(x, y, optimiser="sgd", learning_rate=0.01)
    print("rshgf: ok")
except BaseException as e:  # pyo3 panics are BaseExceptions
    print(f"rshgf: {type(e).__name__}: {e}")
