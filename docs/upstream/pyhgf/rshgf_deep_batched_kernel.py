"""rshgf's batched DeepNetwork step (``batch_update``,
``src/vectorised/batched.rs``) disagrees with JAX's (``jax.vmap`` of the
per-sample sweeps) on networks with an interior layer below a curved
coupling, from the first call on a fresh network:

1. ``batched_posterior_update`` weights the g'' term of the posterior
   precision by the effective precision, ``Wᵀ(eff∘δ)``. Since #456 the
   per-sample kernel and JAX weight it by the relaxed gain, ``Wᵀ(gain∘δ)``,
   the same message as the mean shift. The two agree on leaf children
   (gain = eff there), which is all
   ``test_batch_update_matches_per_sample_reference`` covers (3 layers).
2. ``batched_volatile_prediction`` ignores ``predict_precision``.

Fix: ``sum_pi_vpe = msg`` in ``batched_posterior_update``, and the
``predict_precision`` branches of the per-sample ``layer_prediction`` in the
batched one (``rshgf_deep_batched_kernel.patch``, which also fixes the per-sample panic of ``rshgf_deep_predict_precision_panic.py``). With it the
float64 build matches JAX to ~1e-15 on the hgf.dart fixtures."""

import _common  # noqa: F401, I001

import dataclasses

import jax.numpy as jnp
import numpy as np
from pyhgf.model import DeepNetwork
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork

rng = np.random.default_rng(217)
sizes = (2, 5, 4, 3)
weights = [rng.normal(scale=0.5, size=(sizes[i - 1], sizes[i] + 1)) for i in range(1, 4)]
x, y = rng.normal(size=(32, 3)), rng.normal(size=(32, 2))

for predict_precision in (True, False):
    jx = DeepNetwork(coupling_fn=jnp.tanh, predict_precision=predict_precision)
    rs = RsDeepNetwork(coupling_fn="tanh", predict_precision=predict_precision)
    for size in sizes:
        jx = jx.add_layer(size=size, volatility_parent=predict_precision or None)
        rs = rs.add_layer(size=size, volatility_parent=predict_precision or None)
    layers = list(jx.state.layers)
    for i, w in enumerate(weights, start=1):
        layers[i] = dataclasses.replace(layers[i], weights_mean=jnp.asarray(w))
    jx.state = dataclasses.replace(jx.state, layers=tuple(layers))
    rs.set_weights([w.astype(rs.get_weights()[0].dtype) for w in weights])

    jx.batch_update(x, y)
    errors_rs = rs.batch_update(x.astype(rs.get_weights()[0].dtype), y.astype(rs.get_weights()[0].dtype))
    dev = float(np.max(np.abs(np.asarray(jx.input_errors) - errors_rs)))
    print(f"predict_precision={predict_precision!s:5s}  max |rshgf - JAX| input errors: {dev:.2e}"
          f"  (rshgf {rs.get_weights()[0].dtype})")
