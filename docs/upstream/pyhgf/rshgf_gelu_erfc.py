"""rshgf's GELU coupling uses the Abramowitz-Stegun 7.1.26 erfc (|error| up
to 1.5e-7), although its doc comment calls it the exact erf-based GELU; JAX
(``jax.nn.gelu(approximate=False)``) uses XLA's full-precision erfc. A
value-coupled network shows the gap in float64.

Fix: use a full-precision erfc in ``src/math.rs`` (``libm::erfc``, or the
series + continued fraction of hgf.dart's ``math.dart: erfc``)."""

import _common  # noqa: F401, I001  (first: sets float64 JAX)

from functools import partial

import jax
import numpy as np
from pyhgf.model import Network
from pyhgf.rshgf import Network as RsNetwork

x = _common.usdchf()[:300]
gelu = partial(jax.nn.gelu, approximate=False)

jx = Network().add_nodes().add_nodes(value_children=0, coupling_fn=(gelu,))
jx.input_data(x)
rs = RsNetwork().add_nodes().add_nodes(value_children=0, coupling_fn="gelu")
rs.input_data(x)

a = np.asarray(jx.node_trajectories[0]["expected_mean"])
b = np.asarray(rs.node_trajectories[0]["expected_mean"])
print("max |rshgf - JAX| expected_mean (float64):", float(np.max(np.abs(a - b))))
print("scale of the values:", float(np.max(np.abs(a))))
for v in (-3.0, -1.2, 0.7):
    print(f"gelu({v}): JAX {float(gelu(v)):.17g}")
