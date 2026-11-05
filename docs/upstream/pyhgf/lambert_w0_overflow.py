"""``pyhgf.math.lambert_w0`` returns NaN for large z.

Its Halley step divides by ``2 f1² - f f2`` with ``f1 = (w + 1) eʷ``; for
z ≳ 1e151, f1² overflows to inf and the step becomes inf/inf = NaN. The
unbounded update then drops the expansion that calls it with such a z.
"""

import _common  # noqa: F401, I001  (first: sets float64 JAX)

import jax.numpy as jnp
import numpy as np
from pyhgf.math import lambert_w0


def w(z):
    return float(lambert_w0(jnp.asarray(z, dtype=jnp.float64)))


for z in (1e100, 1e150, 1e151, 2e151, 1e152, 1e200, 1e300):
    v = w(z)
    # Residual of w + log w = log z (finite where W is right).
    res = v + np.log(v) - np.log(z) if np.isfinite(v) and v > 0 else float("nan")
    print(f"z = {z:8.1e}   lambert_w0 = {v!r:<22} residual {res:.1e}")

lo, hi = 1e100, 1e300  # bisect the onset
assert np.isfinite(w(lo)) and not np.isfinite(w(hi))
for _ in range(60):
    mid = float(np.exp((np.log(lo) + np.log(hi)) / 2))
    lo, hi = (mid, hi) if np.isfinite(w(mid)) else (lo, mid)
print(f"NaN from z ≈ {hi:.3e}")
