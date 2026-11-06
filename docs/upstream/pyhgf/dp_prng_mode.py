"""dp-state results depend on JAX's PRNG mode and float precision (P15).

``get_candidate`` draws its 20000 cluster proposals from ``random.key(42)``
on every trial. Which numbers that key gives depends on
``jax_threefry_partitionable`` (default True since JAX 0.5) and on
``jax_enable_x64``. pyhgf's ``tests/conftest.py`` pins the legacy mode, and
``test_get_candidate`` expects float32 values, so the tests check numbers
that users on a current JAX don't get.

The draws are the same on every trial (same key), so they could also be
drawn once: each trial now re-runs 40000 threefry hashes, the erf_inv of
each, and an argsort of 20000 weights.
"""

import subprocess
import sys

CODE = """
import jax
jax.config.update('jax_threefry_partitionable', {p})
jax.config.update('jax_enable_x64', {x})
import jax.numpy as jnp
from pyhgf.updates.prediction_error.dirichlet import get_candidate
m, s = get_candidate(value=5.0, sensory_precision=1.0,
                     expected_mean=jnp.array([0.0, -5.0]), expected_sigma=jnp.array([1.0, 3.0]))
print(f"partitionable={p!s:<5} x64={x!s:<5} mean {{float(m):.6f}}  sigma {{float(s):.6f}}")
"""

print("pyhgf's test_get_candidate expects mean 5.026636, sigma 1.2752448", flush=True)
for p in (True, False):
    for x in (True, False):
        # A fresh process each: the flags must be set before the first trace.
        subprocess.run([sys.executable, "-c", CODE.format(p=p, x=x)], check=True)
