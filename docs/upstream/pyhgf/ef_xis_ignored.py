"""``add_ef_state`` overwrites a ``xis`` given by the user with the default
(N(0, 1) or N(0, I)), silently."""

import _common  # noqa: F401, I001  (first: sets float64 JAX)

import jax.numpy as jnp
from pyhgf.model import Network

net = Network().add_nodes(kind="ef-state", xis=jnp.array([1.0, 2.0]))
print("asked for xis = [1, 2], got", net.attributes[0]["xis"])
