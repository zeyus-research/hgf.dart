"""With HGF learning, a missing observation gives ν = ±inf.

``posterior_update_exponential_family_dynamic`` computes ν = mean(δ/Δ) before
checking ``observed``. On a missing trial the parent's value parent doesn't
move (Δ = 0), so ν = δ/0. ``xis`` is protected by the ``jnp.where``, but the
recorded ``nus`` (and ``to_pandas``) shows -inf/inf on every missing trial.
"""

import numpy as np
from _common import usdchf
from pyhgf.model import Network

y = usdchf()[:8]
observed = np.ones(len(y), dtype=int)
observed[3] = 0
net = Network().add_nodes(kind="ef-state", learning="hgf-2")
net.input_data(y, observed=observed)
print("observed:", observed.tolist())
print("nus:     ", np.asarray(net.node_trajectories[0]["nus"]).round(3).tolist())
