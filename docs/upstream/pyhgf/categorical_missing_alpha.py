"""A missing observation makes a categorical node's alpha ±inf (P12).

``categorical_state_update`` doesn't check ``observed``. On a missing trial
the binary nodes get no prediction error, so their value parents don't move
and Δ = σ(μ_p) − μ̂_b is exactly 0: ν = δ/Δ − 1 is ±inf, and so is α. The KL
divergence is NaN on that trial and the next (it compares with the ±inf α).
The binary HGFs themselves are fine.
"""

from _common import steps  # noqa: F401  (sets float64 / CPU)
import numpy as np
from pyhgf.model import Network

u = np.eye(3)[[0, 0, 1, 0, 2, 0, 0, 1]]
observed = np.array([1, 1, 1, 0, 1, 1, 1, 1])
net = Network().add_nodes(kind="categorical-state", n_categories=3)
net.input_data(u, observed=observed)
traj = net.node_trajectories[0]
np.set_printoptions(precision=3)
print("observed:", observed)
print("alpha:\n", np.asarray(traj["alpha"]))
print("kl_divergence:", np.asarray(traj["kl_divergence"]))
