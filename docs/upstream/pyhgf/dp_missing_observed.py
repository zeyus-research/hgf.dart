"""A dp-state node clusters missing observations like any other (P14).

``dirichlet_node_prediction_error`` never reads the node's ``observed`` flag:
an unobserved trial still picks or opens a cluster, counts in ``n`` and
``n_total``, and passes its value on. With NaN as the placeholder value (the
usual choice for a missing sample) it opens a NaN cluster, which then
poisons every later candidate.
"""

import _common  # noqa: F401  (float64 JAX)
import numpy as np
from pyhgf.model import Network

rng = np.random.default_rng(1)
x = np.concatenate([rng.normal(-3, 1, 10), rng.normal(4, 1, 10)])
observed = np.ones((len(x), 1), dtype=int)
observed[5] = 0

for placeholder in (0.0, np.nan):
    net = Network().add_nodes(kind="dp-state", batch_size=3, alpha=0.1).add_nodes(
        kind="ef-state", n_nodes=3, value_children=0
    )
    data = x.copy()
    data[5] = placeholder
    net.input_data(input_data=data, observed=observed)
    t = net.node_trajectories[0]
    print(f"placeholder {placeholder}: n_total at trials 4-6 {np.asarray(t['n_total'])[4:7]}, "
          f"n at trial 5 {np.asarray(t['n'])[5]}, "
          f"final expected_means {np.round(np.asarray(t['expected_means'])[-1], 3)}")
