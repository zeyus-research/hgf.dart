"""A dp-state node whose batch_size differs from its number of clusters
fails deep inside JAX (P16).

``batch_size`` (default 10) sizes ``n``, ``activated`` and the initial
``expected_means``, but the prediction step overwrites ``expected_means`` with
one entry per ef-state parent. Any mismatch, including the default
batch_size with fewer than 10 clusters, ends in a broadcasting error inside
the scan rather than a message at construction.
"""

import _common  # noqa: F401
import numpy as np
from pyhgf.model import Network

for batch_size in (3, 4, None):
    kwargs = {} if batch_size is None else {"batch_size": batch_size}
    net = Network().add_nodes(kind="dp-state", **kwargs).add_nodes(
        kind="ef-state", n_nodes=3, value_children=0
    )
    try:
        net.input_data(input_data=np.zeros(5))
        print(f"batch_size={batch_size}, 3 clusters: ran")
    except TypeError as e:
        print(f"batch_size={batch_size}, 3 clusters: TypeError {str(e)[:70]}")
