"""A dp-state node added after a continuous HGF crashes pyhgf (P7, the
dp-state case).

The dp-state node's clusters are ef-state nodes with a child, so they enter
the posterior pass of ``get_update_sequence``. ``update_fn`` is reset once
per pass, not per node, and the ef-state type (3) has no branch there: when
the continuous parent of an earlier input was handled first in the same pass,
the clusters inherit its ``continuous_node_posterior_update``, which fails
on their attributes. With the dp-state node first, it works.
"""

from _common import steps
import numpy as np
from pyhgf.model import Network


def with_dp(net, base):
    return net.add_nodes(kind="dp-state", batch_size=3, alpha=0.1).add_nodes(
        kind="ef-state", n_nodes=3, value_children=base
    )


ok = with_dp(Network(), 0)
print("dp alone, clusters 1-3:", [s for s in steps(ok) if s[0] in (1, 2, 3)])

net = with_dp(Network().add_nodes(precision=1e4).add_nodes(value_children=0), 2)
print("after a 2-level HGF, clusters 3-5:", [s for s in steps(net) if s[0] in (3, 4, 5)])
x = np.random.default_rng(1).normal(0.0, 1.0, 10)
try:
    net.input_data(np.stack([x, x], axis=1))
    print("input_data ran")
except KeyError as e:
    print("input_data: KeyError", e)
