"""A categorical node added after a continuous HGF crashes pyhgf (P7).

``get_update_sequence`` sets ``update_fn = None`` once per posterior pass,
not per node, and only assigns it for node types 2 (continuous) and 4
(Dirichlet). The categorical node's binary nodes (type 1) have a child, so
they enter the posterior pass; when a continuous node was handled earlier in
the same pass, they inherit its ``continuous_node_posterior_update``, which
then fails on the binary node's attributes. With the categorical node added
first, the binary nodes come before any continuous node in the pass and get
no step, as intended.
"""

from _common import steps
import numpy as np
from pyhgf.model import Network

ok = Network().add_nodes(kind="categorical-state", n_categories=3)
print("categorical alone, binary nodes 1-3:", [s for s in steps(ok) if s[0] in (1, 2, 3)])

net = (
    Network()
    .add_nodes(precision=1e4)
    .add_nodes(value_children=0)
    .add_nodes(kind="categorical-state", n_categories=3)
)
print("after a 2-level HGF, binary nodes 3-5:", [s for s in steps(net) if s[0] in (3, 4, 5)])
u = np.hstack([np.zeros((5, 1)), np.eye(3)[[0, 1, 2, 0, 1]]])
try:
    net.input_data(u)
    print("input_data ran")
except KeyError as e:
    print("input_data: KeyError", e)
