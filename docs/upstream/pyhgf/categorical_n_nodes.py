"""``add_nodes(kind="categorical-state", n_nodes=2)`` builds a broken graph
(P10).

``add_categorical_state`` computes ``binary_idxs`` once, from the node count
before inserting the categorical nodes, and gives the same list to every
categorical node. With two of them, the first categorical node's binary
parents start at the second categorical node's index, which is then turned
into a binary node, and the two implied networks overlap.
"""

from _common import steps  # noqa: F401  (sets float64 / CPU)
from pyhgf.model import Network

net = Network().add_nodes(kind="categorical-state", n_categories=3, n_nodes=2)
types = {1: "binary", 2: "continuous", 5: "categorical"}
for i, e in enumerate(net.edges):
    print(
        f"{i:2d} {types[e.node_type]:<11} value parents {e.value_parents}, "
        f"value children {e.value_children}"
    )
print("categorical nodes:", [i for i, e in enumerate(net.edges) if e.node_type == 5])
