"""``add_categorical_state`` ignores ``binary_parameters`` (P11).

The docstring of ``Network.add_nodes`` says ``binary_parameters`` sets up the
implied binary HGFs, but ``add_categorical_state`` builds them from a local
dict of defaults; the user's dict only ends up stored on the categorical
node. (The same docstring gives the node count as ``3n + 2``; it is
``2n + 2``.)
"""

from _common import steps  # noqa: F401  (sets float64 / CPU)
from pyhgf.model import Network

net = Network().add_nodes(
    kind="categorical-state",
    n_categories=3,
    binary_parameters={"tonic_volatility_2": -2.0, "mean_3": 1.0},
)
print("asked for tonic_volatility_2 = -2, got", [net.attributes[i]["tonic_volatility"] for i in (4, 5, 6)])
print("asked for mean_3 = 1, got", net.attributes[7]["mean"])
print("stored on the categorical node:", net.attributes[0]["binary_parameters"])
print("nodes for n = 3:", len(net.edges), "(docstring: 3n + 2 =", 3 * 3 + 2, ")")
