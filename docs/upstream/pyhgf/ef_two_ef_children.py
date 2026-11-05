"""A continuous node that is the value parent of two ef-state nodes crashes
``get_update_sequence``: it calls ``nodes_without_posterior_update.remove(i)``
once per ef child, and the second call raises."""

from _common import steps
from pyhgf.model import Network

net = Network().add_nodes(kind="ef-state", n_nodes=2).add_nodes(value_children=[0, 1])
try:
    print(steps(net))
except ValueError as e:
    print(f"ValueError: {e}")
