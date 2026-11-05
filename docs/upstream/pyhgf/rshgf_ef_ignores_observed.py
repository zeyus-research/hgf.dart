"""rshgf's ef-state update ignores ``observed``: a missing observation still
moves ``xis`` (JAX keeps it). rshgf also has only the univariate fixed-rate
update (no ``dimension``, ``learning``)."""

import _common  # noqa: F401, I001  (first: sets float64 JAX)

import numpy as np
from pyhgf.model import Network
from pyhgf.rshgf import Network as RsNetwork

x = [[1.0], [2.0], [3.0]]
observed = [1, 0, 1]

rs = RsNetwork()
rs.add_nodes(kind="ef-state")
rs.input_data(x, observed=[[float(o)] for o in observed])

jx = Network().add_nodes(kind="ef-state")
jx.input_data(np.asarray(x), observed=np.asarray(observed))

print("observed:", observed)
print("rshgf xis:", rs.node_trajectories[0]["xis"].tolist())
print("JAX   xis:", np.asarray(jx.node_trajectories[0]["xis"]).tolist())
