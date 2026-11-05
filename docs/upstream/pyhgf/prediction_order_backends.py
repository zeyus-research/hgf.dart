"""JAX and rshgf schedule independent predictions in different orders.

JAX adds every node whose parents are done in one pass over the remaining
nodes; rshgf takes the first such node and starts again. Values are the same
(independent nodes), but the two schedules can't be compared step by step.
"""

import _common  # noqa: F401

import gen_fixtures as g

spec = g.multi_structure("unbounded")
jax_net = g.build_jax(spec)
jax_net.create_belief_propagation_fn()
rs = g.build_rs(spec)
rs.set_update_sequence()
print("JAX:  ", [int(i) for i, _ in jax_net.update_sequence.prediction_steps])
print("rshgf:", [int(i) for i, name in rs.update_sequence if str(name).startswith("prediction_") and "error" not in str(name)])
