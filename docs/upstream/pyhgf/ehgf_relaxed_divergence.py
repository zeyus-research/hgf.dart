"""Relaxed eHGF diverges on a 3-level continuous HGF (JAX and rshgf).

The eHGF precision update re-predicts the volatility child's volatility as
t·exp(κ μ_b + ω̃) without the Σ κ²/(2 π̂) terms of the relaxed prediction, so
at μ_b = μ̂_b it isn't the child's predicted precision. The increment is then
floored at 0 on most trials, the volatility parent never gains precision, and
the network goes NaN. Mean-field (no MGF terms in its prediction) is fine.
The patched run uses gen_fixtures' ``_MgfCorrection`` (the proposed fix).
"""

import numpy as np
from _common import first_nonfinite, usdchf

import gen_fixtures as g

y = usdchf()
nodes = range(4)


def report(label, traj):
    nan_at = first_nonfinite(traj, nodes)
    end = len(y) if nan_at is None else nan_at
    gain = np.asarray(traj[2]["precision"])[:end] > np.asarray(traj[2]["expected_precision"])[:end]
    print(
        f"{label:<34} first NaN: {nan_at!s:<5} "
        f"x2 gains precision on {int(gain.sum())}/{end} finite trials"
    )


for mean_field in (False, True):
    spec = g.continuous_3level("eHGF")
    spec["network"]["mean_field_updates"] = mean_field
    net = g.build_jax(spec)
    net.input_data(input_data=y[:, None])
    report(f"JAX eHGF, mean_field={mean_field}", net.node_trajectories)

spec = g.continuous_3level("eHGF")
rs = g.build_rs(spec)
rs.input_data(input_data=y[:, None].tolist())
report("rshgf eHGF, relaxed", dict(enumerate(rs.node_trajectories)))

net = g.build_jax(g.continuous_3level("eHGF"))
with g._MgfCorrection():
    net.input_data(input_data=y[:, None])
report("JAX eHGF, relaxed + MGF correction", net.node_trajectories)
