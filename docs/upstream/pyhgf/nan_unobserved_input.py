"""A NaN input value on an unobserved trial (``observed = 0``) still reaches
the parents' beliefs (P22).

The ``observed`` mask is pyhgf's way to say "missing", and NaN is the
natural value for a missing cell (pandas reads empty cells as NaN). With the
mask at 0 the value should be ignored, but the input's prediction error is
computed from it and only then scaled by ``observed``: ``NaN * 0`` is NaN,
so the value parent's mean is NaN from that trial on. Any finite value in the
same cells gives the expected result.
"""

from _common import steps  # noqa: F401  (sets float64 / CPU)
import numpy as np
from pyhgf.model import Network
from pyhgf.rshgf import Network as RsNetwork

N = 40
observed = np.ones(N, dtype=int)
observed[2::5] = 0  # trials 2, 7, 12, ... are missing


def build(cls, kind):
    net = cls()
    net.add_nodes(kind=kind)
    net.add_nodes(value_children=0)
    net.add_nodes(volatility_children=1)
    return net


for kind, u in [
    ("continuous-state", (np.arange(N) * 0.37) % 2 - 1),
    ("binary-state", (np.arange(N) % 3 == 0).astype(float)),
]:
    for backend, cls in [("JAX  ", Network), ("rshgf", RsNetwork)]:
        means = {}
        for fill in ["NaN", "0.0"]:
            x = u.copy()
            x[observed == 0] = np.nan if fill == "NaN" else 0.0
            net = build(cls, kind)
            try:
                net.input_data(input_data=x, observed=observed)
                means[fill] = np.asarray(net.node_trajectories[1]["mean"])
            except Exception as e:  # noqa: BLE001
                means[fill] = f"{type(e).__name__}: {e}"
        a, b = means["NaN"], means["0.0"]
        if isinstance(a, str) or isinstance(b, str):
            print(f"{backend} {kind}: NaN fill -> {a!s:.60}; 0.0 fill -> {b!s:.60}")
            continue
        print(
            f"{backend} {kind}: value parent mean is NaN on "
            f"{int(np.isnan(a).sum())}/{N} trials with NaN at the missing "
            f"trials, {int(np.isnan(b).sum())}/{N} with 0.0"
        )
