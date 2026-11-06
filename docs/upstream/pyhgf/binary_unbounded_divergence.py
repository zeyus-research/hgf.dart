"""The unbounded volatility update (pyhgf's default) runs away on 3-level
binary HGFs (JAX and rshgf).

A binary 3-level HGF on reversal-learning outcomes: under the unbounded
update the volatility parent's mean x2 climbs steadily until x1 explodes
(|mu1| > 50, then values of order 1e5), on simulated reversals and on
pyhgf's own binary example data repeated. The standard update stays bounded
on the same data; eHGF stays finite but x2 drifts far down on long series.
"""

import numpy as np
from _common import first_nonfinite  # noqa: F401  (sets float64 JAX)

import gen_fixtures as g


def spec(mode: str, w2: float, w3: float) -> dict:
    s = g.binary_hgf(3, mode)
    s["nodes"][1]["params"]["tonic_volatility"] = w2
    s["nodes"][2]["params"]["tonic_volatility"] = w3
    return s


def reversal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = np.where((np.arange(n) // 50) % 2 == 0, 0.8, 0.2)
    return (rng.random(n) < p).astype(np.float64)


def runaway(mu1: np.ndarray) -> str:
    bad = np.flatnonzero(~(np.abs(mu1) < 50))
    return "bounded" if bad.size == 0 else f"|mu1|>50 at {int(bad[0])}"


datasets = {
    "reversal, 2000 trials": reversal(2000, 1),
    "pyhgf binary data x6": np.tile(g.binary_u(), 6),
}
for name, u in datasets.items():
    print(f"\n{name} ({len(u)} trials)")
    for w2, w3 in [(-3.0, -4.0), (-3.0, -6.0), (-4.0, -6.0)]:
        for mode in ("standard", "eHGF", "unbounded"):
            net = g.build_jax(spec(mode, w2, w3))
            net.input_data(input_data=u[:, None])
            tr = net.node_trajectories
            mu1 = np.asarray(tr[1]["mean"])
            mu2 = np.asarray(tr[2]["mean"])
            rs = g.build_rs(spec(mode, w2, w3))
            rs.input_data(input_data=u[:, None].tolist())
            rs_mu1 = np.asarray(rs.node_trajectories[1]["mean"])
            print(
                f"  w2={w2:<5} w3={w3:<5} {mode:<9}  JAX: {runaway(mu1):<18} "
                f"max mu2 {np.nanmax(mu2):7.1f}, min mu2 {np.nanmin(mu2):7.1f}"
                f"   rshgf: {runaway(rs_mu1)}"
            )
