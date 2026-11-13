"""The noise prediction error of an input with noise parents counts a bias
parent's placeholder precision as posterior variance (P21, pyhgf #477).

``continuous_node_noise_prediction_error`` (JAX) and
``noise_prediction_error_continuous_input_node`` (rshgf) add
``(α g'(μ_b))² / π_b`` over every value parent b (``(t α g'(μ_b))² / π_b``
before #478). A constant-state (bias)
parent has ``precision`` 1 (a placeholder: ``add_constant_state`` keeps it at
1.0 "because some downstream code reads it directly", and sets
``expected_precision`` to ∞ so the prediction's Taylor term is 0). The bias
is known exactly, so it should add no variance; it adds ``α²`` (``(t α)²`` before #478).

A bias of strength α into the input is the same model as a tonic drift α on
the input, which has no such term: the two disagree from the first trial.
"""

from _common import steps  # noqa: F401  (sets float64 / CPU)
import numpy as np
from pyhgf.model import Network
from pyhgf.rshgf import Network as RsNetwork

ALPHA = 0.4
u = 0.4 + 0.3 * np.sin(np.arange(200) / 9)


def build(cls, bias: bool):
    net = cls(volatility_updates="standard")
    net.add_nodes(tonic_drift=0.0 if bias else ALPHA)  # 0: input
    net.add_nodes(value_children=0, tonic_volatility=-3.0)  # 1: value parent
    net.add_nodes(volatility_children=0, tonic_volatility=-4.0)  # 2: noise parent
    if bias:
        children = 0 if cls is RsNetwork else ([0], [ALPHA])
        net.add_nodes(kind="constant-state", value_children=children)
    return net


def noise_pe(net):
    traj = net.node_trajectories[0]
    if "temp" in traj:
        return np.asarray(traj["temp"]["volatility_prediction_error"])
    return np.asarray(traj["volatility_prediction_error"])


bias, drift = build(Network, True), build(Network, False)
bias.input_data(input_data=u)
drift.input_data(input_data=u)
eps_b, eps_d = noise_pe(bias), noise_pe(drift)
pihat = np.asarray(bias.node_trajectories[0]["expected_precision"])
print(f"JAX   trial 0: ε bias {eps_b[0]:.6f}, ε drift {eps_d[0]:.6f}, "
      f"(Δε / π̂_u) = {(eps_b[0] - eps_d[0]) / pihat[0]:.6f} = α² = {ALPHA**2}")
mu = [np.asarray(n.node_trajectories[2]["mean"]) for n in (bias, drift)]
print(f"JAX   noise parent mean after 200 trials: bias {mu[0][-1]:.4f}, drift {mu[1][-1]:.4f}")

# rshgf takes unit strengths from Python: α = 1.
ALPHA = 1.0
rb, rd = build(RsNetwork, True), build(RsNetwork, False)
rb.input_data(input_data=u.tolist())
rd.input_data(input_data=u.tolist())
eps_b, eps_d = noise_pe(rb), noise_pe(rd)
pihat = np.asarray(rb.node_trajectories[0]["expected_precision"])
print(f"rshgf trial 0: (Δε / π̂_u) = {(eps_b[0] - eps_d[0]) / pihat[0]:.6f} = α² = {ALPHA**2}")
