"""``dirichlet_kullback_leibler`` is NaN once the concentrations sum past
~171 (P13).

It takes ``jnp.log(gamma(a.sum()) / gamma(b.sum()))``: Γ overflows float64
for arguments above 171.6, and inf / inf is NaN. Differences of ``gammaln``
give the same value without overflow. In a categorical node, α grows when
the binary HGFs learn slowly. With pyhgf's defaults the sums stay well below
171 (15-120 over 500-2000 trials, ω2 from -4 to -12), but a confident prior on
the binary HGFs' value parents (precision 100) gets there.
"""

from _common import steps  # noqa: F401  (sets float64 / CPU)
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import digamma, gammaln
from pyhgf.math import dirichlet_kullback_leibler
from pyhgf.model import Network


def kl_gammaln(a1, a2):
    return (
        gammaln(a1.sum()) - gammaln(a2.sum())
        + jnp.sum(gammaln(a2) - gammaln(a1))
        + jnp.sum((a1 - a2) * (digamma(a1) - digamma(a1.sum())))
    )


a1, a2 = jnp.array([60.0, 70.0, 50.0]), jnp.array([61.0, 69.0, 52.0])
print("sum 180:  pyhgf", float(dirichlet_kullback_leibler(a1, a2)), " gammaln", float(kl_gammaln(a1, a2)))
a1, a2 = a1 / 2, a2 / 2
print("sum 90:   pyhgf", float(dirichlet_kullback_leibler(a1, a2)), " gammaln", float(kl_gammaln(a1, a2)))

# A categorical node whose binary HGFs learn slowly (ω2 = -8, precision 100
# on their value parents; set on the attributes, as add_nodes ignores
# binary_parameters, P11).
rng = np.random.default_rng(1)
u = np.eye(3)[rng.choice(3, size=500, p=[0.6, 0.3, 0.1])]
net = Network().add_nodes(kind="categorical-state", n_categories=3)
for i in (4, 5, 6):
    net.attributes[i]["tonic_volatility"] = -8.0
    net.attributes[i]["precision"] = 100.0
net.input_data(u)
alpha = np.asarray(net.node_trajectories[0]["alpha"])
kl = np.asarray(net.node_trajectories[0]["kl_divergence"])
big = alpha.sum(axis=1) > 171.6
print(f"omega2 = -8, precision 100: alpha sums up to {alpha.sum(axis=1).max():.0f}; "
      f"{big.sum()} of {len(u)} trials past 171.6; KL NaN on {np.isnan(kl).sum()}")
