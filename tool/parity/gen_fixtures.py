"""Generate pyhgf reference fixtures for the hgf.dart parity tests.

Each scenario is a declarative network spec (the same format the Dart tests use
to rebuild the network), an input series, and the reference trajectories from
pyhgf's JAX backend run in float64. When the scenario is expressible in the
Rust backend (``pyhgf.rshgf``), it is run there too and the max deviation
between the two backends is recorded, so a Dart mismatch can be attributed.

Run from this directory:

    CARGO_TARGET_DIR=/tmp/cargo-target uv sync   # once, builds pyhgf + rshgf
    uv run python gen_fixtures.py

Writes ``packages/hgf/test/fixtures/<scenario>.json``.
"""

from __future__ import annotations

import os

# Float64 everywhere: the reference must be at least as precise as the port.
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json  # noqa: E402
import math  # noqa: E402
from functools import partial  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import jit  # noqa: E402
import numpy as np  # noqa: E402

import pyhgf  # noqa: E402
from pyhgf import load_data  # noqa: E402
from pyhgf.model import Network  # noqa: E402
from pyhgf.response import (  # noqa: E402
    binary_softmax,
    binary_softmax_inverse_temperature,
    first_level_binary_surprise,
    first_level_gaussian_surprise,
    total_gaussian_surprise,
)
from pyhgf.rshgf import Network as RsNetwork  # noqa: E402
from pyhgf.updates.posterior.continuous import (  # noqa: E402
    posterior_update_precision_continuous_node as _precision_mod,
)
import pyhgf.utils.get_update_sequence  # noqa: E402,F401

# The module (pyhgf.utils re-exports a function under the same name).
_sequence_mod = sys.modules["pyhgf.utils.get_update_sequence"]
from pyhgf.updates.prediction_error import dirichlet as _dirichlet_mod  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYHGF_DIR = REPO.parent / "pyhgf"
OUT_DIR = REPO / "packages" / "hgf" / "test" / "fixtures"

# Fields compared by the Dart parity tests (when the node records them).
FIELDS = (
    "mean",
    "expected_mean",
    "precision",
    "expected_precision",
    "value_prediction_error",
    "volatility_prediction_error",
    "effective_precision",
    "nus",
    # categorical-state nodes
    "kl_divergence",
    "surprise",
    # dp-state nodes
    "alpha",
    "sensory_precision",
    "n_total",
    "n_active_cluster",
)

# Vector-valued fields of ef-state and categorical-state nodes (``mean`` too
# when the node is multivariate), written to the fixture's
# ``vector_trajectories``.
VECTOR_FIELDS = (
    "mean",
    "xis",
    "observation_ss",
    "alpha",
    # dp-state nodes
    "n",
    "expected_means",
    "expected_sigmas",
    "activated",
)

# pyhgf node types with vector-valued attributes (ef-state, dp-state,
# categorical-state).
VECTOR_NODE_TYPES = (3, 4, 5)

JAX_COUPLING_FNS = {
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    # Not named in rshgf: the Dart tests register them as CustomCoupling (with
    # closed-form derivatives), JAX differentiates them with autodiff.
    "sin": jnp.sin,
    "softplus": jax.nn.softplus,
}

# Coupling functions rshgf knows by name.
RS_COUPLING_FNS = {"linear", "identity", "relu", "sigmoid", "tanh", "leaky_relu", "gelu"}


def pyhgf_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PYHGF_DIR), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - informational only
        return "unknown"


# ---------------------------------------------------------------------------
# Scenario specs
# ---------------------------------------------------------------------------
#
# A node spec: {"kind", "n_nodes"?, "value_parents"?, "value_children"?,
# "volatility_parents"?, "volatility_children"?, "coupling_fn"?, "params"?}.
# Edge entries are lists of node indexes, or {"idxs": [...], "strengths": [...]}
# for non-unit coupling strengths (JAX only: rshgf's Python API takes indexes).


def continuous_2level(
    volatility_updates: str,
    mean_field: bool = False,
    input_precision: float = 1e4,
    tonic_volatility_1: float = -8.0,
) -> dict:
    return {
        "network": {
            "volatility_updates": volatility_updates,
            "mean_field_updates": mean_field,
        },
        "nodes": [
            {"kind": "continuous-state", "params": {"precision": input_precision}},
            {
                "kind": "continuous-state",
                "value_children": [0],
                "params": {"mean": 1.04, "tonic_volatility": tonic_volatility_1},
            },
            {
                "kind": "continuous-state",
                "volatility_children": [1],
                "params": {"tonic_volatility": -2.0},
            },
        ],
    }


def continuous_3level(volatility_updates: str) -> dict:
    spec = continuous_2level(volatility_updates)
    spec["nodes"].append(
        {
            "kind": "continuous-state",
            "volatility_children": [2],
            "params": {"tonic_volatility": -2.0},
        }
    )
    return spec


def continuous_input_volatility(volatility_updates: str) -> dict:
    return {
        "network": {"volatility_updates": volatility_updates},
        "nodes": [
            {"kind": "continuous-state"},
            {"kind": "continuous-state", "value_children": [0], "params": {"mean": 1.0}},
            {"kind": "continuous-state", "volatility_children": [0]},
        ],
    }


def noise_parent(volatility_updates: str, mean_field: bool = False, unit: bool = False) -> dict:
    """The input of pyhgf's ``test_noise_parent.py`` (#477): a value parent x
    and a noise parent a, each with a volatility parent. The tonic noise
    ε_u = 0.2 is a shift of ε_u / κ_u of the mean of a. Non-unit couplings
    (JAX only) unless ``unit``."""

    def edge(child: int, strength: float):
        return [child] if unit else {"idxs": [child], "strengths": [strength]}

    kappa_u = 1.0 if unit else 1.4
    return {
        "network": {"volatility_updates": volatility_updates, "mean_field_updates": mean_field},
        "nodes": [
            {"kind": "continuous-state"},  # 0: u
            {  # 1: x
                "kind": "continuous-state",
                "value_children": [0],
                "params": {"mean": 0.5, "tonic_volatility": -2.0},
            },
            {  # 2: x2
                "kind": "continuous-state",
                "volatility_children": edge(1, 0.7),
                "params": {"mean": 1.0, "tonic_volatility": -6.0},
            },
            {  # 3: a, the noise parent
                "kind": "continuous-state",
                "volatility_children": edge(0, kappa_u),
                "params": {"mean": 0.2 / kappa_u, "tonic_volatility": -4.0},
            },
            {  # 4: a2
                "kind": "continuous-state",
                "volatility_children": edge(3, 0.8),
                "params": {"mean": 1.0, "tonic_volatility": -6.0},
            },
        ],
    }


def noise_parent_mixed(volatility_updates: str) -> dict:
    """A noise parent (3) with a noise child (input 0), a state volatility
    child (2) and a value child (input 1): the eHGF and unbounded updates
    with noise children among their volatility children, and the unbounded
    effective prior with both value and noise terms."""
    return {
        "network": {"volatility_updates": volatility_updates},
        "nodes": [
            {"kind": "continuous-state", "n_nodes": 2, "params": {"precision": 4.0}},  # 0, 1
            {"kind": "continuous-state", "value_children": [0, 1], "params": {"tonic_volatility": -2.0}},  # 2
            {
                "kind": "continuous-state",
                "volatility_children": [0, 2],
                "value_children": [1],
                "params": {"tonic_volatility": -4.0},
            },  # 3
            {"kind": "continuous-state", "volatility_children": [3], "params": {"tonic_volatility": -5.0}},  # 4
        ],
    }


def noise_parent_two(volatility_updates: str) -> dict:
    """An input with two noise parents and a value parent coupled by tanh."""
    return {
        "network": {"volatility_updates": volatility_updates},
        "nodes": [
            {"kind": "continuous-state"},  # 0
            {
                "kind": "continuous-state",
                "value_children": [0],
                "coupling_fn": "tanh",
                "params": {"tonic_volatility": -3.0},
            },  # 1
            {"kind": "continuous-state", "volatility_children": [0], "params": {"tonic_volatility": -4.0}},  # 2
            {
                "kind": "continuous-state",
                "volatility_children": [0],
                "params": {"mean": -0.5, "tonic_volatility": -3.0},
            },  # 3
        ],
    }


def noise_parent_bias(strength: float | None = None) -> dict:
    """A bias (constant-state) value parent of an input with a noise parent:
    pyhgf adds the bias node's placeholder precision to the noise prediction
    error (P21, which hgf.dart corrects unless ``NetworkFixes.none``)."""
    spec = noise_parent_two("unbounded")
    spec["nodes"].append(
        {
            "kind": "constant-state",
            "value_children": [0] if strength is None else {"idxs": [0], "strengths": [strength]},
        }
    )
    return spec


def jumps(n_trials: int = 300, seed: int = 7) -> np.ndarray:
    """pyhgf's ``test_noise_parent._inputs``: draws from a Gaussian whose mean
    and standard deviation jump."""
    rng = np.random.default_rng(seed)
    trials = np.arange(n_trials)
    means = rng.normal(0.0, 3.0, n_trials // 50 + 1)[trials // 50]
    stds = rng.choice([0.5, 1.0, 2.5], n_trials // 75 + 1)[trials // 75]
    return means + stds * rng.normal(size=n_trials)


def binary_hgf(levels: int, volatility_updates: str = "unbounded") -> dict:
    nodes = [
        {"kind": "binary-state"},
        {"kind": "continuous-state", "value_children": [0], "params": {"tonic_volatility": -3.0}},
    ]
    if levels >= 3:
        nodes.append(
            {
                "kind": "continuous-state",
                "volatility_children": [1],
                "params": {"tonic_volatility": -2.0, "mean": 1.0},
            }
        )
    return {"network": {"volatility_updates": volatility_updates}, "nodes": nodes}


def nonlinear_coupling(fn: str) -> dict:
    return {
        "network": {"volatility_updates": "unbounded"},
        "nodes": [
            {"kind": "continuous-state", "params": {"precision": 100.0}},
            {
                "kind": "continuous-state",
                "value_children": [0],
                "coupling_fn": fn,
                "params": {"tonic_volatility": -4.0},
            },
            {
                "kind": "continuous-state",
                "volatility_children": [1],
                "params": {"tonic_volatility": -3.0},
            },
        ],
    }


def multi_structure(volatility_updates: str) -> dict:
    """Two inputs, a shared value parent, two value parents on one input, and one
    volatility parent with two volatility children."""
    return {
        "network": {"volatility_updates": volatility_updates},
        "nodes": [
            {"kind": "continuous-state", "n_nodes": 2, "params": {"precision": 10.0}},
            {"kind": "continuous-state", "value_children": [0, 1]},  # 2
            {"kind": "continuous-state", "value_children": [1]},  # 3
            {"kind": "continuous-state", "volatility_children": [2, 3]},  # 4
        ],
    }


def mgf_correction_specs() -> dict[str, dict]:
    """Relaxed eHGF with the MGF correction: the 3-level HGF (which diverges
    without it), the same with a second volatility parent of strength 0.5 on
    node 1 (the sum over parents and κ²), and mean-field (where it is a
    no-op)."""
    three = continuous_3level("eHGF")
    two_parents = continuous_3level("eHGF")
    two_parents["nodes"].append(
        {
            "kind": "continuous-state",
            "volatility_children": {"idxs": [1], "strengths": [0.5]},
            "params": {"tonic_volatility": -3.0},
        }
    )
    mean_field = continuous_3level("eHGF")
    mean_field["network"]["mean_field_updates"] = True
    out = {
        "continuous_3level_eHGF_mgf_correction": three,
        "continuous_3level_eHGF_two_volatility_parents_mgf_correction": two_parents,
        "continuous_3level_mean_field_eHGF_mgf_correction": mean_field,
    }
    for spec in out.values():
        spec["network"]["ehgf_mgf_correction"] = True
    return out


def coupling_strengths() -> dict:
    return {
        "network": {"volatility_updates": "unbounded"},
        "nodes": [
            {"kind": "continuous-state", "params": {"precision": 1e2}},
            {
                "kind": "continuous-state",
                "value_children": {"idxs": [0], "strengths": [0.5]},
            },
            {
                "kind": "continuous-state",
                "volatility_children": {"idxs": [1], "strengths": [2.0]},
                "params": {"tonic_volatility": -2.0},
            },
        ],
    }


def constant_bias(volatility_updates: str, strength: float | None = None) -> dict:
    """A continuous 2-level HGF whose value parent gets a constant drift input
    (a bias node): its prediction is pushed up by ``strength`` every trial and
    the posterior pulls it back to the data."""
    spec = continuous_2level(volatility_updates)
    spec["nodes"].append(
        {
            "kind": "constant-state",
            "value_children": [1]
            if strength is None
            else {"idxs": [1], "strengths": [strength]},
        }
    )
    return spec


def binary_bias() -> dict:
    """A binary 3-level HGF with a bias of +1 on the binary node's logit (the
    binary prediction ignores coupling strengths, and pyhgf ignores parameters
    on constant nodes)."""
    spec = binary_hgf(3)
    spec["nodes"].append({"kind": "constant-state", "value_children": [0]})
    return spec


def ef_state(
    dimension: int = 1,
    learning: str | None = None,
    nus: float | None = None,
    n_nodes: int = 1,
    volatility_updates: str = "unbounded",
) -> dict:
    """Exponential-family state node(s) on their own: fixed learning rate, or
    HGF learning (``hgf-1/2/3``), which adds one chain of continuous nodes per
    sufficient statistic."""
    params: dict = {}
    if dimension > 1:
        params["dimension"] = dimension
        params["distribution"] = "multivariate-normal"
    if learning is not None:
        params["learning"] = learning
    if nus is not None:
        params["nus"] = nus
    node: dict = {"kind": "ef-state"}
    if n_nodes > 1:
        node["n_nodes"] = n_nodes
    if params:
        node["params"] = params
    return {"network": {"volatility_updates": volatility_updates}, "nodes": [node]}


def ef_mixed_inputs() -> dict:
    """A continuous 2-level HGF next to a bivariate ef node (fixed rate) and a
    univariate one with hgf-1 learning: input columns of different widths."""
    spec = continuous_2level("unbounded")
    spec["nodes"] = spec["nodes"][:2]
    spec["nodes"].append(
        {
            "kind": "ef-state",
            "params": {"dimension": 2, "distribution": "multivariate-normal"},
        }
    )
    spec["nodes"].append({"kind": "ef-state", "params": {"learning": "hgf-1"}})
    return spec


def categorical_state(
    n_categories: int | None = 3,
    volatility_updates: str = "unbounded",
    mean_field: bool = False,
) -> dict:
    """A categorical-state node on its own: pyhgf adds one binary HGF per
    category (binary node, continuous value parent) and a shared volatility
    parent, 2n + 2 nodes in all. ``None`` keeps pyhgf's default of 4."""
    node: dict = {"kind": "categorical-state"}
    if n_categories is not None:
        node["params"] = {"n_categories": n_categories}
    network: dict = {"volatility_updates": volatility_updates}
    if mean_field:
        network["mean_field_updates"] = True
    return {"network": network, "nodes": [node]}


def categorical_mixed() -> dict:
    """A categorical node (nodes 0-7) followed by a continuous 2-level HGF
    (nodes 8-9). The other order crashes pyhgf: the binary nodes then inherit
    the continuous parent's posterior update (upstream P7)."""
    spec = categorical_state(3)
    spec["nodes"] += [
        {"kind": "continuous-state", "params": {"precision": 1e4}},
        {"kind": "continuous-state", "value_children": [8], "params": {"tonic_volatility": -6.0}},
    ]
    return spec


def dp_state(
    batch_size: int | None = 3,
    alpha: float | None = 0.1,
    sensory_precision: float | None = None,
    nus: float | None = 15.0,
) -> dict:
    """A Dirichlet process node (node 0) and its clusters, one univariate
    ef-state node each (nodes 1..batch_size), as in pyhgf's tests. ``None``
    keeps pyhgf's default (batch_size 10, alpha 1, sensory precision 1,
    nus 3)."""
    params: dict = {}
    if batch_size is not None:
        params["batch_size"] = batch_size
    if alpha is not None:
        params["alpha"] = alpha
    if sensory_precision is not None:
        params["sensory_precision"] = sensory_precision
    cluster: dict = {
        "kind": "ef-state",
        "n_nodes": batch_size or 10,
        "value_children": [0],
    }
    if nus is not None:
        cluster["params"] = {"nus": nus}
    return {"network": {}, "nodes": [{"kind": "dp-state", "params": params}, cluster]}


def dp_mixed() -> dict:
    """A dp-state node with 3 clusters (nodes 0-3) followed by a continuous
    2-level HGF (nodes 4-5). The other order crashes pyhgf: the clusters then
    inherit the continuous parent's posterior update (upstream P7)."""
    spec = dp_state()
    spec["nodes"] += [
        {"kind": "continuous-state", "params": {"precision": 1e4}},
        {"kind": "continuous-state", "value_children": [4], "params": {"tonic_volatility": -6.0}},
    ]
    return spec


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _edge_arg(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return (value["idxs"], value["strengths"])
    return value


def _jax_edge(value):
    """JAX wants a single int, a list, or an (idxs, strengths) tuple."""
    value = _edge_arg(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def build_jax(spec: dict) -> Network:
    net = Network(
        volatility_updates=spec["network"].get("volatility_updates", "unbounded"),
        mean_field_updates=spec["network"].get("mean_field_updates", False),
    )
    for node in spec["nodes"]:
        kwargs = dict(node.get("params", {}))
        if "coupling_fn" in node:
            kwargs["coupling_fn"] = (JAX_COUPLING_FNS[node["coupling_fn"]],)
        net.add_nodes(
            kind=node["kind"],
            n_nodes=node.get("n_nodes", 1),
            value_parents=_jax_edge(node.get("value_parents")),
            value_children=_jax_edge(node.get("value_children")),
            volatility_parents=_jax_edge(node.get("volatility_parents")),
            volatility_children=_jax_edge(node.get("volatility_children")),
            **kwargs,
        )
    return net


# ---------------------------------------------------------------------------
# Proposed upstream fix: the eHGF MGF correction (hgf.dart's
# `Network.ehgfMgfCorrection`, spec key "ehgf_mgf_correction")
# ---------------------------------------------------------------------------
#
# pyhgf's relaxed eHGF re-predicts a volatility child's volatility as
# t·exp(κ μ_b + ω̃) and leaves out the Σ_p κ_p²/(2 π̂_p) terms that the relaxed
# prediction of that child included. The fixtures with this key are pyhgf's
# JAX backend with the patch below applied: they are a reference for the
# proposed change, not for pyhgf as released.


def _ehgf_increment_mgf_corrected(attributes, edges, node_idx, volatility_child_idx, volatility_coupling):
    """``_ehgf_volatility_precision_increment`` with the MGF terms in ω̃."""
    m = _precision_mod
    mean = attributes[node_idx]["mean"]
    tonic_volatility = m._effective_tonic_volatility(attributes, edges, node_idx, volatility_child_idx)
    for parent_idx, coupling in zip(
        edges[volatility_child_idx].volatility_parents,
        attributes[volatility_child_idx]["volatility_coupling_parents"],
    ):
        tonic_volatility += coupling**2 / (2.0 * attributes[parent_idx]["expected_precision"])
    previous_variance = attributes[volatility_child_idx]["temp"]["current_variance"] + m._value_coupling_variance(
        attributes, volatility_child_idx
    )
    predicted_volatility = attributes[-1]["time_step"] * jnp.exp(volatility_coupling * mean + tonic_volatility)
    expected_precision = 1.0 / (previous_variance + predicted_volatility)
    effective_precision = predicted_volatility * expected_precision
    volatility_error_weight = (predicted_volatility - previous_variance) * expected_precision
    volatility_prediction_error = (
        1.0 / attributes[volatility_child_idx]["precision"]
        + (attributes[volatility_child_idx]["mean"] - attributes[volatility_child_idx]["expected_mean"]) ** 2
    ) * expected_precision - 1.0
    increment = jnp.maximum(
        0.0,
        0.5
        * volatility_coupling**2
        * effective_precision
        * (effective_precision + volatility_error_weight * volatility_prediction_error),
    )
    return increment * attributes[volatility_child_idx]["observed"]


@partial(jit, static_argnames=("edges", "node_idx"))
def _precision_update_ehgf_mgf_corrected(attributes, edges, node_idx):
    """``precision_update_ehgf`` (relaxed only) with the corrected increment.

    The mean-field eHGF update is left alone: its prediction has no MGF terms.
    """
    m = _precision_mod
    pwpe = m._value_coupling_pwpe(attributes, edges, node_idx, m._relaxed_child_factors)
    pwpe += m._volatility_coupling_pwpe(attributes, edges, node_idx, _ehgf_increment_mgf_corrected)
    return m._finalize_precision(attributes[node_idx]["expected_precision"], pwpe)


class _MgfCorrection:
    """Swap pyhgf's relaxed eHGF precision update for the corrected one (and
    clear JAX's caches on both sides, so no trace of the other version is
    reused)."""

    def __enter__(self):
        self._orig = _precision_mod.precision_update_ehgf
        _precision_mod.precision_update_ehgf = _precision_update_ehgf_mgf_corrected
        jax.clear_caches()

    def __exit__(self, *exc):
        _precision_mod.precision_update_ehgf = self._orig
        jax.clear_caches()
        return False


# ---------------------------------------------------------------------------
# Proposed upstream fix: dp-state nodes skip missing observations (P14)
# ---------------------------------------------------------------------------
#
# pyhgf's `dirichlet_node_prediction_error` never reads the node's `observed`
# flag, so a missing observation is clustered like any other. hgf.dart skips
# it: no cluster observes it and no count changes. The fixtures with spec key
# "dp_skip_missing" come from pyhgf's JAX backend with that change.


def _dp_prediction_error_skip_missing(edges, attributes, node_idx, **args):
    """``dirichlet_node_prediction_error``, or, when the node is unobserved,
    only its clusters' ``observed = 0``."""
    observed = attributes[node_idx]["observed"]
    updated = _dirichlet_mod.dirichlet_node_prediction_error(
        edges=edges, attributes=attributes, node_idx=node_idx
    )
    skipped = {k: dict(v) if isinstance(v, dict) else v for k, v in attributes.items()}
    for parent_idx in edges[node_idx].value_parents:
        skipped[parent_idx]["observed"] = jnp.zeros_like(updated[parent_idx]["observed"])
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(observed, a, b).astype(jnp.asarray(a).dtype), updated, skipped
    )


_dp_prediction_error_skip_missing.__name__ = "dirichlet_node_prediction_error"


class _DpSkipMissing:
    def __enter__(self):
        self._orig = _sequence_mod.dirichlet_node_prediction_error
        _sequence_mod.dirichlet_node_prediction_error = _dp_prediction_error_skip_missing
        jax.clear_caches()

    def __exit__(self, *exc):
        _sequence_mod.dirichlet_node_prediction_error = self._orig
        jax.clear_caches()
        return False


def _dp_skip_missing(spec: dict) -> bool:
    return bool(spec["network"].get("dp_skip_missing", False))


def _mgf_corrected(spec: dict) -> bool:
    return bool(spec["network"].get("ehgf_mgf_correction", False))


def rs_supported(spec: dict, observed=None) -> bool:
    if _mgf_corrected(spec):
        return False  # rshgf is unpatched
    for node in spec["nodes"]:
        if node["kind"] == "ef-state" and (
            node.get("params") or node.get("n_nodes", 1) > 1 or observed is not None
        ):
            # rshgf's ef-state has only the univariate fixed-rate update,
            # without missing observations.
            return False
        if node["kind"] in ("categorical-state", "dp-state"):
            return False  # neither node kind exists in rshgf
        if node.get("coupling_fn", "linear") not in RS_COUPLING_FNS:
            return False
        for key in ("value_parents", "value_children", "volatility_parents", "volatility_children"):
            if isinstance(node.get(key), dict):
                return False
    return True


def build_rs(spec: dict) -> RsNetwork:
    net = RsNetwork(
        volatility_updates=spec["network"].get("volatility_updates", "unbounded"),
        mean_field_updates=spec["network"].get("mean_field_updates", False),
    )
    for node in spec["nodes"]:
        net.add_nodes(
            kind=node["kind"],
            n_nodes=node.get("n_nodes", 1),
            value_parents=node.get("value_parents"),
            value_children=node.get("value_children"),
            volatility_parents=node.get("volatility_parents"),
            volatility_children=node.get("volatility_children"),
            coupling_fn=node.get("coupling_fn"),
            **{k: float(v) for k, v in node.get("params", {}).items()},
        )
    return net


def _clean(x: float):
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def _series(arr) -> list:
    return [_clean(v) for v in np.asarray(arr, dtype=np.float64).reshape(-1)]


def jax_trajectories(net: Network) -> tuple[dict, dict]:
    """Scalar fields per node, and the vector fields of ef/categorical nodes
    (``{node: {field: [[...] per trial]}}``)."""
    out, vectors = {}, {}
    for idx, traj in net.node_trajectories.items():
        if idx == -1:
            continue
        fields, vfields = {}, {}
        for field in FIELDS:
            if field in traj:
                values = np.asarray(traj[field], dtype=np.float64)
                if values.ndim == 1:
                    fields[field] = _series(values)
            elif "temp" in traj and field in traj["temp"]:
                fields[field] = _series(traj["temp"][field])
        if net.edges[idx].node_type in VECTOR_NODE_TYPES:
            for field in VECTOR_FIELDS:
                if field not in traj:
                    continue
                values = np.asarray(traj[field], dtype=np.float64)
                if values.ndim == 2:
                    vfields[field] = [_series(row) for row in values]
        out[str(idx)] = fields
        if vfields:
            vectors[str(idx)] = vfields
    return out, vectors


def rs_trajectories(net: RsNetwork) -> dict:
    out = {}
    for idx, traj in enumerate(net.node_trajectories):
        out[str(idx)] = {f: _series(traj[f]) for f in FIELDS if f in traj}
    return out


def max_rel_diff(a: dict, b: dict) -> dict:
    diffs = {}
    for idx, fields in a.items():
        for field, xs in fields.items():
            if idx not in b or field not in b[idx]:
                continue
            xa = np.array([float(v) for v in xs])
            xb = np.array([float(v) for v in b[idx][field]])
            both_nan = np.isnan(xa) & np.isnan(xb)
            # Relative difference, with an absolute floor so values at ~0 (where
            # a 1-ulp difference would read as a large relative one) count as equal.
            denom = np.maximum(np.abs(xa), np.abs(xb))
            absdiff = np.abs(xa - xb)
            rel = np.where(
                both_nan | (absdiff <= 1e-14),
                0.0,
                absdiff / np.maximum(denom, 1e-300),
            )
            diffs[f"{idx}.{field}"] = float(np.nanmax(rel)) if rel.size else 0.0
    return diffs


def surprise_reference(net: Network, spec: dict, name: str) -> dict:
    """Per-node and total surprise (``to_pandas``), plus the response functions
    of ``pyhgf.response`` that apply to the scenario.

    Binary scenarios also get simulated decisions (seeded per scenario) for the
    softmax response models; they are stored with the fixture.
    """
    df = net.to_pandas()
    node_surprise = {
        str(i): _series(df[f"x_{i}_surprise"])
        for i in range(len(net.edges))
        if f"x_{i}_surprise" in df
    }
    out = {
        "node_surprise": node_surprise,
        "total_surprise": _series(df["total_surprise"]),
        "responses": None,
        "response_functions": {},
    }
    kinds = [net.edges[i].node_type for i in range(len(net.edges))]
    rf = out["response_functions"]
    if kinds[0] == 2 and len(kinds) > 1 and kinds[1] == 2 and 3 not in kinds:
        rf["first_level_gaussian_surprise"] = _series(
            first_level_gaussian_surprise(net)
        )
    if len(kinds) > 1 and 2 in kinds:
        rf["total_gaussian_surprise"] = _series(total_gaussian_surprise(net))
    if kinds[0] == 1:
        rf["first_level_binary_surprise"] = _series(first_level_binary_surprise(net))
        seed = sum(ord(c) for c in name)
        rng = np.random.default_rng(seed)
        p = np.asarray(net.node_trajectories[0]["expected_mean"], dtype=np.float64)
        responses = rng.binomial(n=1, p=p).astype(np.float64)
        out["responses"] = _series(responses)
        rf["binary_softmax"] = _series(
            binary_softmax(net, response_function_inputs=jnp.asarray(responses))
        )
        rf["binary_softmax_inverse_temperature"] = {
            "inverse_temperature": 2.5,
            "surprise": _series(
                binary_softmax_inverse_temperature(
                    net,
                    response_function_inputs=jnp.asarray(responses),
                    response_function_parameters=2.5,
                )
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def usdchf() -> np.ndarray:
    return np.asarray(load_data("continuous"), dtype=np.float64)


def binary_u() -> np.ndarray:
    u, _ = load_data("binary")
    return np.asarray(u, dtype=np.float64)


def categories(n_trials: int, n_categories: int, seed: int = 7) -> np.ndarray:
    """One-hot draws (``n_trials x n_categories``) from a categorical
    distribution that changes twice: a learning, a reversal, a flat phase."""
    rng = np.random.default_rng(seed)
    p = np.linspace(1.0, 0.2, n_categories)
    probs = [p / p.sum(), p[::-1] / p.sum(), np.full(n_categories, 1.0 / n_categories)]
    third = -(-n_trials // 3)
    draws = np.concatenate(
        [rng.choice(n_categories, size=third, p=q) for q in probs]
    )[:n_trials]
    return np.eye(n_categories)[draws]


def mixture(n_trials: int, seed: int = 11) -> np.ndarray:
    """Draws from three well-separated normal clusters, in random order, with a
    fourth appearing halfway through."""
    rng = np.random.default_rng(seed)
    centres = np.array([-6.0, 0.5, 7.0])
    first = rng.choice(3, size=n_trials // 2)
    second = rng.choice(4, size=n_trials - n_trials // 2)
    centres4 = np.append(centres, 15.0)
    x = np.concatenate([centres[first], centres4[second]])
    return x + rng.normal(0.0, 1.0, size=n_trials)


def missing_mask(n: int, n_inputs: int = 1) -> np.ndarray:
    mask = np.ones((n, n_inputs))
    mask[::7, 0] = 0.0
    mask[100:120, 0] = 0.0
    return mask


def irregular_time_steps(n: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.uniform(0.25, 2.0, size=n)


def scenarios() -> list[dict]:
    y = usdchf()
    u = binary_u()
    two_inputs = np.stack([y, y[::-1] * 0.5 + 0.3], axis=1)
    out = []
    for mode in ("unbounded", "eHGF", "standard"):
        out.append({"name": f"continuous_2level_{mode}", "spec": continuous_2level(mode), "inputs": y})
        out.append({"name": f"continuous_3level_{mode}", "spec": continuous_3level(mode), "inputs": y})
        out.append(
            {
                "name": f"continuous_input_volatility_{mode}",
                "spec": continuous_input_volatility(mode),
                "inputs": y,
            }
        )
        out.append({"name": f"multi_structure_{mode}", "spec": multi_structure(mode), "inputs": two_inputs})
    # Noise parents of continuous input nodes (pyhgf #477).
    # Scaled to keep every node well conditioned in every mode: the tanh
    # parent runs away (g' -> 0) once the inputs leave tanh's range.
    j = jumps()
    jj = np.stack([j / 3.0, 0.5 * np.tanh(j / 3.0)], axis=1)
    jt = 0.3 * np.tanh(j / 3.0)
    for mode in ("unbounded", "eHGF", "standard"):
        out.append({"name": f"noise_parent_{mode}", "spec": noise_parent(mode), "inputs": j})
        out.append(
            {
                "name": f"noise_parent_mean_field_{mode}",
                "spec": noise_parent(mode, mean_field=True),
                "inputs": j,
            }
        )
        out.append({"name": f"noise_parent_unit_{mode}", "spec": noise_parent(mode, unit=True), "inputs": j})
        out.append({"name": f"noise_parent_mixed_{mode}", "spec": noise_parent_mixed(mode), "inputs": jj})
        out.append({"name": f"noise_parent_two_{mode}", "spec": noise_parent_two(mode), "inputs": jt})
    out.append(
        {
            "name": "noise_parent_missing",
            "spec": noise_parent_mixed("unbounded"),
            "inputs": jj,
            "observed": missing_mask(len(j), 2),
        }
    )
    out.append({"name": "noise_parent_bias", "spec": noise_parent_bias(), "inputs": jt + 1.0})
    out.append(
        {
            "name": "noise_parent_bias_strength",
            "spec": noise_parent_bias(0.4),
            "inputs": jt + 0.4,
            "time_steps": irregular_time_steps(len(j)),
        }
    )
    out.append(
        {
            "name": "noise_parent_time_steps",
            "spec": noise_parent_two("eHGF"),
            "inputs": jt,
            "time_steps": irregular_time_steps(len(j)),
        }
    )
    for mode in ("unbounded", "eHGF", "standard"):
        out.append(
            {
                "name": f"continuous_2level_mean_field_{mode}",
                "spec": continuous_2level(mode, mean_field=True),
                "inputs": y,
            }
        )
    out.append(
        {
            "name": "continuous_2level_missing",
            "spec": continuous_2level("unbounded"),
            "inputs": y,
            "observed": missing_mask(len(y)),
        }
    )
    out.append(
        {
            "name": "continuous_2level_ehgf_missing",
            # Less extreme than the default 2-level spec: with π_u = 1e4 and ω1 = -8
            # the missing-observation blocks drive μ2 to ~-40 and the trajectory
            # becomes ill-conditioned (backends that agree to 1e-12 drift apart).
            "spec": continuous_2level("eHGF", input_precision=1e3, tonic_volatility_1=-6.0),
            "inputs": y,
            "observed": missing_mask(len(y)),
        }
    )
    out.append(
        {
            "name": "continuous_2level_time_steps",
            "spec": continuous_2level("unbounded"),
            "inputs": y,
            "time_steps": irregular_time_steps(len(y)),
        }
    )
    out.append(
        {
            "name": "continuous_2level_unbounded_overflow",
            # ω1 = 706 puts log S = log t + κμ̂2 + ω1 above 700 for the first
            # trials, where hgf.dart's unbounded update leaves its linear-space
            # form (S = exp(log S) would overflow) for pyhgf's log-space one.
            "spec": continuous_2level("unbounded", tonic_volatility_1=706.0),
            "inputs": y,
        }
    )
    # The proposed eHGF MGF correction (see _ehgf_increment_mgf_corrected).
    for name, spec in mgf_correction_specs().items():
        out.append({"name": name, "spec": spec, "inputs": y})
    out.append({"name": "binary_2level", "spec": binary_hgf(2), "inputs": u})
    for mode in ("unbounded", "eHGF", "standard"):
        out.append({"name": f"binary_3level_{mode}", "spec": binary_hgf(3, mode), "inputs": u})
    out.append(
        {
            "name": "binary_3level_missing",
            "spec": binary_hgf(3),
            "inputs": u,
            "observed": missing_mask(len(u)),
        }
    )
    for fn in ("tanh", "sigmoid", "sin", "softplus"):
        out.append({"name": f"coupling_fn_{fn}", "spec": nonlinear_coupling(fn), "inputs": y - y.mean()})
    out.append({"name": "coupling_strengths", "spec": coupling_strengths(), "inputs": y})
    for mode in ("unbounded", "eHGF", "standard"):
        out.append(
            {"name": f"constant_bias_{mode}", "spec": constant_bias(mode), "inputs": y}
        )
    out.append(
        {
            "name": "constant_bias_strength",
            "spec": constant_bias("unbounded", strength=0.002),
            "inputs": y,
        }
    )
    out.append({"name": "binary_bias", "spec": binary_bias(), "inputs": u})

    # Exponential-family state nodes.
    three_cols = np.stack([y, y[::-1] * 0.5 + 0.3, (y - y.mean()) * 10.0], axis=1)
    out.append({"name": "ef_normal_fixed", "spec": ef_state(), "inputs": y})
    out.append(
        {
            "name": "ef_normal_fixed_nus_missing",
            "spec": ef_state(nus=10.0),
            "inputs": y,
            "observed": missing_mask(len(y)),
        }
    )
    out.append({"name": "ef_mvn3_fixed", "spec": ef_state(dimension=3), "inputs": three_cols})
    for learning in ("hgf-1", "hgf-2", "hgf-3"):
        out.append(
            {
                "name": f"ef_normal_{learning.replace('-', '')}",
                "spec": ef_state(learning=learning),
                "inputs": y,
            }
        )
    out.append(
        {
            "name": "ef_normal_hgf2_missing",
            "spec": ef_state(learning="hgf-2"),
            "inputs": y,
            "observed": missing_mask(len(y)),
        }
    )
    out.append(
        {
            "name": "ef_normal_hgf2_eHGF",
            "spec": ef_state(learning="hgf-2", volatility_updates="eHGF"),
            "inputs": y,
        }
    )
    out.append(
        {
            "name": "ef_mvn2_hgf2",
            "spec": ef_state(dimension=2, learning="hgf-2"),
            "inputs": two_inputs,
        }
    )
    out.append(
        {
            "name": "ef_two_nodes_hgf1",
            "spec": ef_state(learning="hgf-1", n_nodes=2),
            "inputs": two_inputs,
        }
    )
    out.append(
        {
            "name": "ef_mixed_inputs",
            "spec": ef_mixed_inputs(),
            "inputs": np.stack([y, y, y[::-1] * 0.5 + 0.3, y[::-1]], axis=1),
            "observed": missing_mask(len(y), 3),
        }
    )

    # Categorical state nodes (JAX only: rshgf has no nodalised version).
    cat3 = categories(300, 3)
    out.append({"name": "categorical_3", "spec": categorical_state(3), "inputs": cat3})
    out.append(
        {"name": "categorical_4_default", "spec": categorical_state(None), "inputs": categories(300, 4)}
    )
    for mode in ("eHGF", "standard"):
        out.append(
            {"name": f"categorical_3_{mode}", "spec": categorical_state(3, mode), "inputs": cat3}
        )
    out.append(
        {
            "name": "categorical_3_mean_field",
            "spec": categorical_state(3, mean_field=True),
            "inputs": cat3,
        }
    )
    out.append(
        {
            "name": "categorical_3_missing",
            "spec": categorical_state(3),
            "inputs": cat3,
            "observed": missing_mask(len(cat3)),
        }
    )
    # Dirichlet process nodes (JAX only: rshgf has none).
    xs = mixture(160)
    out.append({"name": "dp_3_clusters", "spec": dp_state(), "inputs": xs})
    out.append(
        {"name": "dp_default", "spec": dp_state(None, None, None, None), "inputs": xs}
    )
    out.append(
        # Two clusters and a high α: more clusters are opened than exist.
        {"name": "dp_full", "spec": dp_state(2, 5.0), "inputs": xs[:60]}
    )
    out.append(
        {
            "name": "dp_sensory_precision",
            "spec": dp_state(4, 0.5, sensory_precision=3.0, nus=8.0),
            "inputs": xs,
        }
    )
    dp_missing = dp_state(4, 0.5)
    dp_missing["network"]["dp_skip_missing"] = True
    mask = missing_mask(len(xs))
    out.append(
        {
            "name": "dp_missing",
            "spec": dp_missing,
            # Missing values are NaN: pyhgf (unpatched) would open a NaN cluster.
            "inputs": np.where(mask[:, 0] == 0.0, np.nan, xs),
            "observed": mask,
        }
    )
    out.append(
        {
            "name": "dp_mixed",
            "spec": dp_mixed(),
            "inputs": np.stack([xs, y[: len(xs)]], axis=1),
        }
    )
    out.append(
        {
            "name": "categorical_mixed",
            "spec": categorical_mixed(),
            "inputs": np.hstack([cat3, y[: len(cat3), None]]),
            "observed": missing_mask(len(cat3), 2)[:, ::-1],
        }
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _fn_name(fn) -> str:
    while hasattr(fn, "func"):  # jax.tree_util.Partial / functools.partial
        fn = fn.func
    return getattr(fn, "__name__", repr(fn))


def run(scenario: dict, commit: str) -> dict:
    spec = scenario["spec"]
    inputs = np.asarray(scenario["inputs"], dtype=np.float64)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    n_time, n_inputs = inputs.shape
    observed = scenario.get("observed")
    time_steps = scenario.get("time_steps")

    jnet = build_jax(spec)
    jax_kwargs = {"input_data": inputs}
    if time_steps is not None:
        jax_kwargs["time_steps"] = np.asarray(time_steps)
    if observed is not None:
        jax_kwargs["observed"] = np.asarray(observed).astype(int)
    if _mgf_corrected(spec):
        with _MgfCorrection():
            jnet.input_data(**jax_kwargs)
    elif _dp_skip_missing(spec):
        with _DpSkipMissing():
            jnet.input_data(**jax_kwargs)
    else:
        jnet.input_data(**jax_kwargs)
    reference, vector_reference = jax_trajectories(jnet)

    fixture = {
        "name": scenario["name"],
        "pyhgf_commit": commit,
        "pyhgf_version": pyhgf.__version__,
        "reference": (
            "pyhgf JAX with the proposed eHGF MGF correction (not released pyhgf)"
            if _mgf_corrected(spec)
            else "pyhgf JAX with dp-state nodes skipping missing observations (not released pyhgf)"
            if _dp_skip_missing(spec)
            else "pyhgf JAX"
        ),
        "jax_x64": bool(jax.config.jax_enable_x64),
        "spec": spec,
        "inputs": [[_clean(v) for v in row] for row in inputs],
        "observed": None if observed is None else [[_clean(v) for v in row] for row in np.asarray(observed)],
        "time_steps": None if time_steps is None else _series(time_steps),
        "input_idxs": [int(i) for i in jnet.input_idxs],
        "update_sequence": {
            "prediction": [[int(i), _fn_name(fn)] for i, fn in jnet.update_sequence.prediction_steps],
            "update": [[int(i), _fn_name(fn)] for i, fn in jnet.update_sequence.update_steps],
        },
        "trajectories": reference,
        "vector_trajectories": vector_reference,
        "surprise": surprise_reference(jnet, spec, scenario["name"]),
    }

    if rs_supported(spec, observed):
        rnet = build_rs(spec)
        rs_kwargs = {"input_data": inputs.tolist()}
        if time_steps is not None:
            rs_kwargs["time_steps"] = list(map(float, time_steps))
        if observed is not None:
            rs_kwargs["observed"] = np.asarray(observed, dtype=float).tolist()
        rnet.input_data(**rs_kwargs)
        rs = rs_trajectories(rnet)
        diffs = max_rel_diff(reference, rs)
        fixture["rshgf_max_rel_diff"] = max(diffs.values()) if diffs else None
        fixture["rshgf_rel_diffs"] = {k: v for k, v in diffs.items() if v > 1e-9}
        fixture["rshgf_update_sequence"] = [[int(i), str(name)] for i, name in rnet.update_sequence]
        # Where the two pyhgf backends disagree beyond the parity tolerance,
        # keep the rshgf trajectories too: the Dart port follows rshgf's
        # arithmetic, so it is checked against both.
        if fixture["rshgf_max_rel_diff"] is not None and fixture["rshgf_max_rel_diff"] > 1e-9:
            fixture["rshgf_trajectories"] = rs
    else:
        fixture["rshgf_max_rel_diff"] = None
    del n_time, n_inputs
    return fixture


DP_CANDIDATE_CASES = [
    # value, sensory precision, cluster means, cluster sigmas
    (5.0, 1.0, [0.0, -5.0], [1.0, 3.0]),  # pyhgf's own test
    (0.0, 1.0, [0.0], [1.0]),
    (-12.0, 2.0, [0.0, 3.0, 8.0], [1.0, 0.5, 2.0]),
    (3.3, 0.5, [0.0] * 10, [1.0] * 10),  # pyhgf's defaults
    (25.0, 1.0, [-1.0, 1.0], [0.7, 4.0]),
    (1.2, 3.0, [1.0, 1.5, -2.0, 6.0], [0.3, 1.0, 1.0, 2.5]),
]


def dp_reference(commit: str) -> dict:
    """pyhgf's fixed dp-state cluster proposals (every 50th, and the sums of
    all 20000) and some ``get_candidate`` results, to check those parts of the
    port on their own."""
    from jax import random

    mus, sigmas, _ = _dirichlet_mod.likely_cluster_proposal(
        mean_mu_G0=0.0,
        sigma_mu_G0=10.0,
        sigma_pi_G0=3.0,
        expected_mean=jnp.zeros(1),
        expected_sigma=jnp.ones(1),
        key=random.key(42),
        n_samples=20_000,
    )
    mus, sigmas = np.asarray(mus), np.asarray(sigmas)
    candidates = []
    for value, precision, means, sds in DP_CANDIDATE_CASES:
        mean, sigma = _dirichlet_mod.get_candidate(
            value=value,
            sensory_precision=precision,
            expected_mean=jnp.asarray(means),
            expected_sigma=jnp.asarray(sds),
        )
        candidates.append(
            {
                "value": value,
                "sensory_precision": precision,
                "expected_means": means,
                "expected_sigmas": sds,
                "mean": float(mean),
                "sigma": float(sigma),
            }
        )
    return {
        "pyhgf_commit": commit,
        "stride": 50,
        "mu": _series(mus[::50]),
        "sigma": _series(sigmas[::50]),
        "mu_sum": float(mus.sum()),
        "sigma_sum": float(sigmas.sum()),
        "candidates": candidates,
    }


def main(argv: list[str]) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = pyhgf_commit()
    only = set(argv[1:])
    if not only or "dp_reference" in only:
        (OUT_DIR / "dp").mkdir(exist_ok=True)
        path = OUT_DIR / "dp" / "dp_reference.json"
        path.write_text(json.dumps(dp_reference(commit), separators=(",", ":"), allow_nan=False))
        print(f"{'dp_reference':<42} (proposals and candidates)")
    for scenario in scenarios():
        if only and scenario["name"] not in only:
            continue
        fixture = run(scenario, commit)
        path = OUT_DIR / f"{scenario['name']}.json"
        path.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False))
        rs = fixture.get("rshgf_max_rel_diff")
        rs_txt = "n/a" if rs is None else f"{rs:.2e}"
        print(f"{scenario['name']:<42} jax-vs-rshgf max rel diff: {rs_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
