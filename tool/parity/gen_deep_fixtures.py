"""Generate pyhgf DeepNetwork reference fixtures for packages/hgf_deep.

Runs pyhgf's vectorised (JAX) DeepNetwork in float64 on small networks with
fixed weights and data, and records the per-step predictions, the recorded
trajectories, the input-layer errors of batched steps, and the weights and
beliefs after every operation. Where the Rust backend (``pyhgf.rshgf``,
float32) supports the scenario, its largest deviation from JAX is stored
too, for information.

    uv run python gen_deep_fixtures.py

Writes ``packages/hgf_deep/test/fixtures/<scenario>.json``.
"""

from __future__ import annotations

import os

os.environ["JAX_ENABLE_X64"] = "1"

import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import warnings  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

import pyhgf.utils.vectorised_belief_propagation as _vbp  # noqa: E402
from pyhgf.model import DeepNetwork  # noqa: E402
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork  # noqa: E402
from pyhgf.updates.vectorised.volatile import posterior as _vposterior  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "packages" / "hgf_deep" / "test" / "fixtures"

COUPLINGS = {
    "linear": lambda x: x,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "relu": jax.nn.relu,
    "leaky_relu": jax.nn.leaky_relu,
    "gelu": partial(jax.nn.gelu, approximate=False),
}

STATE_FIELDS = (
    "mean",
    "precision",
    "expected_mean",
    "expected_precision",
    "conditional_expected_precision",
    "effective_precision",
    "value_prediction_error",
    "mean_vol",
    "precision_vol",
    "expected_mean_vol",
    "expected_precision_vol",
    "effective_precision_vol",
    "volatility_prediction_error",
)


def _clean(x):
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def _mat(a) -> list:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    return [[_clean(v) for v in row] for row in a]


def _flat(a) -> list:
    return [_clean(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)]


def build(spec: dict) -> DeepNetwork:
    s = spec["settings"]
    net = DeepNetwork(
        coupling_fn=COUPLINGS[s.get("coupling", "linear")],
        volatility_updates=s.get("volatility_updates", "unbounded"),
        max_posterior_precision=s.get("max_posterior_precision", 1e10),
        precision_clipping_value=s.get("precision_clipping_value", 1e-6),
        predict_precision=s.get("predict_precision", True),
        feedforward_uncertainty=s.get("feedforward_uncertainty", False),
        tonic_volatility=s.get("tonic_volatility", False),
        mean_field_updates=s.get("mean_field_updates", False),
        carry_precision=s.get("carry_precision", True),
        update_input_layer=s.get("update_input_layer", False),
    )
    for layer in spec["layers"]:
        kwargs = dict(layer.get("overrides", {}))
        if layer.get("kind") == "continuous":
            for key in (
                "value_children",
                "volatility_children",
                "value_coupling",
                "volatility_coupling",
                "volatility_fully_connected",
            ):
                if key in layer:
                    kwargs[key] = layer[key]
            add_constant = None
        else:
            add_constant = layer.get("add_constant_input", True)
        net = net.add_layer(
            size=layer["size"],
            kind=layer.get("kind", "volatile"),
            add_constant_input=add_constant,
            fully_connected=layer.get("fully_connected", True),
            coupling_fn=COUPLINGS[layer["coupling"]] if "coupling" in layer else None,
            volatility_parent=layer.get("volatility_parent"),
            **kwargs,
        )
    # Weight matrices of layers 1..n (``None`` keeps the built one); for a
    # continuous network the value couplings W, plus the volatility
    # couplings κ in ``volatility_weights``.
    layers = list(net.state.layers)
    for key, field in (("weights", "weights_mean"), ("volatility_weights", "volatility_weights")):
        for i, w in enumerate(spec.get(key) or [], start=1):
            if w is not None:
                layers[i] = dataclasses.replace(layers[i], **{field: jnp.asarray(w)})
    net.state = dataclasses.replace(net.state, layers=tuple(layers))
    return net


def snapshot(net: DeepNetwork) -> dict:
    deltas = [layer.weights_precision_delta for layer in net.state.layers[1:]]
    return {
        "weights": [
            None if layer.weights_mean is None else _mat(layer.weights_mean)
            for layer in net.state.layers[1:]
        ],
        **(
            {
                "volatility_weights": [
                    None
                    if layer.volatility_weights is None
                    else _mat(layer.volatility_weights)
                    for layer in net.state.layers[1:]
                ]
            }
            if net._is_continuous
            else {}
        ),
        **(
            {
                "weights_precision_delta": [
                    None if d is None else _mat(d) for d in deltas
                ]
            }
            if any(d is not None for d in deltas)
            else {}
        ),
        "state": {
            field: [
                None
                if getattr(layer.state, field) is None
                else _flat(getattr(layer.state, field))
                for layer in net.state.layers
            ]
            for field in STATE_FIELDS
        },
    }


def make_optimiser(cache: dict, spec: dict | None):
    if spec is None:
        return None
    key = spec["id"]
    if key not in cache:
        lr = spec["lr"]
        cache[key] = optax.sgd(lr) if spec["kind"] == "sgd" else optax.adam(lr)
    return cache[key]


# ---------------------------------------------------------------------------
# Proposed upstream fix: the curvature floor of the value posterior (P18)
# ---------------------------------------------------------------------------
#
# pyhgf clips a hidden layer's posterior precision
# π̂ + Σ w²·γ·g′² − g″·Σ w·g_a·δ at π̂ from below. When the curvature term
# cancels the information, the precision falls back to the prior while the
# mean still moves by msg·g′/π (steps of 1e3 to 1e15 once π̂ is small), which
# drives the volatility runaway. hgf_deep's DeepFixes.curvatureFloor floors
# the precision at π̂ + Σ w²·γ·g′² instead. Fixtures whose settings carry
# "fixes": {"curvature_floor": true} come from pyhgf's JAX backend with that
# change, computed here independently of the Dart code.


def _layer_posterior_update_curvature_floor(
    layer,
    child,
    weights,
    coupling_fn,
    parent_has_constant=False,
    max_posterior_precision=1e10,
    child_is_input_layer=False,
    mean_field_updates=False,
):
    if parent_has_constant:
        weights = weights[:, :-1]
    precision = _vposterior.vectorised_posterior_update_precision_value_level(
        layer,
        child,
        weights,
        coupling_fn,
        child_is_input_layer=child_is_input_layer,
        mean_field_updates=mean_field_updates,
    )
    # The first-order (information) term alone, with the effective child
    # precision of vectorised_posterior_update_precision_value_level.
    if mean_field_updates:
        effective = child.expected_precision
    elif child_is_input_layer:
        effective = child.conditional_expected_precision
    else:
        pi_y = child.precision - child.expected_precision
        effective = (
            child.conditional_expected_precision
            * pi_y
            / (child.conditional_expected_precision + pi_y)
        )
    g1 = jax.vmap(jax.grad(coupling_fn))(layer.expected_mean)
    information = jnp.matmul((weights**2).T, effective) * g1**2
    posterior_precision = jnp.clip(
        precision, layer.expected_precision + information, max_posterior_precision
    )
    posterior_mean = _vposterior.vectorised_posterior_update_mean_value_level(
        layer,
        child,
        weights,
        coupling_fn,
        posterior_precision,
        mean_field_updates=mean_field_updates,
    )
    return dataclasses.replace(
        layer, precision=posterior_precision, mean=posterior_mean
    )


class _CurvatureFloor:
    """Swap the value posterior for the floored one (and clear JAX's caches on
    both sides, so no trace of the other version is reused)."""

    def __enter__(self):
        self._orig = _vbp.vectorised_layer_posterior_update
        _vbp.vectorised_layer_posterior_update = _layer_posterior_update_curvature_floor
        jax.clear_caches()

    def __exit__(self, *exc):
        _vbp.vectorised_layer_posterior_update = self._orig
        jax.clear_caches()
        return False


class _Nothing:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _patches(spec: dict):
    fixes = spec["settings"].get("fixes", {})
    unknown = set(fixes) - {"curvature_floor"}
    if unknown:
        raise ValueError(f"no JAX reference for the fixes {sorted(unknown)}")
    return _CurvatureFloor() if fixes.get("curvature_floor") else _Nothing()


def pyhgf_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO.parent / "pyhgf"), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run(spec: dict) -> dict:
    with _patches(spec):
        return _run(spec)


def _run(spec: dict) -> dict:
    net = build(spec)
    optimisers: dict = {}
    ops_out = []
    for op in spec["ops"]:
        out = dict(op)
        kind = op["op"]
        if kind == "fit":
            record = tuple(op.get("record", ()))
            net.fit(
                np.asarray(op["x"]),
                np.asarray(op["y"]),
                optimiser=make_optimiser(optimisers, op.get("optimiser")),
                learning_kind=op.get("learning_kind", "precision_weighted"),
                learning_kwargs=op.get("learning_kwargs"),
                weight_update=op.get("weight_update", True),
                time_step=op.get("time_step", 1.0),
                update_precisions=op.get("update_precisions", True),
                record=record or None,
                check_gradient_health=False,
            )
            out["predictions"] = _mat(net.predictions)
            if record:
                out["trajectories"] = {
                    field: [
                        None if traj is None else _mat(traj)
                        for traj in net.trajectories[field]
                    ]
                    for field in record
                }
        elif kind == "batch_update":
            net.batch_update(
                np.asarray(op["x"]),
                np.asarray(op["y"]),
                optimiser=make_optimiser(optimisers, op.get("optimiser")),
                learning_kind=op.get("learning_kind", "precision_weighted"),
                learning_kwargs=op.get("learning_kwargs"),
                update_precisions=op.get("update_precisions", True),
                time_step=op.get("time_step", 1.0),
                sample_weight=None
                if op.get("sample_weight") is None
                else np.asarray(op["sample_weight"]),
            )
            out["input_errors"] = _mat(net.input_errors)
        elif kind == "health":
            h = net.check_gradient_health(
                np.asarray(op["x"]),
                np.asarray(op["y"]),
                learning_kind=op.get("learning_kind", "precision_weighted"),
            )
            out["magnitudes"] = [None if m is None else _clean(m) for m in h["magnitudes"]]
            out["dead"] = list(h["dead"])
            out["quantised"] = list(h["quantised"])
        elif kind == "predict":
            out["predictions"] = _mat(net.predict(np.asarray(op["x"])))
        elif kind == "input_data":
            record = tuple(op.get("record", ()))
            net.input_data(
                np.asarray(op["y"]),
                time_steps=None
                if op.get("time_steps") is None
                else np.asarray(op["time_steps"]),
                record=record or None,
            )
            out["predictions"] = _mat(np.asarray(net.predictions).reshape(len(op["y"]), -1))
            if record:
                out["trajectories"] = {
                    field: [
                        None if traj is None else _mat(traj)
                        for traj in net.trajectories[field]
                    ]
                    for field in record
                }
        else:
            raise ValueError(kind)
        out["after"] = snapshot(net)
        ops_out.append(out)
    return ops_out


RS_SUPPORTED_SETTINGS = {
    "volatility_updates",
    "max_posterior_precision",
    "precision_clipping_value",
    "predict_precision",
    "feedforward_uncertainty",
    "tonic_volatility",
    "carry_precision",
    "coupling",
}


def rshgf_deviation(spec: dict, ops: list) -> float | None:
    """Largest |Rust f32 − JAX f64| over the predictions and weights."""
    if set(spec["settings"]) - RS_SUPPORTED_SETTINGS:
        return None
    if any(layer.get("kind") == "continuous" for layer in spec["layers"]):
        return None  # rshgf has no continuous layers
    if any(
        op.get("learning_kwargs") or op.get("sample_weight") is not None or op["op"] == "health"
        for op in spec["ops"]
    ):
        return None
    s = spec["settings"]
    net = RsDeepNetwork(
        volatility_updates=s.get("volatility_updates", "unbounded"),
        max_posterior_precision=s.get("max_posterior_precision", 1e10),
        precision_clipping_value=s.get("precision_clipping_value", 1e-6),
        coupling_fn=s.get("coupling", "linear"),
        predict_precision=s.get("predict_precision", True),
        feedforward_uncertainty=s.get("feedforward_uncertainty", False),
        tonic_volatility=s.get("tonic_volatility", False),
        carry_precision=s.get("carry_precision", True),
    )
    for layer in spec["layers"]:
        kwargs = {k: float(v) for k, v in layer.get("overrides", {}).items()}
        net = net.add_layer(
            size=layer["size"],
            kind=layer.get("kind", "volatile"),
            add_constant_input=layer.get("add_constant_input", True),
            fully_connected=layer.get("fully_connected", True),
            coupling_fn=layer.get("coupling"),
            volatility_parent=layer.get("volatility_parent"),
            **kwargs,
        )
    net.set_weights([np.asarray(w, dtype=np.float32) for w in spec["weights"]])
    worst = 0.0
    for op in ops:
        opt = op.get("optimiser")
        if op["op"] == "fit":
            if not op.get("update_precisions", True) or opt is None:
                return None
            pred = net.fit(
                np.asarray(op["x"], dtype=np.float32),
                np.asarray(op["y"], dtype=np.float32),
                optimiser=opt["kind"],
                learning_rate=opt["lr"],
                learning_kind=op.get("learning_kind", "precision_weighted"),
                weight_update=op.get("weight_update", True),
                time_step=op.get("time_step", 1.0),
            )
        elif op["op"] == "batch_update":
            kw = {}
            if opt is not None:
                kw = {"optimiser": opt["kind"], "learning_rate": opt["lr"]}
            pred = net.batch_update(
                np.asarray(op["x"], dtype=np.float32),
                np.asarray(op["y"], dtype=np.float32),
                learning_kind=op.get("learning_kind", "precision_weighted"),
                update_precisions=op.get("update_precisions", True),
                time_step=op.get("time_step", 1.0),
                **kw,
            )
        else:
            pred = net.predict(np.asarray(op["x"], dtype=np.float32))
        ref = op.get("predictions", op.get("input_errors"))
        worst = max(worst, float(np.nanmax(np.abs(np.asarray(pred) - np.asarray(ref, dtype=float)))))
        for w_rs, w_jax in zip(net.get_weights(), op["after"]["weights"]):
            worst = max(worst, float(np.nanmax(np.abs(np.asarray(w_rs) - np.asarray(w_jax, dtype=float)))))
    return worst


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def weights_for(layers: list, rng, scale: float = 0.5) -> list:
    out = []
    for i in range(1, len(layers)):
        child = layers[i - 1]["size"]
        cols = layers[i]["size"] + (1 if layers[i].get("add_constant_input", True) else 0)
        if layers[i].get("fully_connected", True):
            out.append(rng.normal(scale=scale, size=(child, cols)).tolist())
        else:
            out.append((np.eye(child) * rng.uniform(0.5, 1.5, size=child)).tolist())
    return out


def targets(kind: str, rng, n: int, size: int) -> list:
    if kind == "binary":
        return rng.integers(0, 2, size=(n, size)).astype(float).tolist()
    if kind == "categorical":
        return np.eye(size)[rng.integers(0, size, size=n)].tolist()
    return rng.normal(size=(n, size)).tolist()


def synaptic(**kwargs) -> dict:
    """Op keywords for learning_kind='synaptic_uncertainty' (no optimiser)."""
    return {
        "learning_kind": "synaptic_uncertainty",
        "learning_kwargs": {"window": 20.0, **kwargs},
    }


def health_op(layers, rng, **kw) -> dict:
    return {
        "op": "health",
        "x": rng.normal(size=layers[-1]["size"]).tolist(),
        "y": targets(layers[0].get("kind", "volatile"), rng, 1, layers[0]["size"])[0],
        **kw,
    }


def fit_op(layers, rng, n=40, optimiser=None, record=STATE_FIELDS, **kw) -> dict:
    return {
        "op": "fit",
        "x": rng.normal(size=(n, layers[-1]["size"])).tolist(),
        "y": targets(layers[0].get("kind", "volatile"), rng, n, layers[0]["size"]),
        "optimiser": None
        if kw.get("learning_kind") == "synaptic_uncertainty"
        else optimiser or {"id": 0, "kind": "sgd", "lr": 0.05},
        "record": list(record),
        **kw,
    }


def batch_op(layers, rng, n=16, optimiser=None, **kw) -> dict:
    return {
        "op": "batch_update",
        "x": rng.normal(size=(n, layers[-1]["size"])).tolist(),
        "y": targets(layers[0].get("kind", "volatile"), rng, n, layers[0]["size"]),
        "optimiser": optimiser,
        **kw,
    }


def predict_op(layers, rng, n=8) -> dict:
    return {"op": "predict", "x": rng.normal(size=(n, layers[-1]["size"])).tolist()}


def scenarios() -> dict:
    out = {}

    def add(name, settings, layers, ops_fn, seed, weight_scale=0.5):
        rng = np.random.default_rng(seed)
        out[name] = {
            "settings": settings,
            "layers": layers,
            "weights": weights_for(layers, rng, weight_scale),
            "ops": ops_fn(layers, rng),
        }

    three = [{"size": 2}, {"size": 4}, {"size": 3}]
    seed = 100
    for vol in ("unbounded", "eHGF", "standard"):
        for coupling in ("linear", "tanh"):
            seed += 1
            add(
                f"fit_{vol}_{coupling}",
                {"volatility_updates": vol, "coupling": coupling},
                three,
                # Linear networks diverge at SGD 0.05 (|w| ~ 1e200 in 40 steps).
                lambda layers, rng, lr=0.005 if coupling == "linear" else 0.05: [
                    fit_op(layers, rng, optimiser={"id": 0, "kind": "sgd", "lr": lr}),
                    predict_op(layers, rng),
                ],
                seed,
            )

    adam = {"id": 1, "kind": "adam", "lr": 0.01}
    add(
        "fit_adam_continued",
        {"coupling": "tanh"},
        three,
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            fit_op(layers, rng, n=30, optimiser=adam, record=()),
            predict_op(layers, rng),
        ],
        201,
    )
    deep4 = [{"size": 2}, {"size": 5}, {"size": 4}, {"size": 3}]
    for vol in ("unbounded", "eHGF"):
        add(
            f"deep4_tanh_{vol}",
            {"volatility_updates": vol, "coupling": "tanh"},
            deep4,
            lambda layers, rng: [
                fit_op(layers, rng, n=30, optimiser=adam),
                batch_op(layers, rng, optimiser=adam),
                batch_op(layers, rng, optimiser=adam),
                predict_op(layers, rng),
            ],
            202 if vol == "unbounded" else 203,
        )
    add(
        "couplings_mixed",
        {},
        [
            {"size": 2},
            {"size": 4, "coupling": "sigmoid"},
            {"size": 4, "coupling": "gelu"},
            {"size": 3, "coupling": "leaky_relu"},
            {"size": 3, "coupling": "relu"},
        ],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            batch_op(layers, rng, optimiser=adam),
        ],
        204,
    )
    binary = [
        {"size": 2, "kind": "binary"},
        {"size": 4, "coupling": "tanh"},
        {"size": 3},
    ]
    for lk in ("precision_weighted", "standard"):
        add(
            f"binary_leaf_{lk}",
            {},
            binary,
            lambda layers, rng, lk=lk: [
                fit_op(layers, rng, optimiser=adam, learning_kind=lk),
                batch_op(layers, rng, optimiser=adam, learning_kind=lk),
                predict_op(layers, rng),
            ],
            205,
            1.0,
        )
    add(
        "categorical_leaf",
        {"coupling": "tanh"},
        [{"size": 3, "kind": "categorical"}, {"size": 4}, {"size": 2}],
        lambda layers, rng: [
            fit_op(layers, rng, optimiser=adam),
            batch_op(layers, rng, optimiser=adam),
            predict_op(layers, rng),
        ],
        206,
        1.0,
    )
    add(
        "binary_interior",
        {"coupling": "tanh"},
        [{"size": 2}, {"size": 3, "kind": "binary"}, {"size": 3}],
        lambda layers, rng: [
            fit_op(layers, rng, optimiser=adam),
            batch_op(layers, rng, optimiser=adam),
        ],
        207,
    )
    for vol in ("unbounded", "eHGF", "standard"):
        add(
            f"feedforward_{vol}",
            {"volatility_updates": vol, "feedforward_uncertainty": True, "coupling": "tanh"},
            deep4,
            lambda layers, rng: [
                fit_op(layers, rng, n=30, optimiser=adam),
                batch_op(layers, rng, optimiser=adam),
            ],
            210,
        )
    for vol in ("unbounded", "eHGF", "standard"):
        add(
            f"mean_field_{vol}",
            {"volatility_updates": vol, "mean_field_updates": True, "coupling": "tanh"},
            deep4,
            lambda layers, rng: [
                fit_op(layers, rng, n=30, optimiser=adam),
                batch_op(layers, rng, optimiser=adam),
            ],
            211,
        )
    add(
        "no_carry",
        {"carry_precision": False, "coupling": "tanh"},
        deep4,
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            batch_op(layers, rng, optimiser=adam),
        ],
        212,
    )
    add(
        "no_predict_precision",
        {"predict_precision": False, "coupling": "tanh"},
        three,
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            batch_op(layers, rng, optimiser=adam),
        ],
        213,
    )
    add(
        "tonic_volatility",
        {"tonic_volatility": True, "coupling": "tanh"},
        [
            {"size": 2},
            {"size": 4, "overrides": {"tonic_volatility": -2.0}},
            {"size": 3, "volatility_parent": False},
            {"size": 2},
        ],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            batch_op(layers, rng, optimiser=adam),
        ],
        214,
    )
    add(
        "frozen_precisions",
        {"coupling": "tanh"},
        three,
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam, update_precisions=False),
            batch_op(layers, rng, optimiser=adam, update_precisions=False),
            fit_op(layers, rng, n=10, optimiser=adam, weight_update=False),
        ],
        215,
    )
    add(
        "overrides_one_to_one",
        {"volatility_updates": "eHGF"},
        [
            {"size": 3, "overrides": {"precision": 5.0, "tonic_volatility_vol": -3.0}},
            {
                "size": 3,
                "add_constant_input": False,
                "fully_connected": False,
                "overrides": {"precision": 2.0, "mean_vol": -1.0, "precision_vol": 4.0},
            },
            {"size": 2},
        ],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam, time_step=0.5),
            batch_op(layers, rng, optimiser=None, time_step=0.5),
        ],
        216,
    )
    add(
        "batch_only",
        {"coupling": "tanh"},
        deep4,
        lambda layers, rng: [
            batch_op(layers, rng, n=32, optimiser={"id": 2, "kind": "sgd", "lr": 0.1}),
            batch_op(layers, rng, n=32, optimiser={"id": 2, "kind": "sgd", "lr": 0.1}),
            batch_op(layers, rng, n=1, optimiser={"id": 2, "kind": "sgd", "lr": 0.1}),
            predict_op(layers, rng),
        ],
        217,
    )

    # The weight-belief rule (learning_kind="synaptic_uncertainty").
    syn_layers = [{"size": 2}, {"size": 5}, {"size": 4}, {"size": 3}]
    for vol in ("unbounded", "eHGF", "standard"):
        add(
            f"synaptic_{vol}",
            {"volatility_updates": vol, "coupling": "tanh"},
            syn_layers,
            lambda layers, rng: [
                fit_op(layers, rng, n=40, **synaptic(prior_variance=0.05)),
                batch_op(layers, rng, n=16, **synaptic(prior_variance=0.05)),
                fit_op(layers, rng, n=10, record=(), **synaptic(prior_variance=0.05)),
                predict_op(layers, rng),
            ],
            400 + len(vol),
            0.25,
        )
    add(
        "synaptic_settings_one_to_one",
        {"coupling": "tanh", "feedforward_uncertainty": True},
        [
            {"size": 3},
            {"size": 3, "add_constant_input": False, "fully_connected": False},
            {"size": 4, "coupling": "sigmoid"},
            {"size": 2},
        ],
        lambda layers, rng: [
            fit_op(
                layers,
                rng,
                n=30,
                **synaptic(
                    window=8.0,
                    prior_variance=0.2,
                    prior_mean=0.1,
                    learning_rate=0.5,
                    increment_scale=3.0,
                ),
            ),
            batch_op(
                layers,
                rng,
                n=12,
                **synaptic(window=8.0, prior_variance=0.2, increment_scale=3.0),
            ),
        ],
        410,
    )
    add(
        "synaptic_binary_leaf",
        {},
        [{"size": 2, "kind": "binary"}, {"size": 4, "coupling": "tanh"}, {"size": 3}],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, **synaptic(prior_variance=0.5)),
            batch_op(layers, rng, n=16, **synaptic(prior_variance=0.5)),
        ],
        411,
        1.0,
    )
    add(
        "synaptic_categorical_leaf",
        {"coupling": "tanh"},
        [{"size": 3, "kind": "categorical"}, {"size": 4}, {"size": 2}],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, **synaptic(prior_variance=0.5)),
            batch_op(layers, rng, n=16, **synaptic(prior_variance=0.5)),
            predict_op(layers, rng),
        ],
        412,
        1.0,
    )

    # Weighted batch means (sample_weight), with an optimiser and with the
    # weight-belief rule; a 0/1 mask and an all-zero mask.
    def weighted_ops(layers, rng):
        def w(n, kind):
            if kind == "mask":
                return [1.0] * (n - 4) + [0.0] * 4
            if kind == "zero":
                return [0.0] * n
            return rng.uniform(0.0, 2.0, size=n).tolist()

        return [
            batch_op(layers, rng, n=16, optimiser=adam, sample_weight=w(16, "real")),
            batch_op(layers, rng, n=16, optimiser=adam, sample_weight=w(16, "mask")),
            batch_op(layers, rng, n=8, optimiser=adam, sample_weight=w(8, "zero")),
            batch_op(layers, rng, n=16, sample_weight=w(16, "real"), **synaptic()),
            batch_op(layers, rng, n=1, optimiser=adam, sample_weight=[3.0]),
        ]

    add("sample_weight", {"coupling": "tanh"}, deep4, weighted_ops, 420)
    add(
        "sample_weight_categorical",
        {"coupling": "tanh", "feedforward_uncertainty": True},
        [{"size": 3, "kind": "categorical"}, {"size": 4}, {"size": 2}],
        weighted_ops,
        421,
        1.0,
    )

    # update_input_layer: the sweeps reach the top layer's precision.
    for vol in ("unbounded", "eHGF", "standard"):
        add(
            f"input_layer_{vol}",
            {"volatility_updates": vol, "coupling": "tanh", "update_input_layer": True},
            deep4,
            lambda layers, rng: [
                fit_op(layers, rng, n=30, optimiser=adam),
                batch_op(layers, rng, n=16, optimiser=adam),
                predict_op(layers, rng),
            ],
            430 + len(vol),
            0.25,
        )
    add(
        "input_layer_feedforward_tonic",
        {
            "coupling": "tanh",
            "update_input_layer": True,
            "feedforward_uncertainty": True,
            "tonic_volatility": True,
        },
        [
            {"size": 2},
            {"size": 4},
            {"size": 3, "overrides": {"tonic_volatility": -2.0, "precision": 3.0}},
        ],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            batch_op(layers, rng, n=16, optimiser=adam),
            fit_op(layers, rng, n=20, **synaptic()),
        ],
        440,
        0.25,
    )
    add(
        "input_layer_two_layers",
        {"update_input_layer": True, "carry_precision": False},
        [{"size": 2}, {"size": 3, "volatility_parent": False, "overrides": {"precision": 0.5}}],
        lambda layers, rng: [
            fit_op(layers, rng, n=30, optimiser=adam),
            batch_op(layers, rng, n=16, optimiser=adam),
        ],
        441,
    )
    add(
        "input_layer_frozen",
        {"update_input_layer": True, "predict_precision": False, "coupling": "tanh"},
        three,
        lambda layers, rng: [
            fit_op(layers, rng, n=20, optimiser=adam),
            fit_op(layers, rng, n=10, optimiser=adam, update_precisions=False),
            batch_op(layers, rng, n=8, optimiser=adam),
        ],
        442,
    )
    add(
        "input_layer_binary_top",
        {"update_input_layer": True, "coupling": "tanh"},
        [{"size": 2}, {"size": 3}, {"size": 2, "kind": "binary"}],
        lambda layers, rng: [fit_op(layers, rng, n=20, optimiser=adam)],
        443,
    )

    # check_gradient_health after some training.
    add(
        "gradient_health",
        {"coupling": "tanh"},
        deep4,
        lambda layers, rng: [
            fit_op(layers, rng, n=20, optimiser=adam, record=()),
            health_op(layers, rng),
            health_op(layers, rng, learning_kind="standard"),
            health_op(layers, rng, learning_kind="synaptic_uncertainty"),
        ],
        450,
    )

    # Patched references: the curvature floor (see _CurvatureFloor). The
    # four-layer tanh network on N(0, 1) data is P18's setting, where pyhgf
    # runs away within 40 samples.
    four = [{"size": 2}, {"size": 4}, {"size": 4}, {"size": 3}]
    seed = 300
    for vol in ("unbounded", "eHGF", "standard"):
        seed += 1
        add(
            f"floor_{vol}_tanh",
            {"volatility_updates": vol, "coupling": "tanh", "fixes": {"curvature_floor": True}},
            four,
            lambda layers, rng: [
                fit_op(layers, rng, n=60, optimiser={"id": 0, "kind": "adam", "lr": 0.01}),
                batch_op(layers, rng, n=16, optimiser={"id": 0, "kind": "adam", "lr": 0.01}),
                predict_op(layers, rng),
            ],
            seed,
        )
    add(
        "floor_binary_mean_field",
        {"coupling": "tanh", "mean_field_updates": True, "fixes": {"curvature_floor": True}},
        [{"size": 2, "kind": "binary"}, {"size": 5}, {"size": 3}],
        lambda layers, rng: [fit_op(layers, rng, n=40), predict_op(layers, rng)],
        310,
    )
    continuous_scenarios(out)
    return out


# ---------------------------------------------------------------------------
# Continuous layers (DeepNetwork.input_data)
# ---------------------------------------------------------------------------

CONTINUOUS_FIELDS = (
    "mean",
    "precision",
    "expected_mean",
    "expected_precision",
    "conditional_expected_precision",
    "effective_precision",
    "value_prediction_error",
    "volatility_prediction_error",
)


def _c(size, **kw) -> dict:
    return {"size": size, "kind": "continuous", **kw}


def _series(rng, n: int, size: int) -> list:
    """A drifting, heteroscedastic series (random walk with bursts)."""
    t = np.arange(n)[:, None]
    scale = 0.3 + 0.7 * (np.sin(t / 7.0) > 0.5)
    walk = np.cumsum(rng.normal(scale=0.1, size=(n, size)), axis=0)
    return (walk + scale * rng.normal(size=(n, size))).tolist()


def input_data_op(rng, n: int, size: int, time_steps: bool = False, record=CONTINUOUS_FIELDS) -> dict:
    return {
        "op": "input_data",
        "y": _series(rng, n, size),
        "time_steps": rng.uniform(0.1, 3.0, size=n).tolist() if time_steps else None,
        "record": [f for f in record],
    }


def continuous_scenarios(out: dict) -> None:
    seed = 500

    def add(name, settings, layers, n=40, time_steps=False, calls=1, matrices=None):
        nonlocal seed
        seed += 1
        rng = np.random.default_rng(seed)
        size = layers[0]["size"]
        record = list(CONTINUOUS_FIELDS)
        # Only fields some layer allocates can be recorded.
        if not any("volatility_children" in layer for layer in layers):
            record.remove("volatility_prediction_error")
        spec = {
            "settings": settings,
            "layers": layers,
            "weights": [],
            "ops": [
                input_data_op(rng, n, size, time_steps, record)
                for _ in range(calls)
            ],
        }
        if matrices is not None:
            spec["weights"], spec["volatility_weights"] = matrices(rng)
        out[name] = spec

    add("continuous_chain", {"volatility_updates": "standard"}, [_c(1), _c(1)])
    for vol in ("standard", "eHGF", "unbounded"):
        one = vol != "unbounded"
        add(
            f"continuous_noise_parent_{vol}",
            {"volatility_updates": vol},
            [_c(1), _c(1), _c(1, volatility_children=0, volatility_fully_connected=one)],
        )
        add(
            f"continuous_three_level_{vol}",
            {"volatility_updates": vol},
            [_c(1), _c(1), _c(1, volatility_children=1, volatility_fully_connected=one)],
            n=60,
        )
        add(
            f"continuous_three_level_{vol}_mean_field",
            {"volatility_updates": vol, "mean_field_updates": True},
            [_c(1), _c(1), _c(1, volatility_children=1, volatility_fully_connected=one)],
        )
        add(
            f"continuous_noise_below_value_{vol}",
            {"volatility_updates": vol},
            [
                _c(1),
                _c(1, volatility_children=0, volatility_fully_connected=one),
                _c(1, value_children=0),
            ],
        )
    add(
        "continuous_unbounded_one_to_one",
        {"volatility_updates": "unbounded"},
        [
            _c(3),
            _c(3, fully_connected=False, value_coupling=0.8),
            _c(3, volatility_children=1, volatility_fully_connected=False, volatility_coupling=0.6),
            _c(3, volatility_children=0, volatility_fully_connected=False),
        ],
        n=50,
    )
    for vol in ("standard", "eHGF"):
        for mf in (False, True):
            tag = "_mean_field" if mf else ""
            add(
                f"continuous_dense_{vol}{tag}",
                {"volatility_updates": vol, "mean_field_updates": mf},
                [_c(2), _c(3), _c(2, volatility_children=1)],
            )
            # General (non-uniform) couplings: random W and κ.
            add(
                f"continuous_random_couplings_{vol}{tag}",
                {"volatility_updates": vol, "mean_field_updates": mf},
                [
                    _c(3),
                    _c(4, coupling="tanh"),
                    _c(2, volatility_children=1),
                    _c(2, volatility_children=0),
                ],
                n=50,
                matrices=lambda rng: (
                    [rng.normal(scale=0.6, size=(3, 4)).tolist(), None, None],
                    [None, rng.normal(scale=0.5, size=(4, 2)).tolist(),
                     rng.normal(scale=0.5, size=(3, 2)).tolist()],
                ),
            )
        add(
            f"continuous_dual_role_{vol}",
            {"volatility_updates": vol},
            [_c(1), _c(1), _c(1, value_children=1, volatility_children=0)],
        )
        add(
            f"continuous_sigmoid_{vol}",
            {"volatility_updates": vol, "coupling": "sigmoid"},
            [_c(2), _c(2, fully_connected=False), _c(1, volatility_children=0)],
        )
    add(
        "continuous_time_steps_overrides",
        {"volatility_updates": "eHGF"},
        [
            _c(1, overrides={"precision": 4.0, "tonic_drift": 0.05}),
            _c(
                1,
                overrides={
                    "tonic_volatility": -2.0,
                    "tonic_drift": 0.1,
                    "autoconnection_strength": 0.9,
                    "mean": 0.3,
                },
            ),
            _c(1, volatility_children=1, volatility_coupling=0.7, overrides={"tonic_volatility": -3.0}),
            _c(1, volatility_children=0, overrides={"precision": 2.0}),
        ],
        n=50,
        time_steps=True,
    )
    add(
        "continuous_carried_state",
        {"volatility_updates": "standard", "max_posterior_precision": 1e4},
        [_c(2), _c(2), _c(1, volatility_children=1), _c(1, volatility_children=0)],
        n=25,
        calls=3,
    )


def main() -> None:
    import sys

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = pyhgf_commit()
    only = set(sys.argv[1:])
    for name, spec in scenarios().items():
        if only and name not in only:
            continue
        ops = run(spec)
        fixes = spec["settings"].get("fixes")
        fixture = {
            "name": name,
            "pyhgf": "JAX float64",
            "pyhgf_commit": commit,
            "reference": (
                "pyhgf JAX float64 with the proposed fixes " + ", ".join(sorted(fixes))
                if fixes
                else "pyhgf JAX float64"
            ),
            "settings": spec["settings"],
            "layers": spec["layers"],
            "weights": spec["weights"],
            **(
                {"volatility_weights": spec["volatility_weights"]}
                if "volatility_weights" in spec
                else {}
            ),
            "ops": ops,
        }
        try:
            dev = rshgf_deviation(spec, ops)
            fixture["rshgf_max_abs_dev"] = None if dev is None else _clean(dev)
        except BaseException as e:  # noqa: BLE001 (pyo3 panics)
            fixture["rshgf_max_abs_dev"] = None
            fixture["rshgf_error"] = str(e)
        path = OUT_DIR / f"{name}.json"
        path.write_text(json.dumps(fixture, separators=(",", ":"), allow_nan=False))
        dev = fixture["rshgf_max_abs_dev"]
        print(f"{name:32s} rshgf f32 dev: {dev if dev is None or isinstance(dev, str) else f'{dev:.2e}'}"
              + (f"  ({fixture['rshgf_error']})" if "rshgf_error" in fixture else ""))


if __name__ == "__main__":
    main()
