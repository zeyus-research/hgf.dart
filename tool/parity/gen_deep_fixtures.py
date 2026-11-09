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

from pyhgf.model import DeepNetwork  # noqa: E402
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork  # noqa: E402

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
    )
    for layer in spec["layers"]:
        kwargs = dict(layer.get("overrides", {}))
        net = net.add_layer(
            size=layer["size"],
            kind=layer.get("kind", "volatile"),
            add_constant_input=layer.get("add_constant_input", True),
            fully_connected=layer.get("fully_connected", True),
            coupling_fn=COUPLINGS[layer["coupling"]] if "coupling" in layer else None,
            volatility_parent=layer.get("volatility_parent"),
            **kwargs,
        )
    layers = list(net.state.layers)
    for i, w in enumerate(spec["weights"], start=1):
        layers[i] = dataclasses.replace(layers[i], weights_mean=jnp.asarray(w))
    net.state = dataclasses.replace(net.state, layers=tuple(layers))
    return net


def snapshot(net: DeepNetwork) -> dict:
    return {
        "weights": [_mat(layer.weights_mean) for layer in net.state.layers[1:]],
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


def run(spec: dict) -> dict:
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
                update_precisions=op.get("update_precisions", True),
                time_step=op.get("time_step", 1.0),
            )
            out["input_errors"] = _mat(net.input_errors)
        elif kind == "predict":
            out["predictions"] = _mat(net.predict(np.asarray(op["x"])))
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


def fit_op(layers, rng, n=40, optimiser=None, record=STATE_FIELDS, **kw) -> dict:
    return {
        "op": "fit",
        "x": rng.normal(size=(n, layers[-1]["size"])).tolist(),
        "y": targets(layers[0].get("kind", "volatile"), rng, n, layers[0]["size"]),
        "optimiser": optimiser or {"id": 0, "kind": "sgd", "lr": 0.05},
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
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in scenarios().items():
        ops = run(spec)
        fixture = {
            "name": name,
            "pyhgf": "JAX float64",
            "settings": spec["settings"],
            "layers": spec["layers"],
            "weights": spec["weights"],
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
