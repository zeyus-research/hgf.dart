"""Time pyhgf's DeepNetwork (JAX, rshgf) on the problems written by
packages/hgf_deep/benchmark/bin/deep_benchmark.dart.

    uv run python bench_deep.py <problems.json> [--f32]
    uv run python bench_deep.py <deep_sweep.json> --sweep rows.jsonl [--f32] [--no-rs]

``--f32`` runs JAX in float32 (its default precision); otherwise float64.
rshgf is timed in whatever precision it was built with (float32 by default).
``--sweep`` takes the problems of hgf_gpu's ``deep_sweep.dart`` (fit on
their ``fit_samples`` first rows) and appends one JSON row per backend and
problem. JAX runs where ``JAX_PLATFORMS`` puts it; results are fetched to
host memory, so a GPU pays for its readback.
"""

from __future__ import annotations

import os
import sys

F32 = "--f32" in sys.argv
os.environ["JAX_ENABLE_X64"] = "0" if F32 else "1"

import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from pyhgf.model import DeepNetwork  # noqa: E402
from pyhgf.rshgf import DeepNetwork as RsDeepNetwork  # noqa: E402

warnings.filterwarnings("ignore")
DT = np.float32 if F32 else np.float64


def best_ms(run, repeat=5) -> float:
    run()
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - t) * 1000)
    return best


def jax_net(p):
    net = DeepNetwork(coupling_fn=jnp.tanh)
    for s in p["sizes"]:
        net = net.add_layer(size=s)
    layers = list(net.state.layers)
    for i, w in enumerate(p["weights"], start=1):
        layers[i] = dataclasses.replace(layers[i], weights_mean=jnp.asarray(w, dtype=DT))
    net.state = dataclasses.replace(net.state, layers=tuple(layers))
    return net


def rs_net(p):
    net = RsDeepNetwork(coupling_fn="tanh")
    for s in p["sizes"]:
        net = net.add_layer(size=s)
    dt = net.get_weights()[0].dtype
    net.set_weights([np.asarray(w, dtype=dt) for w in p["weights"]])
    return net, dt


def main() -> None:
    problems = json.load(open(sys.argv[1]))
    sweep = sys.argv[sys.argv.index("--sweep") + 1] if "--sweep" in sys.argv else None
    no_rs = "--no-rs" in sys.argv
    device = jax.default_backend()
    tag = "JAX f32" if F32 else "JAX f64"
    for p in problems:
        x = np.asarray(p["x"], dtype=DT)
        y = np.asarray(p["y"], dtype=DT)
        b = p["batch"]
        nf = p.get("fit_samples", len(x))
        xf, yf = x[:nf], y[:nf]
        adam = optax.adam(1e-3)

        def fit():
            net = jax_net(p)
            net.fit(xf, yf, optimiser=adam, check_gradient_health=False)
            jax.device_get(net.predictions)

        net = jax_net(p)

        def batch():
            net.batch_update(x[:b], y[:b], optimiser=adam)
            jax.device_get(net.input_errors)

        def predict():
            jax.device_get(net.predict(x))

        res = {"fit": best_ms(fit, 3), "batch_update": best_ms(batch), "predict": best_ms(predict)}
        if sweep:
            rows = [dict(backend=f"jax_{'cpu' if device == 'cpu' else 'gpu'}_{'f32' if F32 else 'f64'}", **res)]
            if not no_rs:
                rs, dt = rs_net(p)
                xr, yr = x.astype(dt), y.astype(dt)
                rows.append(dict(
                    backend=f"rshgf_{'f32' if dt == np.float32 else 'f64'}",
                    fit=best_ms(lambda: rs_net(p)[0].fit(xr[:nf], yr[:nf], optimiser="adam", learning_rate=1e-3), 3),
                    batch_update=best_ms(lambda: rs.batch_update(xr[:b], yr[:b], optimiser="adam", learning_rate=1e-3)),
                    predict=best_ms(lambda: rs.predict(xr)),
                ))
            with open(sweep, "a") as f:
                for r in rows:
                    f.write(json.dumps(dict(name=p["name"], width=p.get("width"), **r)) + "\n")
                    print(f"{p['name']:6s} {r['backend']:13s} fit {r['fit']:9.2f}  batch {r['batch_update']:8.2f}  predict {r['predict']:8.2f} ms", flush=True)
            continue

        rs, dt = rs_net(p)
        xr, yr = x.astype(dt), y.astype(dt)
        rs_res = {
            "fit": best_ms(lambda: rs_net(p)[0].fit(xr, yr, optimiser="adam", learning_rate=1e-3), 3),
            "batch_update": best_ms(lambda: rs.batch_update(xr[:b], yr[:b], optimiser="adam", learning_rate=1e-3)),
            "predict": best_ms(lambda: rs.predict(xr)),
        }
        d = p["dart_ms"]
        print(f"{p['name']:7s} {p['sizes']}")
        for k in ("fit", "batch_update", "predict"):
            print(f"  {k:12s} Dart {d[k]:9.2f} ms   {tag} {res[k]:9.2f} ms   rshgf {dt} {rs_res[k]:9.2f} ms")


if __name__ == "__main__":
    main()
