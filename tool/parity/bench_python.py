"""Python-side benchmarks (pyhgf JAX and rshgf) for the hgf.dart performance gate.

Runs the same scenarios as ``packages/hgf/benchmark/bin/hgf_benchmark.dart``
on the same inputs (``packages/hgf/benchmark/data/inputs.json``, written here
if missing), and writes ``customSmallerIsBetter`` JSON:

    uv run python bench_python.py --repeat 7 --out ../../bench_python.json

Every timing is the median of ``--repeat`` warm runs. JAX compile/trace time is
reported separately as ``.../jax_first_call``.

JAX runs on the CPU unless ``JAX_PLATFORMS`` says otherwise (``rocm``,
``cuda``); then names get a ``_gpu`` tag. Timings end with the results in host
memory (``jax.device_get``, on a GPU), and the batch takes its omegas from host memory,
so a GPU pays for its transfers, as hgf_gpu's end-to-end timings do.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.lax import scan  # noqa: E402

import gen_fixtures as g  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "packages" / "hgf" / "benchmark" / "data" / "inputs.json"

N_LONG = 10_000
N_BATCH_TRIALS = 1_000
N_BATCH = 1_000


def make_inputs() -> dict:
    rng = np.random.default_rng(20261105)
    # Continuous: a random walk whose step size switches regime every 2000
    # trials (USD-CHF-like scale), so the volatility levels have work to do.
    sd = np.where((np.arange(N_LONG) // 2000) % 2 == 0, 0.002, 0.008)
    continuous = 1.0 + np.cumsum(rng.normal(0.0, sd))
    # Binary: Bernoulli draws whose probability flips every 100 trials.
    p = np.where((np.arange(N_LONG) // 100) % 2 == 0, 0.8, 0.2)
    binary = (rng.uniform(size=N_LONG) < p).astype(float)
    omegas = np.linspace(-10.0, -4.0, N_BATCH)
    return {
        "continuous": continuous.tolist(),
        "binary": binary.tolist(),
        "omegas": omegas.tolist(),
        "usdchf": g.usdchf().tolist(),
        # Identical network specs for every runtime (NetworkSpec JSON format).
        "specs": {
            "continuous_3level": g.continuous_3level("unbounded"),
            "binary_3level": g.binary_hgf(3),
            "continuous_2level": g.continuous_2level("unbounded"),
        },
    }


def load_inputs() -> dict:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(make_inputs()))
    return json.loads(DATA.read_text())


def median_ms(fn, repeat: int) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def block(x):
    """Wait for [x]; on a GPU, also bring it to host memory (the readback
    hgf_gpu's timings include). On the CPU the arrays are already there."""
    if jax.default_backend() != "cpu":
        return jax.device_get(x)
    return jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------


# Result-name suffix for JAX runs: float64 is the like-for-like comparison
# (the Dart port is float64); `JAX_ENABLE_X64=0` gives pyhgf's default float32.
JAX_EXTRA = f"JAX {jax.__version__}, {jax.devices()[0].device_kind}"
JAX_TAG = ("" if jax.default_backend() == "cpu" else "_gpu") + ("" if jax.config.jax_enable_x64 else "_f32")


def jax_series(spec: dict, data: np.ndarray, repeat: int, name: str, out: list):
    """Filter a whole series with trajectories, as users do (`input_data`), and
    as a jitted scan (the compute floor)."""
    x = data[:, None]

    net = g.build_jax(spec)
    t0 = time.perf_counter()
    block(net.input_data(input_data=x).node_trajectories)
    first = (time.perf_counter() - t0) * 1e3
    out.append({"name": f"{name}/jax_first_call{JAX_TAG}", "unit": "ms", "extra": JAX_EXTRA, "value": first})

    out.append(
        {
            "name": f"{name}/jax_input_data{JAX_TAG}",
            "unit": "ms",
            "extra": JAX_EXTRA,
            "value": median_ms(lambda: block(net.input_data(input_data=x).node_trajectories), repeat),
        }
    )

    attrs = net.attributes
    n = x.shape[0]
    inputs = ((jnp.asarray(x),), (jnp.ones(n, dtype=int),), jnp.ones(n), None)
    run = jax.jit(lambda a, i: scan(net.scan_fn, a, i))
    block(run(attrs, inputs))
    out.append(
        {
            "name": f"{name}/jax_jit_scan{JAX_TAG}",
            "unit": "ms",
            "extra": JAX_EXTRA,
            "value": median_ms(lambda: block(run(attrs, inputs)), repeat),
        }
    )


def jax_batch(spec: dict, data: np.ndarray, omegas: np.ndarray, repeat: int, name: str, out: list):
    """Fitting-shaped workload: one forward pass per parameter value, final
    state only, vectorised with vmap."""
    x = data[:, None]
    net = g.build_jax(spec)
    net.input_data(input_data=x[:2])  # create the scan function
    attrs = net.attributes
    n = x.shape[0]
    inputs = ((jnp.asarray(x),), (jnp.ones(n, dtype=int),), jnp.ones(n), None)

    def no_traj(a, i):
        new, _ = net.scan_fn(a, i)
        return new, None

    def one(omega):
        a = jax.tree_util.tree_map(lambda v: v, attrs)
        a[1] = dict(a[1])
        a[1]["tonic_volatility"] = omega
        last, _ = scan(no_traj, a, inputs)
        return last[1]["mean"], last[2]["mean"]

    batched = jax.jit(jax.vmap(one))
    om = np.asarray(omegas, dtype=np.float64 if jax.config.jax_enable_x64 else np.float32)
    t0 = time.perf_counter()
    block(batched(om))
    out.append({"name": f"{name}/jax_first_call{JAX_TAG}", "unit": "ms", "extra": JAX_EXTRA, "value": (time.perf_counter() - t0) * 1e3})
    out.append(
        {"name": f"{name}/jax_vmap{JAX_TAG}", "unit": "ms", "extra": JAX_EXTRA, "value": median_ms(lambda: block(batched(om)), repeat)}
    )


# ---------------------------------------------------------------------------
# rshgf
# ---------------------------------------------------------------------------


def rs_series(spec: dict, data: np.ndarray, repeat: int, name: str, out: list):
    rows = data[:, None].tolist()

    def go():
        net = g.build_rs(spec)
        net.input_data(input_data=rows)
        _ = net.node_trajectories  # include the export to Python, as used

    go()
    out.append({"name": f"{name}/rshgf", "unit": "ms", "value": median_ms(go, repeat)})

    def compute_only():
        net = g.build_rs(spec)
        net.input_data(input_data=rows, record_trajectories=False)

    compute_only()
    out.append(
        {"name": f"{name}/rshgf_no_traj", "unit": "ms", "value": median_ms(compute_only, repeat)}
    )


def rs_batch(spec: dict, data: np.ndarray, omegas: np.ndarray, repeat: int, name: str, out: list):
    rows = data[:, None].tolist()

    def go():
        for omega in omegas:
            s = json.loads(json.dumps(spec))
            s["nodes"][1].setdefault("params", {})["tonic_volatility"] = float(omega)
            net = g.build_rs(s)
            net.input_data(input_data=rows, record_trajectories=False)

    go()
    out.append({"name": f"{name}/rshgf_loop", "unit": "ms", "value": median_ms(go, repeat)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--out", default=str(HERE.parents[1] / "bench_python.json"))
    ap.add_argument("--skip-jax", action="store_true")
    ap.add_argument("--jax-only", action="store_true", help="skip rshgf")
    args = ap.parse_args()

    data = load_inputs()
    cont = np.asarray(data["continuous"])
    binary = np.asarray(data["binary"])
    usd = np.asarray(data["usdchf"])
    omegas = np.asarray(data["omegas"])

    spec3 = data["specs"]["continuous_3level"]
    specb = data["specs"]["binary_3level"]
    spec2 = data["specs"]["continuous_2level"]

    out: list[dict] = []
    for name, spec, x in [
        ("continuous_3level_usdchf_614", spec3, usd),
        ("continuous_3level_10k", spec3, cont),
        ("binary_3level_10k", specb, binary),
    ]:
        if not args.jax_only:
            rs_series(spec, x, args.repeat, name, out)
        if not args.skip_jax:
            jax_series(spec, x, args.repeat, name, out)
        print(*(f"{r['name']}: {r['value']:.3f} ms" for r in out if r["name"].startswith(name)), sep="\n")

    name = "batch_1000x1000_continuous_2level"
    if not args.jax_only:
        rs_batch(spec2, cont[:N_BATCH_TRIALS], omegas, max(1, args.repeat // 2), name, out)
    if not args.skip_jax:
        jax_batch(spec2, cont[:N_BATCH_TRIALS], omegas, args.repeat, name, out)
    print(*(f"{r['name']}: {r['value']:.3f} ms" for r in out if r["name"].startswith(name)), sep="\n")

    Path(args.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
