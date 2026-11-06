"""How far pyhgf's float32 mode (JAX's default) is from float64, on the
phase 7 GPU probe's accuracy problems: each scenario's total surprise for
1000 tonic volatilities of node 1, over the first 1000 trials of the
benchmark inputs (as ``packages/hgf_gpu/benchmark/web/gpu_bench.dart``).

    uv run python gpu_f32_reference.py

Runs the vmapped filter in a float64 and a float32 subprocess and prints,
per scenario, the relative error of the f32 totals against the f64 ones:
the yardstick for the GPU's f32 kernels.

    uv run python gpu_f32_reference.py --fixtures NAME...

does the same for parity fixtures (``packages/hgf/test/fixtures/NAME.json``,
the fixture's own parameters and inputs), with the response the GPU test
uses (total Gaussian surprise, or first-level binary surprise).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

INPUTS = Path(__file__).resolve().parents[2] / "packages/hgf/benchmark/data/inputs.json"
N, TRIALS = 1000, 1000


def scenarios(specs: dict) -> dict:
    out = {}
    for mode in ["standard", "eHGF", "unbounded"]:
        for key, name in [("c2", "continuous_2level"), ("c3", "continuous_3level")]:
            s = json.loads(json.dumps(specs[name]))
            s["network"]["volatility_updates"] = mode
            s["network"]["mean_field_updates"] = False
            if name == "continuous_3level":
                for i in (2, 3):
                    s["nodes"][i]["params"]["tonic_volatility"] = -6.0
            out[f"{key}_{mode}"] = (s, "continuous", -10.0, -4.0)
    for mode in ["standard", "eHGF"]:
        s = json.loads(json.dumps(specs["binary_3level"]))
        s["network"]["volatility_updates"] = mode
        out[f"b3_{mode}"] = (s, "binary", -6.0, -1.0)
    return out


FIXTURES = INPUTS.parents[2] / "test/fixtures"


def _decode(v):
    return float(v) if not isinstance(v, str) else float(v.replace("Infinity", "inf"))


def fixture_worker(names: list[str]) -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp
    import numpy as np
    from jax.lax import scan

    sys.path.insert(0, str(Path(__file__).parent))
    import gen_fixtures as g

    result = {}
    for name in names:
        fx = json.loads((FIXTURES / f"{name}.json").read_text())
        spec = fx["spec"]
        x = np.asarray([[_decode(v) for v in row] for row in fx["inputs"]])
        n = x.shape[0]
        obs = (
            np.asarray([[_decode(v) for v in row] for row in fx["observed"]]).astype(int)
            if fx.get("observed")
            else np.ones_like(x, dtype=int)
        )
        steps = np.asarray([_decode(v) for v in fx["time_steps"]]) if fx.get("time_steps") else np.ones(n)
        net = g.build_jax(spec)
        net.input_data(input_data=x[:2])
        ins = (tuple(jnp.asarray(x[:, k]) for k in range(x.shape[1])),
               tuple(jnp.asarray(obs[:, k]) for k in range(x.shape[1])),
               jnp.asarray(steps), None)
        _, traj = scan(net.scan_fn, net.attributes, ins)
        kinds = [node["kind"] for node in spec["nodes"]]
        if kinds[0] == "binary-state":
            u = traj[0]["mean"]
            m = jnp.clip(traj[0]["expected_mean"], 1e-6, 1 - 1e-6)
            s = jnp.where(u != 0, -jnp.log(m), -jnp.log(1 - m))
        else:
            s = jnp.zeros(n)
            for i, k in enumerate(kinds):
                if k != "continuous-state":
                    continue
                d = traj[i]["mean"] - traj[i]["expected_mean"]
                ep = traj[i]["expected_precision"]
                s = s + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(ep) + ep * d * d)
        result[name] = float(jnp.where(jnp.isnan(s), jnp.inf, s).sum())
    json.dump(result, sys.stdout)


def worker() -> None:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.lax import scan

    sys.path.insert(0, str(Path(__file__).parent))
    import gen_fixtures as g

    inputs = json.loads(INPUTS.read_text())
    result = {}
    for name, (spec, data_key, lo, hi) in scenarios(inputs["specs"]).items():
        x = np.asarray(inputs[data_key][:TRIALS])[:, None]
        net = g.build_jax(spec)
        net.input_data(input_data=x[:2])
        attrs = net.attributes
        n = x.shape[0]
        ins = ((jnp.asarray(x),), (jnp.ones(n, dtype=int),), jnp.ones(n), None)
        kinds = [node["kind"] for node in spec["nodes"]]

        def total(omega, attrs=attrs, ins=ins, kinds=kinds):
            a = jax.tree_util.tree_map(lambda v: v, attrs)
            a[1] = dict(a[1])
            a[1]["tonic_volatility"] = omega
            _, traj = scan(net.scan_fn, a, ins)
            s = jnp.zeros(n)
            if kinds[0] == "binary-state":
                u = traj[0]["mean"]
                m = jnp.clip(traj[0]["expected_mean"], 1e-6, 1 - 1e-6)
                s = jnp.where(u != 0, -jnp.log(m), -jnp.log(1 - m))
            else:
                for i, k in enumerate(kinds):
                    if k != "continuous-state":
                        continue
                    d = traj[i]["mean"] - traj[i]["expected_mean"]
                    ep = traj[i]["expected_precision"]
                    s = s + 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(ep) + ep * d * d)
            return jnp.where(jnp.isnan(s), jnp.inf, s).sum()

        omegas = jnp.linspace(lo, hi, N)
        result[name] = [float(v) for v in jax.jit(jax.vmap(total))(omegas)]
    json.dump(result, sys.stdout)


def fixtures_main(names: list[str]) -> None:
    runs = {}
    for x64 in ("1", "0"):
        env = dict(os.environ, JAX_ENABLE_X64=x64, JAX_PLATFORMS="cpu")
        out = subprocess.run(
            [sys.executable, "-I", __file__, "--fixture-worker", *names],
            env=env, check=True, capture_output=True, text=True, cwd=Path(__file__).parent,
        )
        runs[x64] = json.loads(out.stdout)
    import math

    print("| fixture | f64 total | f32 total | rel err |")
    print("|---|--:|--:|--:|")
    for name in names:
        a, b = runs["1"][name], runs["0"][name]
        rel = abs(a - b) / max(abs(a), 1.0) if math.isfinite(a) and math.isfinite(b) else float("nan")
        print(f"| {name} | {a:.6g} | {b:.6g} | {rel:.2g} |")


def main() -> None:
    runs = {}
    for x64 in ("1", "0"):
        env = dict(os.environ, JAX_ENABLE_X64=x64, JAX_PLATFORMS="cpu")
        out = subprocess.run(
            [sys.executable, "-I", __file__, "--worker"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        runs[x64] = json.loads(out.stdout)
    import math

    print("| scenario | median rel err | p99 | max | non-finite: both / mismatch | argmin f64 / f32 |")
    print("|---|--:|--:|--:|--:|--:|")
    for name, ref in runs["1"].items():
        got = runs["0"][name]
        rel, both, mism = [], 0, 0
        for a, b in zip(ref, got):
            if not (math.isfinite(a) and math.isfinite(b)):
                if math.isfinite(a) == math.isfinite(b):
                    both += 1
                else:
                    mism += 1
                continue
            rel.append(abs(a - b) / max(abs(a), 1.0))
        rel.sort()
        lo, hi = (-6.0, -1.0) if name.startswith("b3") else (-10.0, -4.0)
        grid = [lo + (hi - lo) * k / (N - 1) for k in range(N)]
        am = lambda xs: grid[min(range(N), key=lambda k: xs[k])]  # noqa: E731
        med = rel[len(rel) // 2] if rel else float("nan")
        p99 = rel[int(len(rel) * 0.99)] if rel else float("nan")
        mx = rel[-1] if rel else float("nan")
        print(f"| {name} | {med:.2g} | {p99:.2g} | {mx:.2g} | {both} / {mism} | {am(ref):.3f} / {am(got):.3f} |")


if __name__ == "__main__":
    if "--fixture-worker" in sys.argv:
        fixture_worker(sys.argv[sys.argv.index("--fixture-worker") + 1 :])
    elif "--fixtures" in sys.argv:
        fixtures_main(sys.argv[sys.argv.index("--fixtures") + 1 :])
    elif "--worker" in sys.argv:
        worker()
    else:
        main()
