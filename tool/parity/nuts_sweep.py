"""NUTS across network types, agent counts and chain counts, Python side:
NumPyro (CPU, one core, float64; and GPU, float32 with vectorised chains)
and PyMC (pyhgf's documented `wrap_jax` route; CPU, one core) on the problems
that `packages/hgf_gpu/benchmark/bin/nuts_sweep.dart` writes. The models are
that program's `model_<name>.py`, generated from the Dart models by
hgf_fit's PythonExport, so both sides sample the same posterior.

    # every cell, each in its own process with a time cap:
    uv run python nuts_sweep.py all ../../.dart_tool/nuts_sweep \\
        --gpu-python .venv-rocm/python-rocm
    # one cell:
    uv run python nuts_sweep.py cell DIR --model b2 --agents 20 --chains 4 \\
        --backend numpyro

Rows are appended to DIR/python.jsonl. The agents are sampled jointly (one
model with 2N parameters), as pyhgf's notebooks do; hgf_fit samples each
agent's posterior separately, which is the same posterior since agents are
independent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_model(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cell(args) -> None:
    os.environ.setdefault("JAX_ENABLE_X64", "1" if args.precision == "f64" else "0")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from arviz_stats.base import array_stats

    d = Path(args.dir)
    prob = json.loads((d / f"problem_{args.model}_{args.agents}.json").read_text())
    mod = load_model(d / f"model_{args.model}.py")
    params = prob["model_json"]["parameters"]
    names = [p["name"] for p in params]
    k = len(names)
    fdt = jnp.float64 if args.precision == "f64" else jnp.float32
    inputs = jnp.asarray(np.asarray(prob["inputs"])[:, None], dtype=fdt)
    responses = jnp.asarray(np.asarray(prob["responses"]), dtype=fdt)
    n = responses.shape[0]

    def agent_nll(theta, resp):
        data = dict(inputs=inputs, observed=None, time_steps=None, responses=resp)
        return mod.negative_log_likelihood(dict(zip(names, theta)), data)

    nll = jax.vmap(agent_nll)  # (n, k), (n, T) -> (n,)

    # Bounded priors on the logit scale, as hgf_fit's transforms do.
    def bounds(p):
        pr = p["prior"]
        return float(pr.get("lower", -math.inf)), float(pr.get("upper", math.inf))

    for p in params:
        lo, hi = bounds(p)
        if not (math.isfinite(lo) and math.isfinite(hi)) or p["prior"]["type"] not in (
            "TruncatedNormal",
            "Uniform",
            "Normal",
        ):
            raise SystemExit(f"prior {p['prior']} not handled here")

    los = jnp.asarray([bounds(p)[0] for p in params], dtype=fdt)
    his = jnp.asarray([bounds(p)[1] for p in params], dtype=fdt)
    mus = jnp.asarray([p["prior"].get("mean", 0.0) for p in params], dtype=fdt)
    sds = jnp.asarray([p["prior"].get("sd", 1.0) for p in params], dtype=fdt)
    gauss = jnp.asarray([p["prior"]["type"] != "Uniform" for p in params])

    def constrain(y):  # (n, k) unconstrained -> constrained
        return los + (his - los) * jax.nn.sigmoid(y)

    def potential(y):
        y = y.reshape(n, k)
        x = constrain(y)
        z = (x - mus) / sds
        log_post = -nll(x, responses).sum() - 0.5 * jnp.sum(jnp.where(gauss, z * z, 0.0))
        log_post += jnp.sum(jnp.log(his - los) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y))
        return -log_post

    start = np.asarray(
        [
            float(p.get("initial", p["prior"].get("mean", 0.5 * (bounds(p)[0] + bounds(p)[1]))))
            for p in params
        ]
    )
    lo_np, hi_np = np.asarray(los), np.asarray(his)
    y0 = np.log((start - lo_np) / (hi_np - start))
    rng = np.random.default_rng(args.seed)
    init = np.tile(y0, (args.chains, n, 1)) + rng.uniform(-1, 1, size=(args.chains, n, k))
    init = init.reshape(args.chains, n * k).astype(np.float64 if args.precision == "f64" else np.float32)

    device = jax.default_backend()
    if args.backend == "numpyro":
        from numpyro.infer import MCMC, NUTS

        method = "vectorized" if device != "cpu" else "sequential"

        def mcmc(w, s, c):
            return MCMC(
                NUTS(potential_fn=potential),
                num_warmup=w,
                num_samples=s,
                num_chains=c,
                chain_method=method,
                progress_bar=False,
            )

        t0 = time.perf_counter()
        mcmc(2, 2, args.chains).run(jax.random.key(args.seed), init_params=jnp.asarray(init))
        compile_s = time.perf_counter() - t0
        m = mcmc(args.warmup, args.draws, args.chains)
        t0 = time.perf_counter()
        m.run(
            jax.random.key(args.seed),
            init_params=jnp.asarray(init),
            extra_fields=("diverging", "num_steps"),
        )
        y = np.asarray(m.get_samples(group_by_chain=True))  # (c, d, n*k)
        wall = time.perf_counter() - t0
        extra = m.get_extra_fields()
        div = int(np.sum(extra["diverging"]))
        x = np.asarray(constrain(jnp.asarray(y).reshape(*y.shape[:2], n, k)))
        tag = f"numpyro_{'gpu' if device != 'cpu' else 'cpu'}_{args.precision}"
    else:
        import pymc as pm
        from pytensor import wrap_jax

        @wrap_jax
        def loglike(theta):
            return -nll(theta, responses).sum()

        with pm.Model():
            cols = []
            for j, p in enumerate(params):
                pr = p["prior"]
                lo, hi = bounds(p)
                iv = np.full(n, start[j])
                if pr["type"] == "Uniform":
                    cols.append(pm.Uniform(p["name"], lo, hi, shape=n, initval=iv))
                else:
                    cols.append(
                        pm.TruncatedNormal(
                            p["name"], pr["mean"], pr["sd"], lower=lo, upper=hi, shape=n, initval=iv
                        )
                    )
            pm.Potential("hgf", loglike(pm.math.stack(cols, axis=1)))
            t0 = time.perf_counter()
            idata = pm.sample(
                draws=args.draws,
                tune=args.warmup,
                chains=args.chains,
                cores=1,
                random_seed=args.seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
            wall = time.perf_counter() - t0
        compile_s = 0.0
        div = int(np.asarray(idata.sample_stats["diverging"]).sum())
        x = np.stack([np.asarray(idata.posterior[nm]) for nm in names], axis=-1)
        tag = f"pymc_cpu_{args.precision}"

    ess, rhat = [], []
    for i in range(n):
        for j in range(k):
            dr = x[:, :, i, j]
            ess.append(float(array_stats.ess(dr, method="bulk", chain_axis=0, draw_axis=1)))
            rhat.append(float(array_stats.rhat(dr, method="rank", chain_axis=0, draw_axis=1)) if args.chains > 1 else float("nan"))
    ess.sort()
    # NumPyro compiles its sampling loop again on every run (fori_loop over a
    # fresh closure); the warm-up run measured that compile, so the
    # sampling time is the rest. PyMC's wall time has no such step.
    wall_total = wall
    wall = max(wall - compile_s, 1e-9)
    row = dict(
        model=args.model,
        agents=n,
        chains=args.chains,
        warmup=args.warmup,
        draws=args.draws,
        backend=tag,
        ms=wall * 1e3,
        wall_ms=wall_total * 1e3,
        compile_ms=compile_s * 1e3,
        min_ess=ess[0],
        median_ess=ess[len(ess) // 2],
        min_ess_per_s=ess[0] / wall,
        median_ess_per_s=ess[len(ess) // 2] / wall,
        max_rhat=max(rhat) if args.chains > 1 else None,
        divergences=div,
        mean=x.mean(axis=(0, 1)).tolist(),
        sd=x.std(axis=(0, 1), ddof=1).tolist(),
        device=str(jax.devices()[0]) if device != "cpu" else "cpu",
    )
    with open(d / "python.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(
        f"{args.model} agents={n} chains={args.chains} {tag}: {wall:.1f} s sampling "
        f"({wall_total:.1f} s wall, {compile_s:.1f} s compile), min ESS {ess[0]:.0f} ({ess[0] / wall:.1f}/s), "
        + (f"max R̂ {max(rhat):.3f}, " if args.chains > 1 else "") + f"{div} divergences",
        flush=True,
    )


def run_all(args) -> None:
    d = Path(args.dir)
    dart = [json.loads(line) for line in (d / "dart.jsonl").read_text().splitlines() if line]
    cells = sorted({(r["model"], r["agents"], r["chains"]) for r in dart}, key=lambda c: (c[0], c[1], c[2]))
    w, s = dart[0]["warmup"], dart[0]["draws"]
    runs = []  # (label, cmd prefix, backend, precision, cell filter)
    one_core = ["taskset", "-c", "0"] if sys.platform.startswith("linux") else []
    if "numpyro_cpu" in args.backends:
        runs.append((one_core + [args.cpu_python], "numpyro", "f64", lambda c: True))
    if "numpyro_gpu" in args.backends and args.gpu_python:
        # JAX's scan costs ~40 ms per gradient on a GPU here (kernel launches
        # every trial; 0.6 ms on the CPU), so by default only the first cell.
        first = cells[0]
        runs.append(([args.gpu_python], "numpyro", "f32",
                     lambda c: args.gpu_all or (c[0], c[1]) == first[:2] and c[2] == 4))
    if "pymc" in args.backends:
        runs.append((one_core + [args.cpu_python], "pymc", "f64", lambda c: c[1] <= args.pymc_max_agents))
    for model, agents, chains in cells:
        for prefix, backend, precision, ok in runs:
            if not ok((model, agents, chains)):
                continue
            cmd = prefix + [
                str(HERE / "nuts_sweep.py"), "cell", str(d),
                "--model", model, "--agents", str(agents), "--chains", str(chains),
                "--warmup", str(w), "--draws", str(s),
                "--backend", backend, "--precision", precision,
            ]
            env = dict(os.environ)
            env.pop("JAX_ENABLE_X64", None)
            if prefix[-1] == args.cpu_python:
                env["JAX_PLATFORMS"] = "cpu"
            is_gpu = prefix[-1] == args.gpu_python
            cap = args.gpu_timeout_s if is_gpu else args.timeout_s
            try:
                subprocess.run(cmd, check=True, timeout=cap, env=env, cwd=HERE)
            except subprocess.TimeoutExpired:
                tag = f"{backend}_{'gpu' if is_gpu else 'cpu'}_{precision}"
                print(f"{model} agents={agents} chains={chains} {tag}: over {cap} s", flush=True)
                with open(d / "python.jsonl", "a") as f:
                    f.write(json.dumps(dict(model=model, agents=agents, chains=chains, warmup=w,
                                            draws=s, backend=tag, timeout_s=cap)) + "\n")
            except subprocess.CalledProcessError as e:
                print(f"{model} agents={agents} chains={chains} {backend}: failed ({e.returncode})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("cell")
    c.add_argument("dir")
    c.add_argument("--model", required=True)
    c.add_argument("--agents", type=int, required=True)
    c.add_argument("--chains", type=int, default=4)
    c.add_argument("--warmup", type=int, default=500)
    c.add_argument("--draws", type=int, default=500)
    c.add_argument("--backend", choices=["numpyro", "pymc"], default="numpyro")
    c.add_argument("--precision", choices=["f64", "f32"], default="f64")
    c.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("all")
    a.add_argument("dir")
    a.add_argument("--cpu-python", default=sys.executable)
    a.add_argument("--gpu-python")
    a.add_argument("--backends", default="numpyro_cpu,numpyro_gpu,pymc")
    a.add_argument("--pymc-max-agents", type=int, default=20)
    a.add_argument("--timeout-s", type=int, default=900)
    a.add_argument("--gpu-timeout-s", type=int, default=1200)
    a.add_argument("--gpu-all", action="store_true", help="NumPyro on the GPU for every cell")
    args = ap.parse_args()
    if args.mode == "cell":
        cell(args)
    else:
        args.gpu_python = str(Path(args.gpu_python).absolute()) if args.gpu_python else None
        run_all(args)


if __name__ == "__main__":
    main()
