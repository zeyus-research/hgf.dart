"""NUTS with pyhgf + JAX, for comparison with ``hgf_fit``'s ``sampleNuts``.

Samples the problem written by ``packages/hgf_fit/benchmark/bin/fit_benchmark.dart
--nuts --out problem.json`` (pyhgf's parameter-recovery notebook: binary 2-level
HGF, ``binary_softmax_inverse_temperature`` responses, ω₂ and β per agent) with
the same priors as hgf_fit (ω₂ ~ N(−3, 5) truncated to [−8, 2], β ~ U(0.2, 20);
both logit-transformed), as one joint model over all agents, the way the
notebook does:

- ``--backend pymc``: the notebook's route, a ``wrap_jax`` log-probability
  (``vmap`` over agents) as a ``pm.Potential``, sampled by PyMC's NUTS.
  (The notebook uses a ``Censored`` prior on ω₂; here it is the truncated
  normal, so the posterior is hgf_fit's.)
- ``--backend numpyro``: the same log density, jitted whole with NumPyro's
  NUTS (exact gradients, everything compiled).

    uv run --with numpyro python fit_nuts_jax.py problem.json --backend numpyro

Prints wall time (compile time separately for NumPyro), divergences, bulk
ESS and ESS per second, and how far the posterior means and sds are from
hgf_fit's (in units of their Monte Carlo error). Run it on one core
(``taskset -c 0``) to compare with hgf_fit's single-thread numbers.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from arviz_stats.base import array_stats  # noqa: E402

from pyhgf.model import Network  # noqa: E402
from pyhgf.response import binary_softmax_inverse_temperature  # noqa: E402

LO_W, HI_W, MU_W, SD_W = -8.0, 2.0, -3.0, 5.0
LO_B, HI_B = 0.2, 20.0


def make_nll(u, responses):
    def agent_nll(omega, beta, resp):
        return (
            Network()
            .add_nodes(kind="binary-state")
            .add_nodes(value_children=0, tonic_volatility=omega)
            .input_data(input_data=u)
            .surprise(
                response_function=binary_softmax_inverse_temperature,
                response_function_inputs=resp,
                response_function_parameters=beta,
            )
        ).sum()

    return lambda omega, beta: jax.vmap(agent_nll)(omega, beta, responses)


def ess_rhat(draws):
    """Bulk ESS and rank R̂ of ``draws`` shaped (chain, draw)."""
    kw = {"chain_axis": 0, "draw_axis": 1}
    return (
        float(array_stats.ess(draws, method="bulk", **kw)),
        float(array_stats.rhat(draws, method="rank", **kw)),
    )


def run_numpyro(nll, n, chains, warmup, draws, seed):
    from numpyro.infer import MCMC, NUTS

    def logit_to(y, lo, hi):
        return lo + (hi - lo) * jax.nn.sigmoid(y)

    def log_jac(y, lo, hi):
        return math.log(hi - lo) + jax.nn.log_sigmoid(y) + jax.nn.log_sigmoid(-y)

    def potential(y):
        omega = logit_to(y[:n], LO_W, HI_W)
        beta = logit_to(y[n:], LO_B, HI_B)
        z = (omega - MU_W) / SD_W
        # Constants dropped: they don't change the posterior.
        log_post = -nll(omega, beta).sum() - 0.5 * jnp.sum(z * z)
        log_post += jnp.sum(log_jac(y[:n], LO_W, HI_W))
        log_post += jnp.sum(log_jac(y[n:], LO_B, HI_B))
        return -log_post

    rng = np.random.default_rng(seed)
    y0 = np.concatenate([np.full(n, 0.0), np.full(n, -2.3)])  # ω −3, β ≈ 1
    init = y0 + rng.uniform(-1, 1, size=(chains, 2 * n))

    mcmc = MCMC(
        NUTS(potential_fn=potential),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        chain_method="sequential",
        progress_bar=False,
    )
    # Compile once on a tiny run, then time a fresh run.
    t0 = time.perf_counter()
    small = MCMC(
        NUTS(potential_fn=potential),
        num_warmup=2,
        num_samples=2,
        num_chains=1,
        progress_bar=False,
    )
    small.run(jax.random.key(seed), init_params=jnp.asarray(init[0]))
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    mcmc.run(
        jax.random.key(seed),
        init_params=jnp.asarray(init),
        extra_fields=("diverging", "num_steps"),
    )
    y = np.asarray(mcmc.get_samples(group_by_chain=True))
    wall = time.perf_counter() - t0
    extra = mcmc.get_extra_fields()
    div = int(np.sum(extra["diverging"]))
    steps = int(np.sum(extra["num_steps"]))
    omega = np.asarray(logit_to(y[..., :n], LO_W, HI_W))
    beta = np.asarray(logit_to(y[..., n:], LO_B, HI_B))
    return wall, compile_s, div, steps, omega, beta


def run_pymc(nll, n, chains, warmup, draws, seed):
    import pymc as pm
    from pytensor import wrap_jax

    @wrap_jax
    def loglike(omega, beta):
        return -nll(omega, beta).sum()

    with pm.Model():
        omega = pm.TruncatedNormal(
            "omega", MU_W, SD_W, lower=LO_W, upper=HI_W, shape=n,
            initval=np.full(n, MU_W),
        )
        beta = pm.Uniform("beta", LO_B, HI_B, shape=n, initval=np.ones(n))
        pm.Potential("hgf_loglike", loglike(omega, beta))
        t0 = time.perf_counter()
        idata = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
        wall = time.perf_counter() - t0
    post = idata.posterior
    stats = idata.sample_stats
    div = int(np.asarray(stats["diverging"]).sum())
    steps = int(np.asarray(stats["n_steps"]).sum())
    return (
        wall, 0.0, div, steps,
        np.asarray(post["omega"]), np.asarray(post["beta"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--backend", choices=["numpyro", "pymc"], default="numpyro")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(args.problem) as f:
        prob = json.load(f)
    u = np.asarray(prob["inputs"], dtype=np.float64)
    responses = jnp.asarray(prob["responses"], dtype=jnp.float64)
    n = responses.shape[0]
    nll = make_nll(u, responses)
    run = run_numpyro if args.backend == "numpyro" else run_pymc
    wall, compile_s, div, steps, omega, beta = run(
        nll, n, args.chains, args.warmup, args.draws, args.seed
    )
    ess = []
    rhat = []
    for draws in (omega, beta):
        for i in range(n):
            e, r = ess_rhat(draws[..., i])
            ess.append(e)
            rhat.append(r)
    min_ess = min(ess)
    med_ess = float(np.median(ess))
    print(
        f"{args.backend} NUTS, {n} agents jointly, {args.chains} chains x "
        f"({args.warmup} + {args.draws}): {wall:.1f} s"
        + (f" (+{compile_s:.1f} s compile)" if compile_s else "")
        + f", {steps} leapfrog steps after warmup, {div} divergences, "
        f"bulk ESS min {min_ess:.0f} ({min_ess / wall:.1f}/s), median "
        f"{med_ess:.0f} ({med_ess / wall:.1f}/s), max R̂ {max(rhat):.3f}"
    )
    dart = prob.get("dart", {}).get("nuts")
    if dart:
        d_mean = np.asarray(dart["mean"])
        d_sd = np.asarray(dart["sd"])
        d_mcse = np.asarray(dart["mcse_mean"])
        mean = np.stack([omega.mean(axis=(0, 1)), beta.mean(axis=(0, 1))], 1)
        sd = np.stack([omega.std(axis=(0, 1), ddof=1),
                       beta.std(axis=(0, 1), ddof=1)], 1)
        mcse = np.stack([
            [array_stats.mcse(d[..., i], method="mean", chain_axis=0,
                              draw_axis=1) for i in range(n)]
            for d in (omega, beta)
        ], 1)
        z = (mean - d_mean) / np.sqrt(mcse**2 + d_mcse**2)
        print(
            f"hgf_fit NUTS, {n} agents: {dart['ms'] / 1e3:.1f} s, "
            f"{dart['divergences']} divergences, bulk ESS min "
            f"{dart['min_ess_bulk']:.0f} "
            f"({dart['min_ess_bulk'] / dart['ms'] * 1e3:.1f}/s), median "
            f"{dart['median_ess_bulk']:.0f} "
            f"({dart['median_ess_bulk'] / dart['ms'] * 1e3:.1f}/s)"
        )
        print(
            "posterior means vs hgf_fit: |z| max "
            f"{np.max(np.abs(z)):.2f}, rms {np.sqrt(np.mean(z**2)):.2f} "
            f"(z = Δmean / combined MCSE); sd ratio "
            f"{np.min(sd / d_sd):.3f}–{np.max(sd / d_sd):.3f}"
        )


if __name__ == "__main__":
    main()
