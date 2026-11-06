"""Hierarchical NUTS with pyhgf + JAX, for comparison with ``hgf_fit``'s
``sampleHierarchical``.

Samples the problem written by
``packages/hgf_fit/benchmark/bin/hierarchical_benchmark.dart --out problem.json``:
pyhgf's multilevel notebook (``3-Multilevel_HGF``), a binary 2-level HGF per
agent with ``binary_softmax_inverse_temperature`` decisions, and

    μ_ω ~ N(−5, 5),  σ_ω ~ HalfNormal(10),  ω₂ᵢ ~ N(μ_ω, σ_ω)
    μ_β ~ N(0, 2),   σ_β ~ HalfNormal(2),   βᵢ ~ LogNormal(μ_β, σ_β)

- ``--backend pymc``: the notebook's route: ``vmap`` over agents, exposed with
  ``pytensor.wrap_jax`` as a ``pm.CustomDist``, ``pm.sample(backend="jax")``
  (PyMC's NUTS, JAX-compiled log density). Centred, as in the notebook, or
  ``--noncentered``.
- ``--backend numpyro``: the same model in NumPyro, jitted whole.

    uv run --with numpyro python fit_hier_jax.py problem.json --backend numpyro

Parametrisation defaults to the one hgf_fit used (stored in the problem).
Prints wall time (NumPyro's compile time separately), divergences, bulk ESS
(all parameters and the group-level ones) and ESS per second, and how far the
posterior means are from hgf_fit's (in units of their combined Monte Carlo
error). Run it on one core (``taskset -c 0``) to match hgf_fit.
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
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from arviz_stats.base import array_stats  # noqa: E402

from pyhgf.model import Network  # noqa: E402
from pyhgf.response import binary_softmax_inverse_temperature  # noqa: E402


def make_logp(u, responses):
    def agent_logp(omega, beta, resp):
        return -(
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

    def logp(omega, beta, resp=responses):
        return jax.vmap(agent_logp)(omega.ravel(), beta.ravel(), resp).sum()

    return logp


def run_numpyro(logp, n, centered, chains, warmup, draws, seed):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    def model():
        mu_w = numpyro.sample("omega2_mu", dist.Normal(-5.0, 5.0))
        sd_w = numpyro.sample("omega2_sigma", dist.HalfNormal(10.0))
        mu_b = numpyro.sample("inverseTemperature_mu", dist.Normal(0.0, 2.0))
        sd_b = numpyro.sample("inverseTemperature_sigma", dist.HalfNormal(2.0))
        with numpyro.plate("subject", n):
            if centered:
                omega = numpyro.sample("omega2", dist.Normal(mu_w, sd_w))
                log_b = numpyro.sample("log_beta", dist.Normal(mu_b, sd_b))
            else:
                z_w = numpyro.sample("z_omega2", dist.Normal(0.0, 1.0))
                z_b = numpyro.sample("z_beta", dist.Normal(0.0, 1.0))
                omega = numpyro.deterministic("omega2", mu_w + sd_w * z_w)
                log_b = mu_b + sd_b * z_b
        numpyro.deterministic("inverseTemperature", jnp.exp(log_b))
        numpyro.factor("hgf", logp(omega, jnp.exp(log_b)))

    def mcmc(w, d, c):
        return MCMC(
            NUTS(model),
            num_warmup=w,
            num_samples=d,
            num_chains=c,
            chain_method="sequential",
            progress_bar=False,
        )

    t0 = time.perf_counter()
    mcmc(2, 2, 1).run(jax.random.key(seed + 1))
    compile_s = time.perf_counter() - t0
    m = mcmc(warmup, draws, chains)
    t0 = time.perf_counter()
    m.run(jax.random.key(seed), extra_fields=("diverging", "num_steps"))
    s = {k: np.asarray(v) for k, v in m.get_samples(group_by_chain=True).items()}
    wall = time.perf_counter() - t0
    extra = m.get_extra_fields()
    return (
        wall,
        compile_s,
        int(np.sum(extra["diverging"])),
        int(np.sum(extra["num_steps"])),
        s,
    )


def run_pymc(logp, n, centered, responses, chains, warmup, draws, seed):
    import pymc as pm
    from pytensor import wrap_jax

    @wrap_jax
    def two_level_logp(value, omega, beta):
        return logp(omega, beta, value)

    with pm.Model():
        mu_w = pm.Normal("omega2_mu", -5, 5)
        sd_w = pm.HalfNormal("omega2_sigma", 10)
        mu_b = pm.Normal("inverseTemperature_mu", 0, 2)
        sd_b = pm.HalfNormal("inverseTemperature_sigma", 2)
        if centered:
            omega = pm.Normal("omega2", mu=mu_w, sigma=sd_w, shape=n)
            beta = pm.LogNormal(
                "inverseTemperature", mu=mu_b, sigma=sd_b, shape=n
            )
        else:
            z_w = pm.Normal("z_omega2", 0, 1, shape=n)
            z_b = pm.Normal("z_beta", 0, 1, shape=n)
            omega = pm.Deterministic("omega2", mu_w + sd_w * z_w)
            beta = pm.Deterministic(
                "inverseTemperature", pm.math.exp(mu_b + sd_b * z_b)
            )
        pm.CustomDist(
            "log_likelihood", omega, beta, logp=two_level_logp,
            observed=np.asarray(responses),
        )
        t0 = time.perf_counter()
        idata = pm.sample(
            draws=draws,
            tune=warmup,
            chains=chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            backend="jax",
        )
        wall = time.perf_counter() - t0
    post = idata.posterior
    stats = idata.sample_stats
    s = {k: np.asarray(post[k]) for k in post.data_vars}
    return (
        wall,
        0.0,
        int(np.asarray(stats["diverging"]).sum()),
        int(np.asarray(stats["n_steps"]).sum()),
        s,
    )


def flat_params(s, n):
    """Draws (chain, draw) of each parameter in hgf_fit's joint order."""
    out = {}
    for p in ("omega2", "inverseTemperature"):
        out[f"{p}_mu"] = s[f"{p}_mu"]
        out[f"{p}_sigma"] = s[f"{p}_sigma"]
    for i in range(n):
        for p in ("omega2", "inverseTemperature"):
            out[f"{p}[{i}]"] = s[p][..., i]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--backend", choices=["numpyro", "pymc"], default="numpyro")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--centered", action="store_true", default=None)
    group.add_argument("--noncentered", dest="centered", action="store_false")
    ap.add_argument("--chains", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--draws", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(args.problem) as f:
        prob = json.load(f)
    dart = prob["dart"]
    centered = dart["centered"] if args.centered is None else args.centered
    chains = args.chains or dart["chains"]
    warmup = args.warmup or dart["warmup"]
    draws = args.draws or dart["draws"]
    u = np.asarray(prob["inputs"], dtype=np.float64)
    responses = jnp.asarray(prob["responses"], dtype=jnp.float64)
    n = responses.shape[0]
    logp = make_logp(u, responses)
    if args.backend == "numpyro":
        res = run_numpyro(logp, n, centered, chains, warmup, draws, args.seed)
    else:
        res = run_pymc(
            logp, n, centered, responses, chains, warmup, draws, args.seed
        )
    wall, compile_s, div, steps, samples = res
    params = flat_params(samples, n)
    kw = {"chain_axis": 0, "draw_axis": 1}
    ess = {k: float(array_stats.ess(v, method="bulk", **kw))
           for k, v in params.items()}
    rhat = max(float(array_stats.rhat(v, method="rank", **kw))
               for v in params.values())
    vals = sorted(ess.values())
    group_min = min(v for k, v in ess.items() if k.endswith(("_mu", "_sigma")))
    print(
        f"{args.backend} NUTS ({'centred' if centered else 'non-centred'}), "
        f"{n} agents, {chains} chains x ({warmup} + {draws}): {wall:.1f} s"
        + (f" (+{compile_s:.1f} s compile)" if compile_s else "")
        + f", {steps} leapfrog steps after warmup, {div} divergences, bulk "
        f"ESS min {vals[0]:.0f} ({vals[0] / wall:.1f}/s), median "
        f"{vals[len(vals) // 2]:.0f} ({vals[len(vals) // 2] / wall:.1f}/s), "
        f"group min {group_min:.0f} ({group_min / wall:.1f}/s), max R̂ "
        f"{rhat:.3f}"
    )
    ms = dart["ms"]
    print(
        f"hgf_fit ({'centred' if dart['centered'] else 'non-centred'}): "
        f"{ms / 1e3:.1f} s, {dart['leapfrog']} leapfrog steps after warmup, "
        f"{dart['divergences']} divergences, bulk ESS min "
        f"{dart['min_ess_bulk']:.0f} ({dart['min_ess_bulk'] / ms * 1e3:.1f}/s)"
        f", median {dart['median_ess_bulk']:.0f} "
        f"({dart['median_ess_bulk'] / ms * 1e3:.1f}/s), group min "
        f"{dart['group_min_ess_bulk']:.0f} "
        f"({dart['group_min_ess_bulk'] / ms * 1e3:.1f}/s)"
    )
    z = []
    worst = ("", 0.0)
    for name, d_mean, d_mcse in zip(
        dart["names"], dart["mean"], dart["mcse_mean"]
    ):
        v = params[name]
        mcse = float(array_stats.mcse(v, method="mean", **kw))
        zi = (float(v.mean()) - d_mean) / np.hypot(mcse, d_mcse)
        z.append(zi)
        if abs(zi) > abs(worst[1]):
            worst = (name, zi)
    z = np.asarray(z)
    print(
        f"posterior means vs hgf_fit: rms z {np.sqrt(np.mean(z**2)):.2f}, "
        f"max |z| {abs(worst[1]):.2f} ({worst[0]}) "
        f"(z = Δmean / combined MCSE)"
    )


if __name__ == "__main__":
    main()
