"""MAP fitting with pyhgf + JAX, for comparison with ``hgf_fit``.

Fits the problem written by ``packages/hgf_fit/benchmark/bin/fit_benchmark.dart
--out problem.json`` (pyhgf's parameter-recovery notebook: binary 2-level HGF,
``binary_softmax_inverse_temperature`` responses, ω₂ and β free per agent) the
way one would with JAX: the notebook's per-agent log-probability, ``vmap``-ed
over agents, the same priors and transforms as hgf_fit (truncated normal on
[-8, 2] and uniform on [0.2, 20], logit-transformed), exact gradients from
``jax.value_and_grad``, and SciPy's L-BFGS-B on all agents' parameters jointly
(the objective is a sum of independent terms).

    uv run python fit_map_jax.py problem.json [--repeat 7]

Prints the median warm time, the JIT compile time, and how far the estimates
and objective values are from hgf_fit's.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.stats import norm  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from pyhgf.model import Network  # noqa: E402
from pyhgf.response import binary_softmax_inverse_temperature  # noqa: E402

LO_W, HI_W, MU_W, SD_W = -8.0, 2.0, -3.0, 5.0
LO_B, HI_B = 0.2, 20.0
LOG_Z_W = math.log(
    SD_W * (norm.cdf((HI_W - MU_W) / SD_W) - norm.cdf((LO_W - MU_W) / SD_W))
) + 0.5 * math.log(2 * math.pi)


def to_constrained(y, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(y)


def to_unconstrained(x, lo, hi):
    p = (x - lo) / (hi - lo)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--repeat", type=int, default=7)
    args = ap.parse_args()
    prob = json.load(open(args.problem))
    u = np.asarray(prob["inputs"], dtype=np.float64)
    responses = jnp.asarray(prob["responses"], dtype=jnp.float64)
    n = responses.shape[0]

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

    def objective(y):
        y = y.reshape(n, 2)
        omega = to_constrained(y[:, 0], LO_W, HI_W)
        beta = to_constrained(y[:, 1], LO_B, HI_B)
        nll = jax.vmap(agent_nll)(omega, beta, responses)
        z = (omega - MU_W) / SD_W
        log_prior = -0.5 * z * z - LOG_Z_W - math.log(HI_B - LO_B)
        return jnp.sum(nll - log_prior), nll - log_prior

    value_and_grad = jax.jit(jax.value_and_grad(lambda y: objective(y)[0]))
    per_agent = jax.jit(lambda y: objective(y)[1])

    def fun(y):
        v, g = value_and_grad(jnp.asarray(y))
        return float(v), np.asarray(g, dtype=np.float64)

    y0 = np.tile(
        [to_unconstrained(MU_W, LO_W, HI_W), to_unconstrained(1.0, LO_B, HI_B)],
        n,
    )
    t0 = time.perf_counter()
    jax.block_until_ready(value_and_grad(jnp.asarray(y0)))
    compile_s = time.perf_counter() - t0

    times = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B")
        times.append(time.perf_counter() - t0)
    ms = statistics.median(times) * 1e3
    y = res.x.reshape(n, 2)
    est = np.stack(
        [
            np.asarray(to_constrained(y[:, 0], LO_W, HI_W)),
            np.asarray(to_constrained(y[:, 1], LO_B, HI_B)),
        ],
        axis=1,
    )
    post = np.asarray(per_agent(jnp.asarray(res.x)))
    print(
        f"JAX + L-BFGS-B, {n} agents: {ms:.1f} ms (median of {args.repeat}), "
        f"{res.nfev} value+grad calls, {res.nit} iterations, "
        f"compile {compile_s:.2f} s, success={res.success} ({res.message})"
    )
    dart = prob.get("dart", {}).get("lbfgs")
    if dart:
        d_est = np.asarray(dart["estimates"])
        d_post = np.asarray(dart["neg_log_posterior"])
        print(
            f"hgf_fit L-BFGS, {n} agents: {dart['ms']:.1f} ms, "
            f"{dart['passes']} passes ({ms / dart['ms']:.1f}x faster)"
        )
        print(
            "max |Δ estimate| vs hgf_fit: "
            f"ω₂ {np.max(np.abs(est[:, 0] - d_est[:, 0])):.2e}, "
            f"β {np.max(np.abs(est[:, 1] - d_est[:, 1])):.2e}; "
            f"objective (JAX − Dart) min {np.min(post - d_post):+.2e}, "
            f"max {np.max(post - d_post):+.2e}"
        )
        # The Dart objective at its own optimum, evaluated by JAX.
        y_d = np.stack(
            [
                to_unconstrained(d_est[:, 0], LO_W, HI_W),
                to_unconstrained(d_est[:, 1], LO_B, HI_B),
            ],
            axis=1,
        ).reshape(-1)
        post_d = np.asarray(per_agent(jnp.asarray(y_d)))
        print(
            "JAX objective at hgf_fit's estimates − hgf_fit's value: "
            f"max |Δ| {np.max(np.abs(post_d - d_post)):.2e}"
        )


if __name__ == "__main__":
    main()
