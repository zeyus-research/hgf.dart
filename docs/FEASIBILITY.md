# hgf.dart: feasibility report

Status: phases 1–3 of the spike are done (scaffold and parity harness, minimal
engine, performance gate). **Verdict: go.** On equal float64 arithmetic, the
pure-Dart engine runs at the speed of pyhgf's Rust backend and its JAX backend.
The one clear exception is batched fitting against JAX in its default float32
mode. Phases 4–7 (remaining node types, Flutter playground, fitting, deep/GPU)
are still to do.

Reproduce everything on this page with:

```sh
melos bootstrap
melos run fixtures           # pyhgf -> test/fixtures (needs ../pyhgf, uv, cargo)
melos run test:parity        # Dart VM
CHROME_EXECUTABLE=/usr/bin/chromium melos run test:web   # dart2js + dart2wasm
melos run bench              # all runtimes -> docs/benchmark_results.md
```

## 1. Does the HGF need numpy / pandas?

No, not for the HGF itself. The nodalised `Network` steps through a small
graph doing scalar arithmetic per node. In pyhgf, JAX supplies four things:
- speed (`jit`/`scan`)
- batching (`vmap`)
- `grad` (non-linear coupling functions and gradient-based fitting)
- the PyMC bridge

None of these is an array-library need:

| pyhgf uses | hgf.dart answer |
|---|---|
| `jnp` arrays of node attributes | one `Float64List` per attribute (struct of arrays), CSR edge lists |
| `jit` + `lax.scan` | AOT-compiled loop over an int-coded update schedule, no allocation per step |
| `grad` of coupling functions | closed-form g, g′, g″ per named function, as rshgf already does |
| `node_trajectories` / `to_pandas()` | columnar `Trajectories` with `toCsv()` / `toJson()` |
| `vmap` over parameters | isolates (one per core); see §3 for where this falls short |
| PyMC / NUTS | not needed for filtering; phase 6 probes dual-number gradients + HMC |

A dataframe library would only matter for analysis around the model (group
stats, plotting, model comparison). Exporting CSV/JSON to pandas, R or ArviZ
covers that. It doesn't block the port.

## 2. Parity with pyhgf

`tool/parity/gen_fixtures.py` builds 25 scenarios in pyhgf (pinned commit in
each fixture). The Dart port is checked against them:

- **Network shapes:** continuous 2/3-level, a volatility parent on the input,
  two inputs with shared value and volatility parents, non-unit coupling
  strengths, `tanh`/`sigmoid` coupling functions, binary 2/3-level.
- **Update variants:** all three volatility updates (standard, eHGF,
  unbounded), plus the mean-field variants.
- **Inputs:** missing observations and irregular time steps.

| check | result |
|---|---|
| Dart vs pyhgf **JAX (float64)**, every node/field/time step | max relative error ≤ 1e-9 on 24/25 scenarios |
| Dart vs **rshgf**, update schedule | identical on all scenarios |
| Dart vs **rshgf**, the one scenario where JAX and rshgf differ | ≤ 1e-9 (Dart follows rshgf's arithmetic) |
| same suite compiled with dart2js (Node, Chrome) and dart2wasm (Chrome) | passes at the same tolerance |

JAX and rshgf agree to about 1e-12 except at an ill-conditioned spike of
`continuous_2level_eHGF` (3.7e-7, where node 1's predicted precision dips to
2e-4). For the `continuous_2level_ehgf_missing` scenario, the generator uses
better-conditioned parameters: with π_u = 1e4 and ω1 = −8, the
missing-observation blocks drive μ2 to about −40, and backends that agree to
1e-12 then drift apart by O(1). That is a property of the filter in that
regime, not a porting bug. pyhgf maintainers may still find it interesting.

## 3. Performance gate

Machine: Intel i7-7700 (4 cores / 8 threads), Linux, Dart 3.13.7, Node 26.
Median of 9 warm runs, ms. Full table: [benchmark_results.md](benchmark_results.md).

| scenario | Dart AOT | Dart AOT, 8 isolates | rshgf (native Rust) | rshgf (via Python) | JAX f64 (jit scan) | JAX f32 (jit scan) |
|---|--:|--:|--:|--:|--:|--:|
| 3-level continuous, usdchf (614 trials) | **0.76** | | 0.83 | 1.04 | 1.51 | 1.28 |
| 3-level continuous, 10k trials | 13.8 | | 18.9 | 24.7 | 14.6 | **11.8** |
| 3-level binary, 10k trials | 7.33 | | 8.80 | 11.7 | 7.27 | **6.26** |
| batch: 1,000 parameter sets × 1k trials | 509 | 102 | 661 | 830 | 134 (vmap) | **41.6** (vmap) |
| one-off cost before the first result | ~0 | | ~0 | ~0 | 480–930 (trace + compile) | 420–840 |

Notes:
- "With trajectories" is the default for the series rows (as `input_data`
  is normally used). Without recording, Dart AOT takes 9.2 ms on the
  3-level 10k scenario; native rshgf takes 11.5 ms.
- On the web: JS and Wasm in Node run 1.5–2× slower than AOT, about the
  speed of rshgf called from Python.
- The cost is dominated by transcendentals. The standard/eHGF updates cost
  about 190–220 ns per 4-node trial on Dart AOT; the unbounded update (the
  default) costs about 1 µs, because each update makes about 20 exp/log
  calls (Lambert-W iterations, logaddexp, softmax). rshgf does the same work.

**Update after Phase W1–W3 (2026-11-05)**, same machine
(`docs/benchmark_results.md`): Dart AOT 9.20 ms on the 3-level 10k
scenario (6.59 without recording), Wasm 13.6 ms (9.72), and the batch on
8 AOT isolates 84.6 ms. That makes Dart AOT 1.4× faster than JAX f64 and
1.2× faster than JAX f32 on single series. Wasm is now about the speed of
JAX f64. JAX f32 `vmap` still wins the batch, by 2.3×.

**Update after W4 (2026-11-05)**: `BatchNetwork`, a lock-step batch with
stage-wise kernels, runs the batch in 62.2 ms on 8 AOT isolates (263 ms on
one) and in 443 ms on Wasm. That is 2.1× faster than JAX f64 `vmap`
(133 ms). JAX f32 `vmap` (37.6 ms) is still 1.65× faster. Each element is
bit-identical to `Network`.

**Update with the GPU (2026-11-11)**, same machine plus an RX 7600 XT
([benchmark_results.md](benchmark_results.md), which also has the NUTS and
deep-network comparisons). hgf_gpu runs the batch end to end in 2.6 ms on
wgpu-native and 4.0 ms on WebGPU in Chromium, 15× faster than JAX f32
`vmap` on the CPU (37.7 ms). pyhgf's JAX on the same GPU (ROCm,
`tool/parity/jax_rocm.sh`) gains nothing: 39.8 ms in f32 and 49.5 ms in f64.
Its `lax.scan` launches kernels for every trial (~40 µs each), while
hgf_gpu runs the whole trial loop in one dispatch. Single series on JAX-GPU
take 0.2–3.5 s for the same reason.

- **NUTS** (`bench:nuts-sweep`, 500 + 500, the same models on both sides
  through hgf_fit's pyhgf export):
  - On the binary models hgf_fit leads in min-ESS/s at 20 agents: the GPU
    by 4–9× over NumPyro on one core, and 1 core is on par with NumPyro or
    ahead.
  - At 200 agents, NumPyro's joint sampler takes over 15 min, while
    hgf_fit's GPU run takes 8–15 s.
  - NumPyro on the GPU needs ~40 ms per gradient (0.6 ms on the CPU):
    19 min for the b2 cell that hgf_gpu samples in 6 s.
  - **Open issue:** on the continuous models, hgf_fit's per-agent chains
    mix worse than the joint samplers. At 20 agents its min ESS is 23–137
    against NumPyro/PyMC's 310–400. At 200 agents it does not converge
    (R̂ up to 1.4), though it is still ahead in ESS/s. Warmup adaptation on
    weakly identified ω₂ is the first suspect.
- **Deep networks** (`bench:deep-sweep`):
  - hgf_gpu's batchUpdate stays at 0.4–1.6 ms from H = 8 to 512; JAX on the
    GPU takes ~2 ms.
  - JAX-GPU's predict is flat at 0.6 ms and beats hgf_gpu's from H = 128.
  - On one core, hgf_deep trails JAX and rshgf beyond H ≈ 32 (scalar code,
    no SIMD). It is fastest only at H = 8.

Gate criteria (from the plan):

| criterion | result |
|---|---|
| Dart AOT within 2× of rshgf per trial | **pass**: Dart is 0.7–0.9× of native rshgf (faster) |
| Dart AOT at least as fast as warm JAX on single networks | **pass vs float64** (0.5–1.0×); **close vs float32** (1.0–1.2× slower on 10k series; faster on short series) |
| batch within 3× of JAX `vmap` on CPU | **pass vs float64** (0.47× with 8 isolates, `BatchNetwork`); **pass vs float32** (1.65× with 8 isolates; 7× on one isolate) |

Why the Dart port beats rshgf on equal arithmetic (likely reasons; not
profiled in Rust):
- rshgf compares `node_type` strings inside the posterior loop.
- It allocates `Vec`s in every unbounded update.
- It clones vectors when recording trajectories.

These are easy wins for rshgf too.

Why JAX float32 wins the batch: `vmap` turns 1,000 independent filters into
SIMD-vectorised exp/log over the batch. Dart has no auto-vectorisation, and
`dart:math` transcendentals are scalar. Ways to close the gap if fitting
throughput matters:
- a lock-step batched engine (`[node][batch]` arrays). Done as
  `BatchNetwork` (W4), with scalar stage-wise kernels; a `Float64x2`/
  `Float32x4` polynomial exp/log variant (AOT only) is still open;
- the GPU route of phase 7;
- leaving large-scale fitting in pyhgf.

## 4. Verdict by use case

| use case | verdict | evidence / remaining risk |
|---|---|---|
| Interactive teaching / demos (Flutter, incl. web) | **Go, strong fit** | Sub-millisecond full re-runs of a 600-trial HGF on AOT, about 1.5 ms on Wasm, with no JIT warm-up. Same numbers on every platform. |
| Real-time experiments (trial by trial, on device) | **Go, strong fit** | `Network.predict()` + `step()` cost about 1 µs per trial on a 4-node unbounded HGF and allocate nothing. Deterministic and offline. |
| Research analysis (fitting) | **Feasible for MAP/small-scale; keep pyhgf for heavy MCMC** | Forward passes are as fast as rshgf and float64 JAX. Not built yet: gradients (dual numbers, phase 6), samplers, the ArviZ ecosystem. On batched forward passes, `BatchNetwork` on 8 isolates is 2.1× faster than JAX float64 `vmap`; float32 `vmap` is 1.65× faster than Dart. |
| Deep / vectorised networks (GPU) | **Unknown, phase 7** | Not started. It depends on Flutter GPU compute or a WebGPU binding maturing; the CPU baseline would be a `Float32List` matmul port of `src/vectorised/layer.rs`. |

## 5. Scope of the current port

Implemented (`packages/hgf`):
- **Nodes:** continuous-state and binary-state nodes; value and volatility
  coupling with strengths.
- **Coupling functions:** linear, relu, sigmoid, tanh, leaky_relu, gelu.
- **Updates:** relaxed (rHGF) and mean-field updates; standard, eHGF and
  unbounded volatility updates.
- **Inputs:** missing observations; irregular time steps.
- **API:**
  - `inputData` (whole series), `step` (real time), `append`, `reset`
  - `predict` (the upcoming trial's prediction, before the observation)
  - `setInitial` (parameters for fitting loops)
  - `NetworkSpec` (JSON)

Not yet (phase 4): constant-state, exponential-family, categorical and
Dirichlet nodes, surprise / response functions, sampling ("generative") mode,
and custom (non-named) coupling functions.

Semantics to know about:
- **Coupling functions belong to the parent.** As in rshgf, a coupling
  function is declared on the parent and applies to all its value children.
  pyhgf's JAX backend allows one per child.
- **Input nodes are overridden.** As in pyhgf's JAX backend, continuous
  input nodes always get `autoconnection_strength = 0` and
  `tonic_volatility = 0`, even if they were set explicitly.
- **ω1 and μ2 are not separately identifiable.** They only enter as
  ω1 + κ·μ2. On long series, a shift in ω1 is absorbed by μ2 and μ1 is
  unchanged (see `test/network_test.dart`). This matters for phase 6.

## 6. Also noticed in pyhgf

- `LICENSE` is MIT while `pyproject.toml` declares `GPL-3.0`. hgf.dart uses
  MIT, as liblsl.dart does; worth settling upstream.
- rshgf's hot-path allocations and string comparisons (§3).
