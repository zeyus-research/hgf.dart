# hgf.dart roadmap

Picks up after the feasibility spike (phases 1–3, see
[FEASIBILITY.md](FEASIBILITY.md)). The original plan lives at
`~/.claude/plans/so-i-m-associated-with-radiant-cloud.md`; this file
supersedes its phase list.

## Where things stand (2026-11-10, after MAP, Laplace, NUTS, exact gradients, hierarchical models, worker pools, GPU fitting, the playground's fitting view and settings, and HGF neural networks on the CPU and GPU)

- `packages/hgf` covers:
  - continuous, binary, constant (bias), exponential-family (`ef-state`)
    nodes, the latter with fixed or HGF-implied (`hgf-1/2/3`) learning,
    categorical nodes (`categorical-state`) and Dirichlet process nodes
    (`dp-state`);
  - standard, eHGF and unbounded volatility updates, relaxed or mean-field;
  - named and custom coupling functions, missing observations, irregular
    time steps;
  - `inputData`, `predict`/`step`, `setInitial`, `Trajectories`,
    `NetworkSpec`;
  - `BatchNetwork`: lock-step batches with per-element parameters and
    data, bit-identical to `Network` (W4);
  - surprise and response functions (`Network.surprise`), generative
    sampling (`Network.sample`), `saveState`/`restoreState`.
- 632 engine tests pass (plus 112 fitting, 16 widget and 27 playground tests). Parity with float64 pyhgf JAX holds to 1e-9 on 61 fixtures
  (4 of them against a patched pyhgf: see `ehgfMgfCorrection`, and
  `dp_missing`; 11 ef-state, 7 categorical, 6 dp-state),
  on the VM, dart2js, dart2wasm, and a dart2wasm `-O4` release build.
  `BatchNetwork` matches `Network` bit for bit on all of them, on the same
  four targets.
- Performance on AOT, after W1/W3 (`docs/benchmark_results.md`): Dart is
  1.9× faster than native rshgf, 1.4× faster than float64 JAX and 1.2×
  faster than float32 JAX on single series. On batched fitting,
  `BatchNetwork` on 8 isolates (62 ms) is 2.1× faster than float64 JAX
  `vmap`; float32 `vmap` is still 1.65× faster (38 ms).
- Performance on Wasm (`-O4`): about 1.6× slower than AOT in every update
  mode, which is about the speed of float64 JAX.

- `packages/hgf_fit` (Phase 6, first part): MAP estimation of per-subject
  parameters with priors and response models, every subject fitted in one
  batched pass per optimiser step. On pyhgf's parameter-recovery problem it
  is 3.2× (20 agents) to 5.4× (200 agents) faster than pyhgf + JAX
  gradients + SciPy L-BFGS-B, single-threaded, and evaluates the same
  objective to 3e-13. `laplace` adds standard errors, intervals and the log
  evidence (within 0.01 nats of quadrature) in one more pass. `sampleNuts`
  draws full posteriors (Stan's NUTS, all chains of all subjects in lock
  step). Gradients are exact (forward-mode tangents through the batched
  filter): MAP is now 6.9× (20 agents) to 13× (200 agents) faster than
  pyhgf + JAX + SciPy, and NUTS on the notebook's 20 agents 3.3× faster
  than NumPyro in wall time on one core and ahead in ESS/s (1.2× min,
  1.08× median).
  `HierarchicalModel` adds group-level inference: `sampleHierarchical`
  (joint NUTS; on pyhgf's multilevel notebook 1.2× NumPyro's and 1.85×
  PyMC's ESS/s on one core) and `fitEmpiricalBayes` (Laplace-EM, 45 ms).
  `FitWorkers` runs all of these on a pool of isolates with bit-identical
  results: 3.5–3.8× on 4 cores for large problems (MAP and NUTS on 200
  agents), 3× for NUTS on 20 agents (42 s → 13.8 s), 2.6× for
  hierarchical NUTS. On the web the same pool runs on Web Workers (a
  separately compiled worker, models as JSON): 3.3–4.2× on 4 cores with
  Wasm workers, 2.1–3× with JS ones. The playground fits on them too
  (MAP + Laplace in ~50–80 ms, NUTS in 11–18 s on 4 Wasm workers, 60 fps
  throughout).

- `packages/hgf_gpu` (Phase 7 probe, 2026-11-07): **GPU is a go.**
  `HgfKernel` compiles a `Network` into a WGSL compute shader (one GPU
  thread per parameter set), and `GpuBatch` runs it on WebGPU in the
  browser or on wgpu-native through `dart:ffi`, from the same host code.
  For 1000 parameter sets × 1000 trials, the end-to-end time is 2.2 ms
  (native) or 2.7–4 ms (browser) on an RX 7600 XT for the 2-level unbounded
  HGF. That is 28× / 15–23× faster than `BatchNetwork` on 8 AOT isolates,
  and 17× / 9–14× faster than JAX float32 `vmap`. At 10⁶ parameter sets it takes
  40–115 ms. Accuracy is f32: the GPU matches f64 to ~1e-6, which is how
  close pyhgf's own float32 mode gets. See Phase 7.
  Since 2026-11-09 (Phase 7b) the fitting functions run on it too:
  `GpuFitter` evaluates hgf_fit's passes with exact gradients from
  forward-mode tangents compiled into the kernel. MAP on 2000 agents takes
  89 ms (8 isolates: 332 ms), and NUTS on 200 agents 5.0–5.4 s (34 s),
  on 2000 agents 30 s (319 s).

- `packages/hgf_deep` (2026-11-09): pyhgf's vectorised `DeepNetwork` (HGF
  neural networks) in f64, matching JAX to 1e-9 on 26 fixtures. Sequential
  `fit` is ahead of pyhgf on small networks and ~2× behind JAX on wider ones.
  Since 2026-11-10 the batched work runs on the GPU too (`GpuDeepNetwork`
  in hgf_gpu, f32, ~1e-6 from f64): a learning step on 256 samples takes
  0.55–0.75 ms on an RX 7600 XT, 1.7–16× faster than JAX f32 on one core
  and 3.6–120× faster than the f64 CPU engine; at 4096 samples per step it is
  40–470× faster than the CPU engine. See "hgf_deep on the GPU".

- CI (`.github/workflows/ci.yml`, every push and PR): lint, VM + Flutter
  tests, parity on dart2js/dart2wasm in Chrome and Node, and the `-O4`
  parity build. `pages.yml` deploys the playground (Wasm `-O4`, JS
  fallback) to GitHub Pages after CI passes on `main`.

## Decisions so far

| decision | why |
|---|---|
| Pure-Dart engine, no FFI to rshgf | runs everywhere, including the web; FFI stays a fallback |
| melos/pub workspace laid out like liblsl.dart; packages created only when their phase starts | independent builds, tests and publishing; low maintenance |
| Port from rshgf's imperative code, use JAX as the semantic reference | rshgf is closest to the Dart shape; JAX is canonical |
| Struct-of-arrays `Float64List`, CSR edges, int-coded schedule, no per-step allocation | measured: as fast as rshgf |
| `vm:unsafe:no-bounds-checks` on the hot path, with invariants validated in `build()` | 35–45% on the standard/eHGF updates; indices guaranteed by construction |
| Parity tolerance 1e-9 rel against JAX f64; widened to the measured JAX↔rshgf gap where those disagree, with a strict 1e-9 check against rshgf instead | the port can't be more exact than pyhgf agrees with itself |
| **Wasm is the preferred web target** | the direction the platform is heading; JS stays tested |
| MIT licence | as in liblsl.dart (pyhgf's LICENSE says MIT, its pyproject says GPL-3.0: worth settling upstream, [P9](upstream/README.md#p9)) |

## Measured runtime facts that shape the design

From `benchmark/bin/runtime_probe.dart` (re-run it on new Dart releases):

- **`dart:math` exp/log are slower on Wasm**: they are JS imports. Per call,
  exp costs 11 ns on Wasm vs 6 ns on AOT, and 25 vs 10 ns when each call
  waits on the previous one. The table-driven `tableExp` (math.dart, 1 ulp)
  costs 8 ns (17 chained) on both, so `fastExp` uses it on dart2wasm only.
  (An earlier, coarser probe had found no difference.)
- **`Float32x4`/`Float64x2` are emulated in dart2wasm**, 10–15× slower than
  scalar. On AOT, `Float32x4` mul-add is 2.5× faster than scalar and
  `Float64x2` 1.3×. Any SIMD work is AOT-only until dart2wasm lowers these to
  Wasm SIMD. Re-check each release. (2026-11-10: re-measured on 3.13.7, a
  Float64x2 GEMM is still ~10× slower than scalar on Wasm. The SDK's SIMD
  tracking issue, dart-lang/sdk#64170, lists `v128` backing for
  `Float32x4`/`Float64x2` and their lists in dart2wasm as landed on main, so
  a 3.14+ release may change this; `useSimd` in hgf_deep's `kernels.dart` is
  the switch to flip.)
- **`-O4` helps Wasm** by about 10% on the engine, and 15–40% on plain
  typed-data loops.
- The cost is set by transcendental calls. Per 4-node trial on AOT, the
  standard/eHGF updates take about 200 ns and the unbounded update about
  1 µs.

## Next phases

Suggested order: 4, then W (performance, Wasm first), then 5 / 6 / 7. W1–W2
are cheap and also benefit everything after them.

**Decision (2026-11-05): Phase W comes next, starting with W1.** Phase 4
is done except the vector-valued node kinds (ef-state, categorical,
dp-state), which come back after W. The user considers them important, so
the constraint for W is: **nothing in W may make these node kinds need a
major rewrite.** Concretely:

- Keep the scalar struct-of-arrays columns and the int-coded schedule
  open to more step codes and node kinds (the `_kind`/`UpdateStep` dispatch
  must stay extensible; no hard-coding of "continuous or binary").
- Leave room for a vector-attribute side store: per vector field one
  `Float64List` plus per-node offsets (CSR-like), read by kind-specific
  steps outside the scalar hot loop.
- W4's `BatchNetwork` (`[field][node][batch]`) must be able to grow the
  same side store per batch element (`[vectorField][offset][batch]`), or
  fall back to the scalar engine for networks containing such nodes.
- The Dirichlet process node changes which branches are active at run
  time; don't bake in the assumption that the schedule never changes
  after `build()` (e.g. keep the per-step dispatch, or allow a schedule
  swap between trials).

### Phase 4: complete the nodalised engine

- [x] `surprise()` per node and total (Gaussian, binary), plus the response
      functions from `pyhgf/response.py`. Fitting needs these first.
      Done 2026-11-05: `ResponseFunction` (pyhgf's five plus
      `GaussianResponse` and closures), `Network.surprise`,
      `Trajectories.nodeSurprise`/`totalSurprise`; surprise references in
      every fixture. Overhead per trial on AOT: ~10 ns (one-node models),
      ~50 ns (`TotalGaussianSurprise`, 4 logs). On Wasm `TotalGaussian`
      costs ~150 ns, mostly `Network.get` (see W1).
- [ ] Node kinds:
  - [x] constant-state (bias). Done 2026-11-05; 5 fixtures. Parameters are
        rejected (pyhgf ignores them), volatility edges too (JAX raises;
        rshgf would allow volatility children).
  - [x] exponential-family (`ef-state`). Done 2026-11-05, following JAX
        (rshgf has only the univariate fixed-rate update and ignores
        `observed`). `Network.addEfState(dimension, learning, nus, xis)`;
        `hgf-1/2/3` adds pyhgf's implied chains in pyhgf's node order
        (reverse node order for `nNodes > 1`). 11 fixtures: fixed d=1 (also
        vs rshgf), fixed with missing data, fixed d=3, hgf-1/2/3, hgf-2 with
        missing data (ν = −∞ there, as in pyhgf) and with eHGF, d=2 hgf-2,
        two nodes, and mixed with a continuous HGF.
    - **Vector side store** as planned: per `VectorField` (`mean`, `xis`,
      `observation_ss`) one `Float64List` plus CSR offsets per node (`xis`
      and `observation_ss` share them); read only by the three ef step
      codes (10–12), outside the scalar hot loop. Saved by `saveState`,
      reset from per-field initial arrays, recorded by `Trajectories`
      (node-major blocks of `nTime × len`). `NodeField.nus` is a scalar
      column.
    - **Inputs can be wider than one column:** `nInputColumns`, per-input
      column offset/width. `observed` became one flag per input node, as in
      pyhgf (same as before for scalar inputs). The scalar input loop is
      unchanged; networks with ef inputs take a separate one.
    - **pyhgf quirk, replicated:** `get_update_sequence` moves the ef
      posteriors to the end with `remove` + `append` while iterating the
      same list, which skips the element after each move; with two ef nodes
      the order comes out reversed. `_moveDeferredSteps` replays it
      ([P3](upstream/README.md#p3); categorical nodes go through the same
      loop).
    - The parity suite now also checks the update steps against pyhgf's JAX
      schedule (`UpdateStep.pyhgfName`) on every fixture. Predictions are
      compared as a set: JAX schedules them a pass at a time, rshgf (and
      hgf.dart) one node at a time, so independent nodes differ in order
      (already the case on `multi_structure_*`; no effect on values).
    - Not yet: `BatchNetwork` (rejects the step codes, as planned), `sample`
      (throws), multivariate `TotalGaussianSurprise` terms (none in pyhgf
      either). A spec ef node can't declare edges; pyhgf allows
      `value_children` (for dp-state parents), to add with dp-state.
    - Cost (`profile_modes.dart ef`, ns/trial, AOT | Wasm -O4): fixed d=1
      50 | 41, fixed d=3 76 | 65, hgf-2 d=1 802 | 1265. pyhgf JAX f64 on the
      same trials (recording, as JAX always does): 296, 434, 1937.
    - Regression check (A/B against a copy with the hot-path changes
      reverted, interleaved runs): AOT within noise on every mode. Wasm: the
      filter modes within noise; recording 3–7% slower, not fully pinned
      down (the vector-recording hook explains part; it now sits in `_run`
      as a never-inlined call behind a loop-invariant flag).
  - [x] categorical-state. Done 2026-11-06, following JAX (rshgf has no
        nodalised categorical node). `Network.addCategoricalState(nCategories,
        ...)` adds pyhgf's `2n + 2` nodes in pyhgf's order: the categorical
        input, n binary nodes (its value parents), their n continuous value
        parents, one shared volatility parent (pyhgf's docstring says
        `3n + 2`). 7 fixtures: n = 3 under unbounded / eHGF / standard /
        mean-field updates, n = 4 (pyhgf's default), missing observations,
        and a categorical node followed by a continuous HGF.
    - **Vector store:** `VectorField.alpha` (own offsets); the categorical
      node's `VectorField.mean` holds the one-hot observation until its
      posterior replaces it with σ(μ) of each value parent, as pyhgf does.
      Scalar `NodeField.klDivergence` and `NodeField.surprise` (not in the
      default recorded fields: record them explicitly).
    - **Steps:** `predictionErrorCategorical` (indicators and `observed` to
      the binary nodes) and `posteriorCategorical` (α from ν = δ/Δ − 1, KL
      divergence, summed binary surprise), deferred to the end through the
      same replayed move loop as ef ([P3](upstream/README.md#p3)).
    - **Math:** `logGamma` (Lanczos, g = 7), `gammaSign`, `digamma`
      (recurrence + asymptotic series, reflection below 0),
      `dirichletKullbackLeibler` in log space: equal to pyhgf's where
      pyhgf's is finite, finite where pyhgf's Γ ratio overflows to NaN
      ([P13](upstream/README.md#p13)).
    - **Differences from pyhgf, all upstream bugs:** a categorical node may
      follow a continuous one (pyhgf crashes: [P7](upstream/README.md#p7),
      now reproduced); one node per `addCategoricalState` call, any number
      of them (pyhgf's `n_nodes > 1` is broken,
      [P10](upstream/README.md#p10)); the implied nodes take parameters
      (pyhgf ignores `binary_parameters`, [P11](upstream/README.md#p11)).
      Same as pyhgf: α = ±∞ on a missing observation
      ([P12](upstream/README.md#p12)). `categorical_test.dart` checks that a
      continuous HGF before the node, and a second categorical node after
      it, leave its trajectories bit-identical.
    - Not yet: `BatchNetwork` (rejects the step codes), `sample` (throws).
    - Cost (`profile_modes.dart categorical`, ns/trial, AOT | Wasm -O4):
      n = 3 1505 | 2360, n = 8 4806 | 7952. pyhgf JAX f64 (recording):
      6192 (n = 3), 20195 (n = 8); its jitted scan 5593 and 19377. The 2n + 2
      lgamma and n + 1 digamma calls of the KL divergence are a sizeable
      part; skip them when it isn't recorded, if that ever matters.
    - **Regression check (A/B, pinned core):** AOT filter modes within
      noise; AOT recording ~7% slower (278 → 296 ns/trial) with *fewer*
      instructions and ~20% more branch misses, i.e. code layout, not work.
      Moving the vector-kind step codes (10–14) out of the inlined
      `_dispatch` into a never-inline `_dispatchVectorStep` recovered
      ~4 ns (→ 292). Wasm -O4: every mode within noise of the baseline.
  - [x] Dirichlet process (`dp-state`). Done 2026-11-06, following JAX
        (rshgf has none). `Network.addDpState(batchSize, alpha,
        sensoryPrecision)`, then the clusters with
        `addEfState(nNodes: batchSize, valueChildren: [dp])`, as in pyhgf
        (`NetworkSpec`: ef-state nodes take `value_children`). 6 fixtures:
        3 clusters, pyhgf's defaults (10 clusters, α = 1), more openings
        than clusters, sensory precision 3, missing observations (patched
        JAX, [P14](upstream/README.md#p14)), and dp followed by a continuous
        HGF.
    - **Cluster proposals:** pyhgf's `get_candidate` weighs 20000 proposals
      drawn from `random.key(42)` on every trial (the same draws each time).
      `lib/src/jax_random.dart` ports JAX's threefry-2x32 (partitionable
      mode, the default since JAX 0.5), `split`, and float64 `normal` with
      XLA's `erf_inv` (Giles' polynomials, coefficients read off the
      compiled HLO). The draws match JAX to ≤ 4 ulp, not bit for bit: XLA's
      CPU backend fuses the Horner steps into FMAs and has its own `log1p`;
      Dart has neither. Drawn once per process (`DpProposals.instance`).
      pyhgf's own tests pin the legacy PRNG mode and expect float32 values
      ([P15](upstream/README.md#p15)).
    - **Store:** scalar `NodeField.alpha`, `sensoryPrecision`, `nTotal`,
      `nActiveCluster`; `VectorField.n`, `expectedMeans`, `expectedSigmas`,
      `activated`, one per cluster, sharing `_dpStart`. Clusters reuse the
      ef-state store; the dp step writes their mean, `observed` and, for a
      new cluster, `xis`. Openings past `batchSize` only count (pyhgf's
      out-of-bounds writes are dropped).
    - **Steps:** `predictionDp` (cluster means and sigmas from the parents'
      `xis`) and `predictionErrorDp` (assignment), in
      `_dispatchVectorStep`. As in pyhgf, an ef-state node without parents
      takes its step on the first update pass even when it has children,
      so the clusters run right after their dp node.
    - **Differences from pyhgf:** missing observations are skipped (pyhgf
      clusters them, and a NaN opens a NaN cluster: P14); a dp node may
      follow a continuous one (pyhgf crashes: [P7](upstream/README.md#p7));
      a cluster count other than `batchSize` fails at `build()` (pyhgf:
      broadcasting error in the scan, [P16](upstream/README.md#p16)).
      Clusters must be univariate with a fixed learning rate. `dp_state_test`
      checks that a continuous HGF before the node, and a second dp node
      after it, leave its trajectories bit-identical.
    - **The W constraint held:** pyhgf switches branches on and off through
      the clusters' `observed` flags, not by changing the schedule, so the
      int-coded schedule and per-step dispatch needed no change.
    - Not yet: `BatchNetwork` (rejects the step codes), `sample` (throws).
    - **Cost** (`profile_modes.dart dp`, µs/trial, 300 trials):

      | | 3 clusters | 10 clusters |
      |---|---|---|
      | pyhgf JAX f64 (`input_data`) | 3820 | 4447 |
      | hgf.dart AOT | 1281 | 1466 |
      | hgf.dart Wasm -O4 | 1827 | 2365 |

      A straight port took 2625 / 6128 µs on AOT. What brought it down:
      each cluster's two rows of 20000 densities are cached, keyed on its
      (mean, sigma), so only the cluster that learned is recomputed (same
      sums in the same order, so the same result); the four min-max passes
      became one (`max(v − min) = max − min` under monotonic rounding);
      the top-1000 quickselect runs on the ~1600 weights above a cutoff from
      a stride sample, exact because everything at or above the 1000th
      largest passes it (375 → 125 µs); `/ −2` became `· −0.5`. What is left
      is about 3 exps and 4 divisions per proposal per trial. 320 KB of
      cache per cluster.
    - **Regression check (A/B against HEAD, pinned core):** AOT recording
      ~6% faster, unbounded and categorical within noise; standard, eHGF,
      2-node and ef d=3 3–6% slower but with 3–5% *fewer* instructions and
      fewer branch misses (code layout again, as with categorical). Wasm
      -O4: every mode within noise.

  These need vector-valued attributes (`xis`, `alpha`, cluster means, ...):
  one side store per vector field (offsets per node, as CSR), recorded in
  `Trajectories`; these nodes are few per network, so they live outside the
  scalar hot path.
- [x] Custom coupling functions: `Coupling` interface, `CustomCoupling`
      (g, g', g'' plus a finite-difference `check()`), named in
      `NetworkSpec` via `build(couplings:)`. The named enum keeps its fast
      path (no measurable change). Fixtures: sin and softplus, against JAX
      autodiff.
- [x] Generative mode: `Network.sample` (Bernoulli / Gaussian draws from
      the predictive distribution, seeded `math.Random`), and
      `saveState`/`restoreState` for several futures from one state.
      Checked by replaying the draws through `inputData` (identical
      trajectories) and by moment tests; random streams can't match JAX.
- [ ] Per-child coupling functions, as JAX allows, if anyone needs them.
- [ ] One fixture per new feature in `tool/parity/gen_fixtures.py`, and each
      new node type added to the benchmark suite.
- Accept when parity holds on VM, dart2js and dart2wasm, and there is no
  ns/trial regression on `benchmark/bin/profile_modes.dart`.

### Phase W: performance, Wasm first

- **W1 done, W2 done, W3 done for the unbounded update (2026-11-05).**
  ns/trial from `profile_modes.dart`, AOT | Wasm -O4:
  standard 185 | 308; eHGF 225 | 365; unbounded 645 | 1030;
  2-node floor 79 | 130; recording, reusing a `Trajectories`, 260 | 505.
  Mean-field is within ±10% of relaxed on both. Parity is unchanged: the
  max rel err vs JAX per fixture is the same as before, 0–4e-7.
  `melos run bench`, 3-level 10k series: AOT 13.8 → 9.20 ms, Wasm
  24.2 → 13.6 ms; batch loop on Wasm 1109 → 549 ms.
  - **Benchmark artifact (fixed):** on the benchmark's 3-level spec, relaxed
    eHGF goes NaN after 48 trials and relaxed standard after 2229, in pyhgf
    as well. The earlier standard/eHGF numbers and the "mean-field is 20%
    slower on Wasm" puzzle were measuring NaN propagation, which Wasm runs
    faster. `profile_modes` now uses ω2 = ω3 = −6 on 3000 trials, where all
    modes stay finite, and warns if a mode goes non-finite. The gate
    benchmarks use the unbounded update, which stays finite.
  - **Fixed:** CSR strength columns were assigned through a record in
    `_compile`. dart2wasm's TFA then lost the exact type, so every `_vpK[e]`
    read was a `call_indirect` returning a *boxed double*.
    Rule: assign hot-path columns straight from a typed-data constructor.
  - **Fixed:** `vm:prefer-inline` doesn't reach dart2wasm, so
    `wasm:prefer-inline` sits beside it.
  - **Fixed:** `fastExp` uses `tableExp` on dart2wasm (see the runtime
    facts). Write float coefficients as double literals: dart2wasm doesn't
    fold `1 / 6` (BoxedInt.toDouble calls).
  - **W3 (unbounded update):**
    - S/(S+P), 1/(S+P) and log(S+P) in linear space, with a log-space
      fallback above logS = 700 (fixture `continuous_2level_unbounded_overflow`;
      mutation-checked on expansion 1 and the x* expansions);
    - `logT = 0` at t = 1;
    - wArg without logs;
    - no exp(0) in the softmax;
    - `lambertW0` stops when the step is ≤ 1e-12·|w| (≈4 instead of 6
      iterations, 1 ulp from pyhgf).
  - **Recording:** the recorder stages 64 trials time-major and then writes
    each node-major series a block at a time, instead of one strided store
    per value. On Wasm the stage copy is a `setRange`. Also
    `inputData(into: traj)` reuses a `Trajectories`, so there are no
    first-touch page faults (playground re-runs). On the 10k series, AOT
    recording went from ~430 to ~200 ns/trial over the filter, and to ~70
    with `into:`.
  - **Inherent to dart2wasm:**
    - Wasm GC `array.get` is always bounds-checked;
    - once any view exists in the program, reads of Float64Lists held in
      fields load `_offsetInElements` (lists passed straight as arguments
      keep their exact type). Removing all views gains 8–10%, which isn't
      achievable in real apps ([D1](upstream/README.md#d1));
    - i64→i32 index wraps;
    - top-level `final` tables (`_pow2`, `_exp2Frac`) are null-checked at
      every use, with the initialiser inlined there, ~1.5% of the eHGF
      profile ([D3](upstream/README.md#d3)).
  - **Upstream (pyhgf):** `lambert_w0` returns NaN for z ≳ 2.7e151 because
    `f1²` overflows; the unbounded update then drops that expansion
    ([P2](upstream/README.md#p2)).
  - **Upstream (pyhgf), relaxed eHGF diverges.** The eHGF increment
    re-predicts the child's volatility as `t·exp(κμ_b + ω̃)`. The relaxed
    prediction includes the MGF terms `Σ κ²/(2π̂_b)`, which the re-prediction
    leaves out. So "at μ_b = μ̂_b the re-predicted precision is the child's
    marginal predicted precision" (the docstring) holds only under
    mean-field. As a result:
    - in relaxed mode the increment is floored at 0 on ~95% of trials, so
      volatility parents never gain precision;
    - the 3-level continuous HGF then goes NaN on every ω setting tried
      (trial 44 on USD/CHF, `continuous_3level_eHGF` fixture), in JAX and
      in Dart alike.

    **Implemented as the opt-in `Network.ehgfMgfCorrection`** (off by
    default, so the default engine keeps pyhgf parity). It adds
    `Σ_p κ_p²/(2π̂_p)` over all of the child's volatility parents to ω̃,
    in relaxed mode only. The fixtures below come from pyhgf JAX with the
    same patch applied to `precision_update_ehgf` (mean-field left alone),
    and Dart matches them to 1e-9:
    - `continuous_3level_eHGF_mgf_correction`;
    - `..._two_volatility_parents_...`, with a second parent at κ = 0.5;
    - `continuous_3level_mean_field_eHGF_mgf_correction`, where the flag is
      a no-op.

    Demo: `example/ehgf_mgf_correction.dart`. The stock update goes NaN at
    trial 44; corrected, it stays finite and x2 gains precision on 586/614
    trials instead of 28. The patch in `gen_fixtures.py` is the shape of the
    upstream PR; rshgf needs the same change. If upstream adopts it, flip
    the default and regenerate the fixtures.
  - Tools: `profile_modes.dart` is platform-neutral (takes a mode arg),
    and `tool/web/run_wasm.mjs` passes args. Profile with
    `--no-strip-wasm`, and disassemble with the SDK's
    `bin/cache/dart-sdk/bin/utils/wasm-opt --all-features --print`.
  - Left, optional: an AOT breakdown of the unbounded update with `perf`.
- [x] **W1: find out why Wasm is 2× slower.** Answer: there was no single
      cause. Exp/log are JS imports, a lost exact type caused boxed doubles,
      and inlining pragmas were missing; all are fixed above. What remains
      (~1.6×) is mostly inherent to dart2wasm's typed-data access.
      The suspects were checked as follows:
  - typed-data views: inherent, see above;
  - the `switch` dispatch: a `br_table`, cheap;
  - the record destructuring in the trajectory copy: replaced by the
    staged recorder.
- [x] **W2: ship `-O4` for Wasm release builds.** `melos run
      test:wasm-release` compiles `packages/hgf/tool/parity_check.dart` (the
      parity suite as an app; `fixtureMismatches` in
      `test/src/parity_check.dart` is shared with `web_parity_test.dart`)
      with `-O4` and runs it under Node: 34/34 match. The playground:
      `flutter build web --wasm --release` defaults to **`-O2`** (checked on
      Flutter 3.47.9 with `-v`), so `build:playground`, `bench:playground`
      and the Pages workflow pass `-O4` explicitly (filter 6–8% faster,
      see phase 5).
- [x] **W3: cut transcendentals in the unbounded update.** Done (above). The
      standard/eHGF updates have one exp per node and nothing to cut.
- [x] **W4: batched lock-step engine** (`BatchNetwork`), scalar version done
      2026-11-05. B copies of a network with the same graph, stored
      `[field][node][element]`. Per element: initial values, coupling
      strengths (κ/α can be fitted), and optionally the observations
      (`perElement`). Also `inputData`, `surprise` (per-element totals and
      per-trial values, via `ResponseFunction.batchSurprise`) and `step`.
      No recording. Each element matches `Network` bit for bit; checked on
      all fixtures with perturbed parameters, per-element data and missing
      observations, on VM, dart2js, dart2wasm and `-O4`
      (`test/src/batch_check.dart`).
  - **What worked: stage-wise kernels.** The first version looped
    element-outermost inside each step (the scalar step per element, one
    dispatch per block). It gained only 7–20% on AOT and 5–15% on Wasm, so
    the per-step dispatch was never the cost. Each element's update is a
    chain of dependent exps and divisions, about 1,000 instructions long,
    too long for the CPU to overlap two elements. The kernels now run each
    stage (an edge's contribution, an exp, one Halley iteration of a
    block-wide `lambertW0`) across the block before the next stage starts,
    with per-element partial results in block scratch. Independent work
    then pipelines: `lambertW0` alone is 1.6× faster on AOT and 1.5× on
    Wasm. ns per element-trial (`benchmark/bin/batch_blocks.dart`,
    Network loop → BatchNetwork, block 256):

    | | AOT | Wasm -O4 |
    |---|--:|--:|
    | 2-level standard | 151 → 78 | 214 → 146 |
    | 2-level eHGF | 201 → 112 | 249 → 177 |
    | 2-level unbounded | 376 → 259 | 575 → 475 |
    | 3-level standard | 184 → 111 | 305 → 204 |
    | 3-level eHGF | 223 → 149 | 359 → 280 |
    | 3-level unbounded | 655 → 476 | 1029 → 883 |

    Block size: larger blocks help the unbounded update up to about 256,
    while eHGF is flat from 64 on. The default is 256.
  - Profile (AOT, unbounded, 2-level): the batched unbounded kernel 35%,
    libm exp/log 28%, `lambertW0` 13%, prediction 12%. What remains is
    mostly divisions and the libm calls (each spills the live XMM
    registers).
  - The batch scenario: 1,000 ω × 1,000 trials, 2-level unbounded
    (`melos run bench`, `docs/benchmark_results.md`):

    | | loop of `Network` | `BatchNetwork` |
    |---|--:|--:|
    | AOT, 1 isolate | 360 ms | 263 ms |
    | AOT, 8 isolates | 88.2 ms | **62.2 ms** |
    | JIT, 8 isolates | 128 ms | 92.7 ms |
    | Wasm -O4 | 554 ms | 443 ms |
    | JS | 785 ms | 500 ms |

    For comparison, JAX `vmap` takes 133 ms in f64 and 37.6 ms in f32. So the
    batch on 8 isolates is 2.1× faster than JAX f64 `vmap`, and JAX f32
    is still 1.65× faster (it was 2.3×). On one isolate, Dart is 2× slower
    than JAX f64 `vmap`: XLA vectorises exp/log over the batch.
  - **Vector-node constraint, kept:**
    - the batch shares the network's graph arrays and int-coded schedule;
    - step codes above `_kLastBatchedStep` make the constructor throw
      `UnsupportedError`, so networks with new node kinds fall back to
      `Network` until those kinds get batched kernels;
    - dispatch is still per step, so a schedule swap (dp-state) stays
      possible;
    - a vector side store would be `[offset][element]`, next to the
      columns.
  - The scalar engine's unbounded mixture moved into `_Unbounded.mix` (no
    measurable change on AOT or Wasm).
  - [ ] Open: an AOT-only `Float64x2`/`Float32x4` variant. It needs a SIMD
    exp/log without bit casts (dart2wasm emulates SIMD, so AOT only).
    `Float64x2` measured 1.3× on mul-add, and results would no longer be
    bit-identical to `Network`. Only worth it if fitting throughput on
    desktop is the bottleneck; phase 7's GPU kernel targets the same
    layout.
  - [ ] Optional, both engines: cache `exp(ω)` for nodes without volatility
    parents (constant over trials at Δt = 1). That saves 2 of about 16 exps
    per trial in the 2-level unbounded network.
- [ ] **W5 (deferred): float32 mode.** Only pays off with SIMD, so AOT only
      for now. Semantics change: the 1e-128 variance floor is below float32's
      range (it flushes to 0, as in JAX f32). It needs its own fixtures from
      JAX's default float32 mode, with a looser tolerance.
- [ ] Optional upstream: suggest the rshgf hot-path fixes to pyhgf
      ([R2](upstream/README.md#r2)).

Reference: `vector_math`/`vector_math_64` are fixed-size (2–4) graphics
types without exp/log, so they don't help the nodalised engine or arbitrary
layer widths. They're a useful idiom reference for typed-data/SIMD code
(W4, phase 7).

### Phase 5: Flutter playground (teaching / real time)

First version done 2026-11-06.

- [x] `packages/hgf_widgets`:
  - `TrajectoryPlot`: a `CustomPainter` (as in liblsl.dart's
    `signal_viewer`), no chart library. Each layer (line, points, band
    `μ ± scale/√π`, reference line) is reduced to at most two points per
    pixel column (`SeriesReducer`, min/max in order, so spikes survive),
    then drawn with `drawRawPoints` / triangle-strip `Vertices`. Linked
    hover cursor with a readout; `limit` for playback; `revision` for data
    overwritten in place.
  - `HgfTrajectoryView`: one panel per node, top level first, plus the
    total surprise (pyhgf's `plot_trajectories`). Colours by role (belief,
    observation, surprise), CVD-checked, stepped for light and dark.
  - `NetworkGraph` + `NetworkLayout`: inputs at the bottom, value edges
    solid, volatility dashed, shapes by kind, tap to select, captions.
  - Needed `Network.valueParentsOf` & co. (read-only edges, strengths,
    `couplingOf`) in `packages/hgf`.
- [x] `apps/hgf_playground`: presets (binary 3-level, continuous 2- and
      3-level) on simulated data (reversal learning; a random walk with
      calm/volatile phases) with the hidden truth overlaid; CSV loading
      (NaN = missing); sliders for ω, μ₀, π₀, ρ, λ, π_u and the coupling
      strengths; the volatility update scheme; trial-by-trial playback with
      `predict`/`step` (tested to record exactly what `inputData` does);
      runaway/NaN warnings; light and dark themes following the system by
      default, overridable and saved (`easy_shared_preferences`); wide and
      phone layouts. Re-filters into the same buffers (`inputData(into:)`),
      records only the 3 plotted fields, and coalesces slider events to one
      re-filter per frame.
- [x] Frame time with the engine running each frame:
      `melos run bench:playground` (`?bench=N`, or
      `--dart-define=HGF_BENCH=N` on desktop: ω₂ dragged for 300 frames).
      Wasm release, headless Chromium (software raster), binary 3-level,
      at `-O4` (Flutter's default `-O2` in brackets):

      | trials | filter (median) | frame interval (median / p95) |
      |---|---|---|
      | 300 | 0.35 (0.38) ms | 16.6 / 20.0 ms (60 fps) |
      | 1 000 | 1.12 (1.22) ms | 16.6 / 19.6 ms (60 fps) |
      | 10 000 | 10.6 (11.3) ms | 23.0 / 27.8 ms (~43 fps) |

      Recording 3 fields instead of 7 took the 10k filter from 13.7 to
      11.3 ms. On the web, `FrameTiming` reports only ~20 of 300 frames
      with implausible durations, so the app shows the frame interval there.
- [x] Views for the vector node kinds (2026-11-06). `HgfTrajectoryView`
      plans its panels from the graph and folds implied nodes into the
      panel they serve:
  - **categorical:** each category's predicted probability (its binary
    parents' `μ̂`) with the observed category as a row of dots per
    category above it; the binary parents get no panels, their value
    parents (the logits) share one;
  - **dp-state:** observations coloured by the cluster they joined (where
    `VectorField.n` grew), and each open cluster's predicted mean ± sd
    (`expectedMeans`/`expectedSigmas`, from the trial after it opened);
    the clusters get no panels;
  - **ef-state:** observations and, before each, the mean ± sd implied by
    the previous trial's `ξ` (one line per dimension for d > 1);
  - categories and clusters take the 8-slot categorical palette in fixed
    order (`HgfColors.categories`, light and dark steps), past 8 a grey
    overflow; no surprise panel when no node defines one (dp/ef only);
    `BandLayer.bounds` reads out its half-width (it read the lower bound);
    `PlotLayer.readoutGaps: false` keeps sparse layers out of the readout.
- [x] Playground presets for them: *Categorical HGF (3 outcomes)*
      (simulated: one likely outcome that switches), *Dirichlet process*
      (4 cluster slots, α = 0.1, ν = 3; simulated: Gaussian sources visited
      in turn, which it recovers as 3 clusters of 50 on the default 150
      trials), *Exponential family* (ν = 10 on the random walk, to contrast
      with the HGF's adaptive learning rate). Sliders for α and π_s (dp)
      and ν (ef); edits now live in a per-node map on top of the preset,
      so implied nodes are editable too. Playback (`predict`/`step`)
      records the vector fields as well, tested equal to `inputData`.
      CSV loading takes category indices for categorical inputs.
      `?preset=<id>&trials=N` links to a preset on the web.
  - dp-state is slow for live re-filtering: ~375 ms for 150 trials on Wasm
    (pyhgf's 20 000 proposals per trial), so the preset caps trials at 1000
    and starts at 150. A web worker (below) would keep the UI responsive.
  - The categorical preset's shared volatility parent uses ω = −8: at
    pyhgf's −4 the unbounded update runs away by trial ~300, at −6 by
    ~2400 (the P17 structure: binary HGFs under a volatility parent).
- [x] Fitting view (2026-11-07): see Phase 6.
- [ ] Next: zoom and pan on the time axis;
      editable structure (add/remove nodes); export (CSV of trajectories,
      PNG); a guided "lessons" mode for teaching; measure desktop (GPU)
      frame times (`flutter run --profile --dart-define=HGF_BENCH=10000`);
      if 10k-trial drags must hold 60 fps on Wasm, filter in a web worker
      or the engine needs to get faster there.
- Found with it: [P17](upstream/README.md#p17), the default unbounded
  update runs away on binary 3-level HGFs (pyhgf too), and on the
  categorical node's implied network, which has the same structure.

### Phase 6: fitting probe (`packages/hgf_fit`)

First part done 2026-11-06: MAP estimation.

- [x] Objective: `HgfModel` (network + `ResponseModel` + free `Parameter`s)
      and the negative log posterior `−log p(responses | x) − Σ log p(xᵢ)`,
      evaluated for many (subject, point) pairs in one `BatchNetwork` pass
      (`negLogPosterior`), with an optional Jacobian term for samplers.
  - Parameters target node fields (`tonicVolatility`, initial `mean`,
    input `precision`, …), value/volatility coupling strengths, or response
    parameters; one parameter may set several targets.
  - Priors: `Normal`, `TruncatedNormal` (normalised with an accurate
    `log(Φ(b) − Φ(a))`), `Uniform`, `HalfNormal`, `LogNormal`, `GammaPrior`,
    `Flat`; transforms (identity, log, logit) follow the support.
  - Response models read their parameters per element (the built-ins in
    `package:hgf` hold one fixed value): `GaussianReadout` (the user's first
    priority; noise precision fittable), `BinarySoftmaxReadout` (β
    fittable), `ObservationSurprise` (perceptual fit without responses),
    `ResponseModel.fn` for custom ones. Each simulates responses too
    (`HgfModel.simulate`, via `BatchNetwork.step`).
  - Subjects may have their own series, missing observations, lengths and
    time steps; elements are grouped into one batch per (length, time
    steps).
- [x] Gradients: central finite differences in the same pass
      (`BatchedObjective`, `2k + 1` elements per gradient, one-sided where
      the stencil hits `+∞`). Checked against a Richardson estimate to 1e-5.
- [x] Optimisers: `Lbfgs` (Armijo backtracking with cubic interpolation,
      gradient at every trial point so an accepted step costs no extra
      pass) and `NelderMead` (adaptive coefficients; the reflection,
      expansion and both contractions requested together, so one pass per
      iteration). Both are `sync*` generators yielding evaluation requests:
      `fitMap` advances every subject's optimiser in lock step and answers
      all pending requests with one batched pass. `restarts:` adds starts
      drawn from the priors.
- [x] Parameter recovery on simulated data
      (`example/parameter_recovery.dart`, tests): the notebook's 20 agents
      give r = 0.95 for ω₂ and r = 0.92 for β (MAP; the notebook uses NUTS);
      L-BFGS and Nelder–Mead agree. A continuous 2-level HGF with a Gaussian
      readout recovers ω and the response precision, provided the inputs
      are noisy enough: with input precision 1e2 the learning rate stays
      near 1 whatever ω is, and ω isn't identifiable.
- [x] Compared with pyhgf + JAX (`melos run bench:fit`): single thread,
      AOT, binary 2-level, 320 trials.

      | agents | hgf_fit L-BFGS | JAX f64 grad + SciPy L-BFGS-B | hgf_fit Nelder–Mead |
      |---|---|---|---|
      | 20 | 29 ms, 15 passes | 94 ms, 51 calls (+0.8 s compile) | 113 ms |
      | 200 | 292 ms, 17 passes | 1566 ms, 126 calls | 1115 ms |

      JAX's objective at hgf_fit's estimates equals hgf_fit's to 3e-13;
      hgf_fit's optimum is equal or lower for every agent. Finite
      differences cost `2k + 1` filters per gradient, so the lead narrows as
      `k` grows; exact gradients (below) would remove that factor.
  - Found on the way: `sigmoidInverseTemperature`'s two non-integer `pow`s
    cost 46 ns per trial, more than the binary 2-level filter itself
    (43 ns). Now `σ(β · logit x)` (in `package:hgf`, agrees to 1e-13).
- [x] Laplace approximation (2026-11-06): `laplace(model, mapFit)`.
  - Hessian by central second differences of function values on the
    unconstrained scale, `2k² + 1` elements per subject, every subject in
    one pass (`BatchedObjective.hessians`; cheaper and `O(h²)`, unlike
    forward differences of gradient stencils at `(k+1)(2k+1)`).
  - Covariance `H_y⁻¹`, natural-scale standard errors by the delta method,
    intervals `y ± z·sd` mapped through the transforms (they respect the
    support), correlations, and the log evidence. The MAP is a mode on the
    natural scale, so the evidence is the natural-scale Laplace
    `−f + (k/2) log 2π − ½ log det H_x`, computed from `H_y = J H_x J`.
    A Hessian that isn't positive definite gives NaN, not garbage.
  - Checked against grid quadrature of the exact evidence (density on the
    unconstrained scale with the Jacobian): within 0.002–0.008 nats on
    well-identified binary 2-level fits, 0.08 on a weakly identified one;
    standard errors within 2% (13%). 95% intervals cover 92–98% of true
    values over 40 simulated agents (4 seeds). Model comparison: fixing ω
    at the truth beats fitting it by ~2.4 nats per subject (Occam), and a
    wrong fixed ω loses by ~80 nats per subject.
  - Cost: ~0.4 ms per subject for the Hessian pass at k = 2.
- [x] NUTS (2026-11-06): `sampleNuts(model, subjects)`.
  - Stan's algorithm, ported from `base_nuts.hpp`: multinomial sampling
    (biased progressive at the top level), the generalised no-U-turn
    criterion with the extra checks across merged subtrees, divergence at
    an energy error > 1000, dual averaging (δ = 0.8, γ = 0.05, t₀ = 10,
    κ = 0.75), the step-size doubling/halving heuristic after each metric
    update, and the 75 / 25·2ⁿ / 50 warmup windows with Stan's shrinkage.
    Unit, diagonal (default) or dense metric.
  - Each chain is a `sync*` generator yielding one leapfrog step (point +
    gradient) at a time, the recursive tree builder via `yield*`; the
    driver answers every running chain of every subject with one
    `BatchNetwork` pass. Draws of a subject are bit-identical whether it is
    sampled alone or with others (tested).
  - Forward differences by default then (`Gradients.forward` now; exact
    gradients replaced them as the default, see below; `k + 1`
    filters per step instead of `2k + 1`): leapfrog with any deterministic
    force is reversible and volume preserving, and the multinomial weights
    use the exact density, so the chain stays exact; only efficiency could
    suffer, and it doesn't measurably (same leapfrog steps per draw as
    central differences, 1.6× less time).
  - Diagnostics are ArviZ's estimators (rank-normalised split R̂, bulk and
    tail ESS, MCSE, plus E-BFMI), checked against arviz-stats 1.3 to 1e-9,
    with ties and odd chain lengths. Export: `NutsFit.toJson()` for ArviZ
    `from_dict` (a `subject` dimension), `PosteriorDraws.toCsv()` in Stan's
    column layout.
  - Checked: posterior means within 4 MCSE of grid quadrature and sds
    within a few %, second moments of Gaussians unbiased over long runs
    (z-scores ~ N(0, 1) for all three metrics), and the 20-agent posterior
    agrees with PyMC's and NumPyro's (rms z of the 40 means ≈ 1.1).
  - Found with it: on this input sequence the binary 2-level HGF's
    likelihood jumps by ~500 nats between ω₂ = 0.35 and 0.36 (a
    bifurcation of the μ₂ trajectory, not a NaN). With the notebook's prior
    (upper bound 2), weakly identified agents' trajectories reach it and
    NUTS reports divergences there; they carry no posterior mass, but a
    tighter upper bound avoids them.
  - Performance (`melos run bench:nuts`, one core, 20
    agents × 4 chains × (1000 + 1000)): hgf_fit 64 s; NumPyro (joint
    40-d model, exact gradients, jitted) 147 s + 7.5 s compile; PyMC (the
    notebook's `wrap_jax` route) 214 s. But per draw the joint samplers
    get 3× the bulk ESS (median 5806 / 5515 vs 1922: long joint
    trajectories, 15 leapfrog steps per draw vs 4.5 per agent, make draws
    antithetic), so in ESS/s (median) NumPyro 39.5 > hgf_fit 30.0 > PyMC
    25.8. Posterior means agree with both (rms z 0.94 / 1.10).
    Central differences took 101 s for the same draws per step count.
- [x] Exact gradients (2026-11-06): hand-written forward-mode tangents of
      every batched update step (`BatchNetwork(tangents: k)`,
      `batch/batch_tangents.dart`).
  - Fused with the primal: each `_t*` kernel computes exactly what its
    `_b*` kernel computes (bit-identical, tested) plus the derivatives,
    stage-wise: per-element partials into coefficient arrays, then a tight
    element loop per direction. Tangents are `[node][direction][element]`.
  - Derivatives of the function as computed: branches pass the side taken,
    active clamps pass nothing, Lambert W contributes `W/(z(1+W))`.
    Variances can reach 1e300, so tangents are formed as `π̂ (π̂ dV)` and
    `(dx − (x/V) dV)/V`, never through `1/V²` (which underflowed and gave
    wrong tangents on `constant_bias_eHGF` and the overflow fixture before).
    Non-linear value coupling needs g''': `Coupling.d3f`.
  - Response functions report partial derivatives, not tangents
    (`batchSurprisePartials`, `SurprisePartials.add(node, field, ∂s/∂x)`);
    the engine contracts them. Calling `getTangent` per direction made
    `TotalGaussianSurprise` cost more than the filter.
  - Directions no seed moves are skipped (`_scanSeeds` at reset), as are
    zero coupling / autoconnection seeds: a response parameter (β) costs
    nothing in the filter.
  - Checked: every tangent field of every node and the total surprise
    against Richardson central differences on the 37 batchable fixtures, 6
    directions each (ω, drift, μ₀, π₀, α, κ), on the VM, dart2js and
    dart2wasm; the gradient of every built-in response; `hgf_fit`'s
    objective gradients (all priors and transforms, Jacobian on/off, three
    update modes with 5 parameters, missing data, time steps, response
    parameters, overwritten targets, `ResponseModel.fn`). Two fixtures
    need care: `continuous_2level_eHGF` turns chaotic after ~150 trials
    (finite differences disagree with each other at every step; checked on
    its first 120 trials), and the binary 3-level ones sit in P17's runaway
    regime (tangents ~1e8; differences converge onto them only below
    h = 1e-5).
  - Cost per element-trial, AOT (`benchmark/bin/tangent_cost.dart`): one
    direction ≈ the primal's arithmetic again (the exps are shared), each
    more about half to two-thirds of that. Two directions: 155 ns vs 195 ns
    for forward differences on the binary 2-level model, about level on
    continuous 3-level standard/eHGF, 764 vs 942 ns with unbounded updates.
    Wasm keeps the same ratios. Per direction this is about 330
    instructions on the binary model; a fused per-element direction loop
    and the element-innermost layout cost the same, so the arithmetic is
    the floor. Reverse mode would only pay for many parameters.
  - End to end: MAP 29 → 13.7 ms (20 agents), 292 → 120 ms (200 agents);
    NUTS 64 → 43.7 s, ESS/s 30.0 → 43.0 (median), 12.9 → 23.0 (min);
    NumPyro, same session: 145 s, 39.9 / 18.7.
  - Not done: the Laplace Hessian still uses second differences of values
    (`2k² + 1` plain elements); central differences of exact gradients
    would be more accurate but not cheaper for small k.
- [x] Hierarchical models (2026-11-06): `HierarchicalModel` over a
      subject-level `HgfModel` (`lib/src/hierarchical.dart`,
      `empirical_bayes.dart`).
  - Parameters are pooled (each subject's value, on the parameter's
    unconstrained scale `y = T(x)`, from a `Population` `N(μ, σ²)` with
    hyperpriors on μ and σ: normal, log-normal or logit-normal populations
    by the parameter's transform), shared (one value, its own prior) or
    independent (as before). pyhgf's multilevel notebook is two pooled
    parameters: ω₂ normal, β log-normal.
  - `sampleHierarchical`: NUTS on the joint posterior (`2·pooled + shared +
    n·local` dimensions), non-centred by default (`z = (y − μ)/σ`) or
    centred. Each leapfrog step of every chain is one batched pass of
    `chains × subjects` elements; the per-subject gradients (exact) are
    chained into μ, σ and z by hand. `includePrior` keeps the pooled and
    shared parameters' own priors out of the subject objective.
  - `fitEmpiricalBayes`: type-II ML of the populations by Laplace-EM
    (subject modes in lock step, warm-started; one Hessian pass; closed-form
    `μ = mean ŷ`, `σ² = mean((ŷ − μ)² + Σᵢᵢ)`), via `HgfModel.withPriors`
    (shares the compiled batches). Recovers the sample moments of 60
    simulated agents to ~0.03.
  - Checked: the joint objective against a hand-written density, gradients
    against finite differences (centred and not, pooled/shared/independent
    mixes), prior recovery with a flat likelihood, centred = non-centred
    posteriors, and agreement with NumPyro and PyMC (`melos run
    bench:hier`).
  - Performance on the notebook's problem (10 agents × 320 trials, 4 chains
    × (1000 + 1000), one core, centred):

    | | wall time | leapfrog steps after warmup | bulk ESS min / median | ESS/s min / median | means vs hgf_fit (rms z) |
    |---|---|---|---|---|---|
    | hgf_fit `sampleHierarchical` | 52.3 s | 46400 | 2685 / 4047 | 51.4 / 77.4 | |
    | NumPyro | 62.0 s (+8.1 s compile) | 43840 | 2591 / 3468 | 41.8 / 55.9 | 1.11 |
    | PyMC (notebook route) | 102.8 s | 37612 | 2854 / 3936 | 27.8 / 38.3 | 0.83 |
    | hgf_fit `fitEmpiricalBayes` | 45 ms (7 iterations) | | | | |

    The per-step lead over NumPyro is smaller than in the per-agent
    benchmark (~1.3× vs 3.3×): JAX `vmap`s the 10 agents of each step
    together, and a 40-element pass costs hgf_fit 170 ns per element-trial
    (163 at 320 elements; the hierarchical layer itself adds nothing).
    With 320 trials per agent the agents are well identified, and centred
    beats non-centred (139 s, 2.5× the leapfrog steps); non-centred is
    the default for the usual case of little data per subject. NumPyro
    non-centred (seed 0) did not converge: 404 s, R̂ 1.61, min ESS 7
    (one chain stuck, likely started from its uniform(−2, 2) init with
    σ_ω up to e² near the ω₂ ≈ 0.355 cliff); not investigated further.
  - Not done: empirical Bayes with shared parameters (needs a gradient
    M-step); hyperpriors in EB (MAP-II); group-level covariates
    (regressions on μ).
- [x] Isolate-parallel passes (2026-11-06): `FitWorkers`
      (`lib/src/workers_io.dart`, `pass.dart`).
  - The drivers became `Program`s: `sync*` generators yielding `Pass`es
    (one `negLogPosterior` call each) and `Concurrent` streams. `runInline`
    answers them on the calling isolate (the synchronous functions, as
    before); a `FitWorkers` pool answers them on worker isolates
    (`workers.fitMap`, `.laplace`, `.sampleNuts`, `.sampleHierarchical`,
    `.fitEmpiricalBayes`, `.evaluate`). One copy of each driver serves both.
    Batch elements are independent, so the results are bit-identical
    (tested on every driver, mixed lengths, time steps, shared series,
    `ResponseModel.fn` closures, errors raised in a worker).
  - Two kinds of parallelism: each pass splits into contiguous slices of
    about equal cost (`minSliceTrials`, default 2048), and drivers with
    independent tasks (per-subject optimisers and NUTS chains,
    hierarchical chains) deal them, strided, into one stream per worker,
    each in lock step. All slices go into one queue; idle workers take one
    at a time.
  - The main isolate only routes. Letting it filter slices too stalled
    every stream: the drivers resume there, and a stream resumed after a
    local slice found every worker busy (their replies were still unread
    behind the microtask) and ran entirely on the main isolate. Even with
    the steal moved to the event loop, a busy main isolate holds the
    workers' streams back, so the default is one worker per processor.
  - Persistent pool, spawned once (the VM limits running isolates per
    group; the user hit this in liblsl.dart). Models (by `withPriors`
    family, sharing compiled batches) and subjects (in one message, so
    shared series stay shared) are copied to each worker once and dropped
    via a `Finalizer` when collected on the main isolate.
  - Measured costs (AOT, i7-7700, 4 cores / 8 threads): a slice-sized
    message round trip takes ~5 µs against ~350 µs of filtering per 2048
    trials; the drivers' serial share is ~3%. Typed data is always copied
    (`asUnmodifiableView` too; 80 MB in 7 ms).
    `@pragma('vm:deeply-immutable')` objects are shared, not copied:
    100k of them send in 1.3 ms against 33 ms for plain lists. That
    doesn't pay here, because only small typed lists cross per slice, and
    `Prior`/`ResponseModel` are user-extensible, which the pragma doesn't
    allow. Small batches cost more per element (b = 5: 229 ns,
    b = 40: 167 ns per element-trial), so more streams than workers lose.
  - `melos run bench:parallel` (150 + 150 draws for the NUTS rows):

    | job | 1 isolate | 2 workers | 4 workers | 8 workers |
    |---|---|---|---|---|
    | MAP, 20 agents | 12.8 ms | 1.56× | 1.55× | 1.80× |
    | Laplace, 20 agents | 4.6 ms | 1.83× | 2.16× | 2.64× |
    | NUTS, 20 agents | 7.44 s | 1.87× | 3.07× | 2.90× |
    | MAP, 200 agents | 115 ms | 1.82× | 3.60× | 3.66× |
    | Laplace, 200 agents | 48.9 ms | 2.02× | 3.84× | 4.57× |
    | NUTS, 200 agents | 72.0 s | 1.97× | 3.53× | 3.97× |
    | hierarchical NUTS, 10 agents | 11.7 s | 1.86× | 2.72× | 2.80× |
    | empirical Bayes, 10 agents | 35.4 ms | 1.51× | 2.04× | 1.80× |

    Four physical cores bound it at about 4× (8 workers add the usual
    hyper-threading gain on the large jobs). Small problems are limited by
    lock-step barriers and small batches. Full-length runs (4 chains ×
    (1000 + 1000), as in the PyMC/NumPyro comparisons): per-agent NUTS on
    20 agents 42.0 s → 13.8 s on 4 workers (3.0×), 13.3 s on 8;
    hierarchical NUTS 57.7 s → 21.8 s (2.6×), 21.6 s on 8. (Those
    comparisons ran NumPyro and PyMC on one core; NumPyro can run chains
    in parallel too, `chain_method="parallel"` with several host devices,
    not measured.)
- [x] Web Workers (2026-11-06): `FitWorkers` on the web.
  - One `FitWorkers` class over a `WorkerTransport` (`workers.dart`):
    isolates where `dart.library.io` exists, Web Workers where
    `dart.library.js_interop` does, none otherwise; the worker side is one
    `WorkerHost` for both. Without a script (or in Node) the pool runs
    inline, as before.
  - Workers can't run the page's code, so the worker is a program of its
    own: `package:hgf_fit/fit_worker.dart`, `runFitWorker()`, compiled with
    `dart compile js` or `wasm`; `FitWorkers.spawn(script:
    FitWorkerScript.wasm('fit_worker.wasm'))`. A Wasm worker starts from a
    blob-URL module that imports the `.mjs` loader, so nothing else ships.
  - Models cross as JSON: `Network.toJson`/`fromJson` in `package:hgf` (an
    exact copy: nodes after expansion, edges with both sides' strengths,
    settings, every field's initial value; bit-identical filtering on all
    61 fixtures, also with parameters changed after build) and `FitCodec`
    in `hgf_fit` (`toJson()` on the built-in priors, transforms and
    response models, decoders for custom ones; custom couplings and
    response functions by name; `ResponseModel.fn` can't cross). Subjects
    and slices go as typed arrays, each distinct series once.
  - Checked (`melos run test:web-workers`, in CI): every driver from a
    dart2js and a dart2wasm page with JS and Wasm workers, bit-identical
    when the worker is compiled like the page (MAP within 1e-5 across
    compilers); missing scripts, unserialisable models, undecodable custom
    types and errors thrown in a worker all reach the caller.
  - Found: dart2wasm loses `Expando<int>` values when that is the program's
    only Expando ([D7](upstream/README.md#d7)); the pool's ids are boxed.
  - `melos run bench:web-parallel` (headless Chromium, i7-7700, page and
    worker compiled alike, 150 + 150 draws):

    | job | Wasm main thread | 2 | 4 | 8 Wasm workers | JS main thread | 2 | 4 | 8 JS workers |
    |---|---|---|---|---|---|---|---|---|
    | MAP, 20 agents | 22.9 ms | 1.37× | 2.33× | 2.14× | 30.6 ms | 1.29× | 0.85× | 0.69× |
    | Laplace, 20 agents | 31.7 ms | 1.82× | 2.75× | 2.43× | 42.0 ms | 1.56× | 1.45× | 0.90× |
    | MAP, 200 agents | 209.5 ms | 2.28× | 4.04× | 4.42× | 251.4 ms | 1.85× | 2.83× | 2.77× |
    | Laplace, 200 agents | 302.6 ms | 2.33× | 4.17× | 4.84× | 358.6 ms | 1.92× | 3.02× | 3.40× |
    | NUTS, 20 agents | 12.90 s | 2.20× | 3.54× | 3.67× | 16.70 s | 1.70× | 2.37× | 2.30× |
    | hierarchical NUTS, 10 agents | 77.21 s | 2.13× | 3.29× | 3.28× | 95.90 s | 1.69× | 2.13× | 2.18× |
    | empirical Bayes, 10 agents | 89.8 ms | 1.76× | 2.47× | 2.03× | 125.2 ms | 1.41× | 1.88× | 1.30× |

    One Wasm worker is 1.15× the Wasm main thread (messaging included; why
    the main thread filters slower wasn't investigated), one JS worker
    0.97× its main thread: hence Wasm's gains at or above the core count,
    and the JS build's weaker scaling (a message costs more than between
    isolates). The Wasm main thread is 1.3× the JS one. `dart test`
    compiles its pages without the release flags, so speed-ups measured
    there are inflated; use the standalone page.
- [x] Fitting view in the playground (2026-11-07): a *Fit parameters*
      card.
  - Targets: the observations (`ObservationSurprise`, so the summed
    surprise panel plus the prior), or responses: simulated from an agent
    with the current sliders and a response parameter (β for binary,
    response noise for continuous; `GaussianReadout` /
    `BinarySoftmaxReadout` on x₀), which become the truth to recover, or
    loaded from a CSV.
  - Free parameters as chips: every non-input ω (on by default), μ₀, κ,
    the input π_u, the response parameter. Priors are centred on the
    preset's values, not the sliders' (so not on a simulated truth), and
    shown on hover.
  - MAP (L-BFGS, 3 restarts) + Laplace, or NUTS (4 chains × 500 + 500,
    started at the mode) with a progress bar and cancel. Each parameter
    gets its estimate, 95 % interval, R̂/ESS (NUTS) and a small plot of
    prior vs posterior (Laplace density or NUTS histogram) with the
    estimate and the truth; *Apply* moves the sliders to the estimates.
    Simulated responses are drawn on the input panel.
  - Fits run on a `FitWorkers` pool of 4: Web Workers from
    `fit_worker.wasm`/`.js` (built into `web/` by
    `apps/hgf_playground/tool/build_worker.sh`, in `build:playground`
    and the Pages workflow; 166 KB Wasm, 236 KB JS), isolates on desktop.
    If the worker doesn't load, on the main thread with `yieldInterval`.
  - Vector node presets (categorical, dp, ef) say why they can't be fitted
    (no batched kernels yet).
  - For it, `hgf_fit` gained `FitWorkers.cancel()` / `FitCancelled`,
    `FitWorkers.yieldInterval` and `SamplingProgress`.
  - Found: a coalescing `Playground` swaps in new data at once but
    notifies at the next frame, so `Fitting` re-checks it at each entry
    point (regression test pumps that frame).
  - `melos run bench:playground-fit` (Wasm release, headless Chromium,
    i7-7700, 400 trials, simulated agent, same estimates in both modes):

    | fit | 4 Wasm workers | main thread | frame interval during the fit (workers / main) |
    |---|---|---|---|
    | binary 3-level, MAP + Laplace (3 params) | 52 ms | 146 ms | — |
    | binary 3-level, NUTS | 18.4 s | 49.0 s | median 16.7 / 17.7 ms |
    | continuous 3-level, MAP + Laplace | 81 ms | 96 ms | — |
    | continuous 3-level, NUTS | 11.1 s | 34.5 s | median 16.7 / 17.9 ms |

    Every true value fell inside its 95 % interval. The UI keeps 60 fps
    either way (max frame 80–120 ms, once).
- [ ] Next:
  - tangents for the vector node kinds, once they have batched kernels
    (then the categorical, dp and ef presets can be fitted);
  - NUTS in the playground takes 11–18 s: fewer iterations by default,
    or more chains than workers;
  - responses in the same CSV as the observations (a second column);
  - hierarchical fits in the playground (several simulated agents).

### Phase 7: GPU probe (go/no-go): **go** (2026-11-07)

`packages/hgf_gpu`. Commands: `melos run test:gpu`, `melos run
bench:gpu-web`, `melos run bench:gpu-native`, and
`tool/parity/gpu_f32_reference.py` for JAX f32 against f64.

**Design: codegen, not an interpreter.** `HgfKernel(network)` turns the
compiled network into WGSL, a little like XLA does for JAX:
- the update schedule and every edge are unrolled;
- each GPU thread is one batch element and keeps that element's whole
  state in registers while it loops over trials;
- the arithmetic is `BatchNetwork`'s, step for step, in f32.

Every per-element quantity sits in one `[slot][element]` storage buffer:
node fields, ω, drift, λ, coupling strengths, and the running response
total (a Kahan sum). Any of them can vary per element without
recompiling.

A dispatch loads only the slots that carry over between trials, which a
liveness pass at codegen time finds. It stores those slots plus the
beliefs a caller reads back. A series can therefore be split across
dispatches, which keeps each one under the browser's GPU watchdog.

`GpuBatch` keeps the initial state on the GPU and resets it with a
GPU-to-GPU copy on each call. Between calls, only the slots changed with
`setInitial` and friends cross the bus. One `GpuBackend` interface has two
implementations:
- `WebGpuBackend`: hand-written `dart:js_interop` bindings, because
  package:web 1.1's `webgpu.dart` only has the flag constants;
- `WgpuBackend`: ffigen bindings of `webgpu.h` and `wgpu.h`, wgpu-native
  v29.0.1.1, vendored under `third_party/webgpu`.

Supported now:
- continuous, binary and constant nodes;
- standard, eHGF (with the MGF option) and unbounded updates, relaxed or
  mean-field;
- every named coupling function;
- missing observations, per-trial time steps, shared or per-element data;
- `TotalGaussianSurprise` and `FirstLevelBinarySurprise`.

Not yet: ef-state, categorical, dp-state, custom couplings (Dart closures),
and tangents.

**Results (RX 7600 XT, RADV / Mesa 26.2.6, Chromium 156, i7-7700).**
These are end-to-end times, including the upload of new ω values, the
dispatch and the readback. The CPU comparisons are from
`docs/benchmark_results.md`: `BatchNetwork` on 8 AOT isolates 62 ms, JAX
f64 `vmap` 133 ms, JAX f32 `vmap` 37.6 ms.

| 1000 trials | B = 1 | B = 1000 | B = 64k | B = 10⁶ | ns / element-trial at 10⁶ |
|---|--:|--:|--:|--:|--:|
| 2-level unbounded, native | 1.8 ms | **2.2 ms** | 9.1 ms | 70 ms | 0.07 |
| 2-level unbounded, browser (Wasm) | 2.9–3.9 ms | **2.7–4.0 ms** | 11 ms | 100 ms | 0.10 |
| 3-level unbounded, native | 3.4 ms | 3.8 ms | 15 ms | 115 ms | 0.12 |
| 2-level standard, native | 0.9 ms | 1.0 ms | 3.6 ms | 42 ms | 0.04 |
| binary 3-level eHGF, native | 0.3 ms | 0.4 ms | 2.5 ms | 42 ms | 0.04 |

Browser times vary between runs by about one fence-polling interval
(2.7–4.0 ms at small B).

- **Latency, not throughput, is the GPU's limit on small jobs.** One GPU
  thread runs one series at ~1.6 µs per trial (2-level unbounded). That is
  slower than one CPU core, so a single filter belongs on the CPU. The GPU
  catches up with one AOT core at about 6 parameter sets, and with 8 AOT
  isolates at about 30.
- **Browser fixed cost: ~2.5 ms per call.** It is one round trip to the GPU
  process, and it looks quantised by Chromium's fence polling. Waiting
  with `onSubmittedWorkDone` and then mapping cost two round trips, so now
  only the map is awaited. Native has no such floor (0.3 ms).
- **dart2wasm pitfall:** `toJS` on Dart typed data copies element by
  element (~80 ns per byte here, so 228 ms for 2.8 MB). Host arrays that
  get uploaded are JS-backed (`JSFloat32Array.withLength(n).toDart`), which
  makes the upload free. Readback copies with a JS `slice()` before
  `unmap`.
- **WGSL pitfalls:**
  - Tint rejects any infinite or out-of-range `f32` constant, even
    `bitcast<f32>(0x7f800000u)` and `3.40282347e38f`; naga accepts them.
    Non-finite results are flagged per element and reported through a
    3.4e38 sentinel.
  - WGSL doesn't guarantee NaN/∞ semantics, so finiteness is tested on
    the bits.
  - `log(1)` isn't exactly 0, so `logT` is selected.

**Accuracy (f32).** Totals for 1000 ω × 1000 trials against the f64
engine. pyhgf's own float32 mode (JAX's default) on the same problems is
the yardstick:

| scenario | GPU f32 vs f64: median / max rel | JAX f32 vs f64: median / max rel |
|---|--:|--:|
| 2-level standard | 3.3e-7 / 1.1e-6 | 8.9e-8 / 4.0e-7 |
| 2-level eHGF | 1.1e-6 / 5.3e-6 | 1.8e-6 / 1.0e-5 |
| 2-level unbounded | 9.6e-8 / 3.3e-7 | 6.9e-8 / 1.6e-7 |
| 3-level unbounded | 6.9e-7 / 2.0e-6 | 6.0e-7 / 8.0e-7 |
| binary 3-level standard | same 509 ∞, same 3 flips, max 0.012 | same |

Results by fixture:
- **All 31 parity fixtures with a GPU kernel are checked** on the GPU
  (wgpu-native) and on SwiftShader, 5 perturbed parameter sets each.
- **25 of them** stay within 1e-6 (tolerance 1e-5, because 1e-6 fails
  intermittently).
- **`continuous_3level_eHGF`** is non-finite in both f64 and f32, which is
  the known divergence (P1).
- **The other 5 can't be tracked in f32 by pyhgf either:**
  - `constant_bias_eHGF` and `unbounded_overflow` reach variances of
    1e250 / 1e-307 in f64, where JAX f32 returns ∞;
  - `continuous_2level_eHGF` is chaotic;
  - three binary fixtures run close to a runaway, and JAX f32 is 5e-4 to
    9e-3 off on them. **Fixed on the GPU (2026-11-09):** the error came
    from computing `1 − μ̂` by subtraction, which cancels near 1 in f32.
    Binary nodes now carry it as `π̂ / μ̂` (`π̂ = μ̂ · σ(−s)`), and the
    three fixtures match f64 to ~3e-6, better than JAX f32. Only
    `continuous_2level_eHGF` keeps a looser tolerance.

The argmin over the ω grid is the same, or within 1–4 grid steps where
the optimum is flat. That is the same behaviour as JAX f32.

**Routes:**
- WebGPU in the browser works (Chromium on Linux needs
  `--enable-unsafe-webgpu` and Vulkan; Firefox/Zen works too).
- wgpu-native works on desktop (Vulkan; Metal and D3D12 are the same
  code).
- Flutter GPU (3.47.9) has render pipelines only, no compute, so it is no
  route yet. The other routes can serve Flutter: WebGPU in Flutter web,
  wgpu-native through FFI on desktop and mobile.

**What this changes in the design:**
- **f32 is the GPU's precision.** WebGPU has no f64. Vulkan has f64, but
  mobile and Metal don't, and pyhgf's float32 default and rshgf's f32
  vectorised backend set the same bar. The f64 CPU engine stays the
  reference and the parity target. GPU results are "pyhgf-f32 accurate".
- **The kernel boundary is the batched pass.** hgf_fit's drivers already
  yield `Pass`es, so a GPU transport that evaluates passes on `GpuBatch`
  slots in next to the isolate and Web Worker pools.
  - To make that useful for gradient-based fitting, the codegen needs
    forward-mode tangents. That is mechanical given the unrolled code: an
    expression IR with JVP rules, mirroring `batch_tangents.dart`.
  - Until then, central differences cost only 2k+1 elements per subject,
    which the GPU absorbs. In f32, though, finite differences are only
    good to ~1e-3.
- **Grids and populations become cheap.** Exhaustive grid posteriors,
  large hierarchical models, and simulation-based calibration or
  parameter recovery over 10⁵–10⁶ simulated agents now take
  milliseconds.

#### Phase 7b: GPU fitting (2026-11-09): done

`GpuFitter` (hgf_gpu) answers hgf_fit's passes on the GPU, so every
driver runs there: `gpu.fitMap`, `gpu.sampleNuts`,
`gpu.sampleHierarchical`, `gpu.laplace`, `gpu.fitEmpiricalBayes`.
Commands: `melos run test:gpu` (now with `native_fit_test.dart`), `melos
run bench:gpu-fit`.

**Design:**
- **An expression IR with forward-mode AD** (`ir.dart`). The codegen
  builds `Ex` trees, and each `let`/assignment also emits its tangent: a
  `vecN<f32>` per chunk of up to 4 directions, so k = 5 uses one `vec4`
  and one `f32`. The derivative rules are those of `batch_tangents.dart`:
  - an active clamp passes no derivative;
  - Lambert W contributes `W / (z (1 + W))`;
  - quotients use `(da − (a/b) db) / b`.

  A fixed point decides which slot variables carry tangents. It starts
  from the parameter targets and adds every slot an active value is
  assigned to, then recompiles until nothing changes.
  - The unbounded update's `if (observed)` blocks became `select`s, so
    the accumulations are plain sums.
  - Those selects cost nothing measurable: the probe times match HEAD's
    to the run-to-run noise (2.2–3.5 ms at B = 1000; 11.4 ms at 64k).
  - `TrialCompiler` is shared by `HgfKernel` (unchanged API) and
    `FitKernel`.
- **`FitKernel(model, gradients:)`** compiles an `HgfModel`'s likelihood.
  Parameters are set inside the kernel from a per-pass `job` buffer, which
  holds each element's subject, the initial slots, `x [i][e]` and
  `dx/dy [i][e]`. Their tangents are seeded with `dx/dy`, so the kernel
  returns the gradient with respect to the unconstrained parameters
  directly.
  - Subjects' series are uploaded once per subject list (`u32` words, a
    5-entry subject table with absent-array sentinels), and each element
    loops to its own subject's trial count.
  - So a pass of mixed lengths, masks, time steps and missed responses
    (NaN, tested on the bits) is one dispatch.
- **Responses:**
  - observation surprises: `TotalGaussianSurprise`,
    `FirstLevelGaussianSurprise`, `FirstLevelBinarySurprise`;
  - readouts with per-element parameters in slots: `GaussianReadout`,
    `BinarySoftmaxReadout`.

  `HgfKernel`/`GpuBatch` take them too (`responses:`,
  `setResponseParameter`).
- **f64 where it matters.** `Pass.precise` (new in hgf_fit) marks
  finite-difference and Hessian passes, and the values `fitMap` reports.
  `GpuFitter` runs those on the CPU (on `cpu:`, a `FitWorkers`, if given),
  as well as any model without a kernel. So `laplace` is bit-identical to
  the CPU's at the GPU's optimum. The priors and Jacobians are added in
  f64 on the host (`HgfModel.completeNegLogPosterior`).
- hgf_fit's driver extensions are now on a `PassRunner` interface;
  `FitWorkers` implements it.

**Accuracy** (`test/src/fit_check.dart`, RX 7600 XT and SwiftShader,
dart2wasm and dart2js). Six models, each element checked with and without
gradients and Jacobian:
- values within 3.6e-7 relative of f64;
- gradients within 1.1e-5 of the gradient's scale (most 1e-6);
- `fitMap` estimates within 2e-3 of the CPU's (unconstrained);
- NUTS posterior means within 0.25 sd.

**Speed** (`bench:gpu-fit`, binary 2-level recovery problem, 320 trials;
the CPU is the 8-isolate `FitWorkers` pool on the i7-7700):

| job | agents | pool (8 isolates) | GPU |
|---|--:|--:|--:|
| MAP | 20 | 7.3 ms | 13.8 ms |
| MAP | 200 | 34 ms | 25 ms |
| MAP | 2000 | 332 ms | 89 ms |
| NUTS 4×(300+300) | 20 | 4.2 s | 2.1–3.0 s |
| NUTS 4×(300+300) | 200 | 34 s | 5.0–5.4 s |
| NUTS 4×(300+300) | 2000 | 319 s | 30–31 s |

NUTS rows as of 2026-11-09 with `NutsEnsemble` (below); before it, GPU
2.4 s / 9.4 s / 139 s and pool 4.5 s / 34.7 s / 395 s. The 20-agent GPU
time is mostly round-trip latency, which varies by ±0.5 s between runs.

**Finding: at thousands of chains, the NUTS driver is the bottleneck.**
At 2000 agents (8000 chains), the GPU passes take 7 s of a 55 s run. The
rest is the sampler's own host work per chain and leapfrog step.
- `benchmark/bin/nuts_overhead.dart` measures it with no filter:
  1.9 µs per chain-step at 400 chains, 3.1 at 1600, 6.1 at 8000.
- It is not GC (scavenges are under 1 s, old gen 3.4 MB). It grows with
  the number of suspended generator stacks (recursive `sync*`
  `_buildTree`), which looks like cache footprint.
- A structure-of-arrays NUTS that advances all chains with flat arrays
  instead of one generator per chain would remove most of it. That is
  the lever for GPU NUTS at scale: about 7× at 2000 agents.

**Done (2026-11-09): `NutsEnsemble`** (`sample/nuts_ensemble.dart`).
- Every NUTS driver now advances its chains as a state machine.
- Each chain's states, vectors and per-level tree locals live in one
  `Float64List` at fixed offsets.
- Stan's `build_tree` recursion is an explicit stack of frames: 7 buffer
  offsets and a state per tree level.
- The order of operations and random draws is unchanged, so every chain's
  draws are **bit-identical** to `Nuts.chain`. That generator stays as the
  readable reference.
  - `test/nuts_ensemble_test.dart` checks this for all three metrics,
    divergences, max-depth hits, depth 0, short or no warmup, and failed
    starts, on the VM, dart2js and dart2wasm.
  - Fingerprints of `sampleNuts` (exact and forward differences) and
    `sampleHierarchical` matched before and after.
- `BatchedObjective.evaluateGradientPasses` hands the ensemble's flat
  points to a single pass, with no per-point lists.
- Driver cost per chain-step (`nuts_overhead`, AOT): 0.73 / 0.68 / 0.87 µs
  at 400 / 1600 / 8000 chains (before: 1.7 / 2.7 / 5.8). It no longer
  grows with the number of chains.
- GPU NUTS: 4.6× faster at 2000 agents (139 → 30 s) and 1.8× at 200. The
  pool gains 1.24× at 2000 agents, because its sampler also runs on the
  main isolate.

**What is left at 2000 agents (30 s):**
- GPU wait: 3.4 s.
- Host work per pass: 7 s (prepare 1.8 + finish 5.2), about 210 ns per
  element. That is the transforms, the `f32` job packing, and the priors
  and Jacobians in `completeNegLogPosterior`.
- The sampler itself: about 20 s, around 0.6 µs per chain-step. A perf
  profile is flat: copies, tree bookkeeping, the RNG and `exp`/`log` in
  the weights. Swapping `setRange` for loops changed nothing.
- Overlapping host and GPU work would save at most the 3.4 s wait.
- The next lever is to run the ensembles on worker isolates and gather
  their points into one GPU pass per round: about 3.5× on the sampler's
  share on 4 cores. The per-element host cost could also move into the
  kernel (transforms) or be vectorised.

**Next steps (Phase 7 follow-up):**
- [ ] HGF neural networks: port rshgf's vectorised backend
  (`src/vectorised/`, about 3.4k lines, f32 by default). It is per-layer
  matrix–vector products (GEMM over samples), elementwise volatile-node
  updates and rank-1 weight updates, which is GPU-shaped by construction.
  - [x] CPU baseline (2026-11-09): `packages/hgf_deep`, see "HGF neural
    networks (hgf_deep)" below.
  - [x] WGSL (tiled GEMM plus elementwise kernels), f32, checked against
    the f64 CPU engine like hgf_gpu (2026-11-10): `GpuDeepNetwork`, see
    "hgf_deep on the GPU" below.
  - [ ] The sequential `fit` on the GPU: one persistent workgroup per
    network looping over samples (barriers instead of dispatches), which
    would also run ensembles of networks, one per workgroup.
- [x] Tangents in the codegen (exact gradients on the GPU), then a GPU
  `Pass` transport in hgf_fit (MAP, NUTS with many chains, hierarchical):
  Phase 7b.
- [x] Response models with response data (`BinarySoftmax`,
  `GaussianResponse`): Phase 7b. Per-trial surprise output is still open.
- [x] Structure-of-arrays NUTS across chains (`NutsEnsemble`,
  2026-11-09): GPU NUTS on 8000 chains 139 → 30 s.
- [ ] Sampler ensembles on worker isolates feeding one GPU pass per round,
  and less host work per element (see the Phase 7b note): the remaining
  ~27 s of host time at 2000 agents.
- [ ] `GpuFitter` with `ResponseModel.fn`: needs a way to express the
  response in WGSL (an `Ex` builder callback?).
- [ ] Vector node kinds on the GPU (ef-state first). dp-state's
  data-dependent branching is the hard one.
- [ ] Platform coverage still to measure:
  - Intel and Apple GPUs, and Android (Vulkan) and iOS (Metal) through
    wgpu-native;
  - Firefox and Safari WebGPU.
- [ ] CI: run `test/gpu_parity_test.dart` on SwiftShader. The
  `chrome-gpu` platform names `chromium`, but CI has Chrome through a
  wrapper.
- [ ] Tuning not tried yet: the workgroup size (64 is used), the memory
  layout at very large B, and a kernel variant whose state is in
  registers only, without a store between dispatches.
- [x] The GPU and the newer fitting features in the playground
  (2026-11-09): see Phase 8.

#### HGF neural networks (hgf_deep, 2026-11-09): CPU baseline done

`packages/hgf_deep` ports pyhgf's vectorised `DeepNetwork`: layers of
volatile, binary and categorical nodes, dense or one-to-one weights, every
volatility update, relaxed or mean-field, feedforward uncertainty, carried
or reset precisions, tonic volatility; `fit` (sequential), `batchUpdate`
(returns the input-layer errors), `predict`; SGD and Adam with optax's
arithmetic; four weight initialisers. Not ported: continuous layers
(`input_data`), `synaptic_uncertainty` learning, `update_input_layer`,
`LayerStack`, the two-phase batch step, sample weights.

- **Reference:** pyhgf's JAX backend in float64, not rshgf. It is a
  superset of rshgf's features, and rshgf's batched kernel is wrong (R4).
  **f64 on the CPU** (the roadmap said `Float32List`): Dart has no scalar
  f32 arithmetic, f64 keeps the 1e-9 parity standard, and f32 is the GPU's
  job, as in hgf_gpu.
- **Parity:** 26 fixtures (`gen_deep_fixtures.py`, `melos run
  fixtures:deep`) match to 1e-9 on predictions, recorded trajectories,
  input errors, weights and every belief field after every operation, NaN
  onsets included. 23 behavioural tests besides.
- **One kernel set** over `[node][sample]` buffers serves `fit` (width 1)
  and `batchUpdate` (width B), mirroring JAX's `vmap` of the per-sample
  sweeps.
- **Performance** (`melos run bench:deep`, one core, 2026-11-10): sequential
  `fit` beats JAX f64 and rshgf on a 2-8-8-4 network (17.1 vs 30.2 / 23.1 ms
  per 2000 samples) and is ~2× behind JAX f64 on 10-64-64-16 and
  16-256-256-32 (still ahead of rshgf on the former). `batchUpdate` and
  `predict` trail JAX f64 by 3–5× (medium 17.0 / 9.3 ms vs 6.0 / 2.6; wide
  90 / 17 ms vs 24 / 3.4; they were 22.5 / 14.6 and 143 / 29 before the
  2026-11-10 changes below). The wins that mattered: a register-tiled GEMM,
  width-1 kernels written as dot products or AXPYs, a fused posterior
  contraction, and hoisting Adam's fields into locals (Adam was 27 ns per
  weight because of field loads in the loop; 5 ns after).
  - **SIMD GEMM (2026-11-10):** on native targets the batched GEMM runs on
    `Float64x2` lanes over samples, a 4-row × 4-sample tile (8
    accumulators). Each lane still sums `k` in ascending order, so it is
    **bit-identical** to the scalar 2×4 tile (`test/matmul_test.dart` checks
    every remainder shape against a misaligned view, which forces the
    scalar path). Kernel: 9 GF/s vs 3.2 (2.8×); 4×6, 6×4, 3×8 tiles measured
    no better. Gated by the const `useSimd` (off on dart2wasm and dart2js,
    where `Float64x2` is emulated).
  - **tanh:** hgf's `CouplingFn.tanh` now uses Cephes' scheme (rational
    below 0.625, `1 − 2/(e^{2|x|}+1)` above): 15 ns vs 28, and 1.3 ulp vs
    4.5 against mpmath (`hgf/test/coupling_test.dart`).
  - **Weight gradient:** `_addOuter` (Σ_s u·h / b) was one latency-bound
    dot product per weight. It now transposes `h` to `[s][j]` (padded to
    an even width) and runs as a `matMul` with the samples as depth: same
    summation order, so the same bits (checked by hashing `batchUpdate`
    outputs before and after), 40 MFLOP per step on the wide net through
    the tiles instead of one accumulator. Scratch whose size changes per
    layer uses `Workspace.atLeast` (grow-only): reallocating it per call
    cost 50% on the small net.
  - Together: `predict` 1.6–1.75× faster, `batchUpdate` 1.3–1.6×, `fit`
    ~8%.
  - **Where the batched time goes now** (`phase_profile 16,256,256,32
    256`): `updateSweep` 57 of ~80 ms, and in it the unbounded volatility
    update at ~300 ns per element (exp/log/Lambert W, 19.5 ms per
    256-node layer). JAX vectorises those transcendentals. On the CPU the
    fix would be stage-wise kernels as in W4's `BatchNetwork`; the GPU port
    removes it.
  - **Bug fixed:** `weightInitialisation(seed:)` seeded each layer with
    `Object.hash(seed, l)`, which Dart randomises per process, so a seed
    did not reproduce across runs. Now a fixed arithmetic mix.
  - **What is left is hardware:** JAX's GEMM reaches ~23 GF/s on one core
    (AVX2 + FMA); Dart's `Float64x2` is 128-bit SSE without FMA. The
    batched path's real lever is the GPU port.
  - **Packages evaluated (2026-11-10):** `ml_linalg` 13.12.9 (maintained,
    150 points) is slower than hgf_deep's *scalar* kernel even in f32
    (matrix product = one row `Vector` × matrix per output row, allocating;
    1.2–1.9 GF/s f64), and 20× slower on Wasm; its API is immutable and
    allocating, unlike the in-place strided kernels here. `simd` 0.0.1 is
    an empty placeholder (needs Dart 3.14 beta; it points at the SDK
    tracking issue above). Neither is worth a dependency; the useful
    finding was that hand-written `Float64x2` over the sample lanes keeps
    bit-parity.
- **Found upstream** (with repros and, where it is a code bug, a verified
  fix in `docs/upstream/`):
  - P18: `DeepNetwork`'s volatility levels run away under all three
    updates, frozen weights included, then NaN (JAX and rshgf).
  - R3: rshgf's GELU uses Abramowitz–Stegun `erfc` (1.5e-7). hgf had copied
    it; hgf now has a full-precision `erfc`.
  - R4: rshgf's batched kernel weights the g″ term by `eff` instead of the
    relaxed gain (stale since #456) and ignores `predict_precision`; 0.26
    off JAX on the first batch. Patched and rebuilt with `--features f64`,
    rshgf matches JAX to 1e-15.
  - R5: rshgf panics with `predict_precision=False`.
  - Also: rshgf's `f64::max/min` drop NaN, so rshgf keeps going silently
    where JAX has already gone NaN. That explained all the remaining
    per-sample `fit` gaps.
- **Method that worked:** build rshgf with `--features f64` (maturin
  directly; `uv sync` reuses the cached f32 wheel and ignores
  `MATURIN_PEP517_ARGS`) and compare per op. f32 noise hid which gaps were
  bugs. JAX in float32 also helps as a noise yardstick.

#### hgf_deep on the GPU (2026-11-10): done

`GpuDeepNetwork` (hgf_gpu, `lib/src/deep/`) runs `DeepNetwork`'s batched
operations on the GPU in f32: `batchUpdate`, `trainEpoch` (a whole epoch
of minibatches queued with one upload and no readback), `predict`, and
`upload`/`download` to and from a CPU `DeepNetwork`. The weights, Adam's
moments and the carried beliefs stay on the GPU. The sequential `fit`
stays on the CPU.

- **Kernels** (`DeepKernels`, generated per network shape and batch
  capacity). One step is about 3·layers + 2 dispatches:
  - one kernel per layer for the top-down prediction: a tiled GEMM
    (64×64 tiles, 4×4 outputs per thread) whose epilogue does the layer's
    elementwise work (volatility-level prediction, precisions, `g(μ̂)` for
    the layer below), and a second accumulator, `W²·g′²/π̂`, with
    feedforward uncertainty;
  - one per interior layer for the bottom-up pass: a transposed GEMM with
    two accumulators (`Wᵀ·gain·δ`, `(W²)ᵀ·π_eff`) whose epilogue is the
    value posterior, the prediction error, the volatility update (standard,
    eHGF or unbounded) and the child-side factors for the next layer up;
  - per layer, the weight gradient (a GEMM over samples) fused with the
    SGD or Adam step. When a layer's output tiles are too few to fill the
    GPU it splits the samples into chunks (split-K) and adds a small
    reduce-and-step kernel; only when each chunk gets ≥ 256 samples,
    because at 128 the extra dispatch cost more than it saved;
  - one reduction kernel for `carry_precisions` and the batch-mean
    increments of the carried beliefs.
  The batch width is a uniform (any width up to the capacity); shapes,
  kinds, couplings and settings are compiled in.
- **Backend change:** `GpuBackend` now records dispatches and copies into
  one command buffer (consecutive dispatches in one compute pass) and
  submits at the next queue write, read or `flush`; bind groups are
  cached. Before, every dispatch was its own submission with a new bind
  group. An empty dispatch costs 1.8 µs in a long chain (3.8 µs at 12 per
  submission). The existing hgf_gpu tests are unchanged.
- **Accuracy** (`test/native_deep_test.dart` on wgpu-native, the same
  check on Chromium's WebGPU/SwiftShader in `gpu_parity_test.dart`). Every
  `batch_update` and `predict` of the 26 hgf_deep fixtures runs on the GPU
  from the CPU's state just before it, with fresh optimisers on both sides.
  Input errors, weights, predictions and carried beliefs match f64 to
  ≤ 1e-6 relative to max(1, |x|). After 10 Adam steps on random 5-37-70-6
  networks the gap is 2.6e-6, and the split-K path matches too. Tolerance
  2e-4.
  - Skipped as outside f32 (all P18 runaway states): 4 batch updates whose
    starting state holds values f32 can't represent (means of 1e94 and
    1e129) or a predicted variance above √(f32 max) (5.8e33, 9e49), where
    the posterior's mean shift squares to infinity.
  - One allowed NaN: in `mean_field_eHGF`, two samples send a volatility
    mean from 4.5 to 130 and 137 in one step, so eHGF's `exp(ω + μ_vol)`
    overflows f32. JAX's float32 mode fails the same way.
  - NaN propagation: WGSL's `min`/`max`/`clamp` may drop NaN, and a NaN
    belief then turned into finite gradients (weights off by ±lr). The
    clamps use NaN-propagating `minN`/`maxN` (bit tests), so NaN states
    zero their gradients as on the CPU.
- **Lambert W in f32:** the unbounded update's `W₀(exp(L))` runs as
  Newton on `w + ln w = L` for `L ≥ 1`, so it never forms `exp(L)`
  (which overflows f32 above 88).
- **Speed** (`melos run bench:gpu-deep`, RX 7600 XT via RADV, best of 20;
  CPU f64 on one core of an i7-7700, JAX f32 on one core). Per learning
  step, with Adam:

  | network | samples/step | GPU, epoch (per step) | GPU, one call with readback | CPU f64 | JAX f32 |
  |---|--:|--:|--:|--:|--:|
  | small 2-8-8-4 | 256 | 0.58 ms | 1.1 ms | 2.1 ms | 0.98 ms |
  | | 4096 | 0.89 ms | 1.3 ms | 36 ms | |
  | medium 10-64-64-16 | 256 | 0.55 ms | 0.82 ms | 17.6 ms | 3.1 ms |
  | | 4096 | 1.0–1.1 ms | 2.1 ms | 301 ms | |
  | wide 16-256-256-32 | 256 | 0.71–0.74 ms | 1.2 ms | 89 ms | 11.4 ms |
  | | 4096 | 3.2–3.3 ms | 5.6 ms | 1540 ms | |

  `predict` on 32768 rows: 2.0 / 7.5–7.9 / 14 ms (CPU 15 / 174 / 3350 ms).
  JAX f32 predicts 2000 rows of the medium network in 1.4 ms and 500 of the
  wide one in 2.1 ms.
- **What bounds it now:** at 256 samples every network costs about the
  same ~0.55–0.75 ms per step, and dispatch overhead is only ~50 µs of
  that. The grids are small (a 256×256 GEMM is 16 workgroups of 64×64 on
  32 compute units), so smaller tiles for small batches are the next
  lever. A single call also pays one GPU round trip (~0.3–0.5 ms
  native); `trainEpoch` avoids it.
- **Pitfalls:** `precision` is a reserved word in WGSL. Each kernel
  declares only the buffers it uses, because the bind-group layout is
  `auto`, and `dispatch` now takes `null` for a binding a kernel leaves
  out. The GPU clocks down between light calls, so time with a few
  warm-up runs and best-of-many.

### Phase 8: App -> HGF GUI modelling

- [ ] Since the performance is excellent so far, the playground can be
  extended to be a fully-fledged fitting app, where the user can load
  and map data, fit a model and view and save the results.
- [x] Make sure the app runs with GPU if available, and allow user to
  toggle GPU, precision, other settings/preferences (2026-11-09, quick
  wins; see below).
- [ ] Consider allowing real-time sources and define a model for them
  (e.g. sensor data, think about including LSL streams with liblsl.dart).
- [ ] Allow custom agents / models and parameters, to be able to e.g.
  fit something like the Iowa Gambling Task, or abitrary experimental
  data.

#### Phase 8 quick wins (2026-11-09): done

- **GPU in the playground.** `Fitting` runs passes on a `GpuDevice`:
  WebGPU through `GpuFitter` on the web (its calls are asynchronous, so the
  page keeps drawing), and on desktop wgpu-native through the new
  `WgpuIsolateFitter` (hgf_gpu): a `GpuFitter` in a background isolate,
  since wgpu-native's calls block. Precise passes (Laplace's Hessian, the
  reported values) still run on the CPU pool. A model without a kernel
  falls back to the CPU, and the result says so.
- **Settings drawer** (`SettingsPanel`, saved with `easy_shared_preferences`):
  theme; device (Auto / CPU f64 / GPU f32) with the adapter found or why
  there is none; workers (default now one per logical processor, was 4;
  `FitWorkers.defaultCount`); MAP restarts; NUTS chains, warm-up, draws,
  target acceptance, max tree depth. The fit card shows what the next fit
  runs and links to it.
- **Reset button** (app bar): the preset back as it ships (parameters,
  couplings, update scheme, data settings).
- `GpuFitter.cancel()` and `yieldInterval`.

**When the GPU pays off for one series.** Each NUTS leapfrog round is a
GPU round trip, so a few chains lose to CPU workers; many chains win.
Auto switches to the GPU at 32 chains (`FitConfig.autoGpuChains`).
NUTS 500 + 500 iterations, RX 7600 XT, i7-7700:

| where | chains | CPU (4 workers) | GPU |
|---|--:|--:|--:|
| VM, binary 2-level, 320 trials (`fit_gpu --agents 1`) | 4 | 1.3 s | 2.5 s |
| | 16 | 2.5 s | 2.7 s |
| | 64 | 6.5 s | 4.5 s |
| | 256 | 21 s | 13 s |
| Wasm page, binary 3-level, 400 trials (`?fitbench`) | 4 | 12 s | 30 s |
| | 64 | 89 s | 37 s |
| Flutter test VM, binary 3-level, 400 trials | 64 | 63 s | 19 s |
| Linux desktop app, same (UI idle / a frame every vsync) | 64 | — / 93 s | 37 / 55 s |

MAP on one series: 4 ms CPU vs 14 ms GPU (VM); 50–90 ms vs 90–160 ms in
the apps.

**Found:**
- In the desktop app, wgpu-native on the UI isolate (with yields) held
  frames at 33–110 ms; `WgpuIsolateFitter` gives 60 fps (median 16.7 ms)
  at ~3% cost in the VM bench.
- The Flutter desktop UI isolate still costs 1.5–2× on fits whose driver
  runs there (GPU 19 s in the VM vs 37 s in the app, idle; CPU pool 63 vs
  93 s with frames). Every pass's reply waits for the UI isolate's event
  loop. Next lever: run the whole job (driver + pool + GPU) in a host
  isolate and send only progress and results.
- A 64 × 500 NUTS summary takes ~0.4 s (rank normalisation sorts 32k
  draws several times per parameter), visible as a 0.5–1 s frame at the
  end of a fit. The app no longer calls `converged()` (a second summary);
  `diagnostics.dart` could share the z-scaled draws between R̂ and bulk
  ESS, or the summary could move off the UI thread.

**Next (Phase 8):** fit host isolate (above); a multi-agent parameter
recovery view (simulate N agents, fit all, true-vs-estimated scatter), which
is where the GPU shines (MAP on 2000 agents in 89 ms); the remaining
items above.

## Upstream issues

Found while porting and optimising; none reported yet. Details, suggested
fixes and runnable repros are in [upstream/README.md](upstream/README.md)
(pyhgf/rshgf scripts in `upstream/pyhgf/`, dart2wasm ones in
`upstream/dart2wasm/`). What hgf.dart does about each is noted there too.

- **pyhgf (JAX, and rshgf where noted)**
  - [ ] [P1](upstream/README.md#p1) relaxed eHGF diverges (missing MGF
        terms; JAX and rshgf). Fix ready (`gen_fixtures.py`); hgf.dart has
        it opt-in as `ehgfMgfCorrection`. **The one that matters most.**
  - [ ] [P2](upstream/README.md#p2) `lambert_w0` NaN for z ≳ 2.7e151 (`f1²`
        overflows).
  - [ ] [P3](upstream/README.md#p3) the update-sequence move loop mutates
        the list it iterates, scrambling the deferred steps (hgf.dart
        replays it).
  - [ ] [P4](upstream/README.md#p4) a parent of two ef-state nodes crashes
        the scheduler (`ValueError`).
  - [ ] [P5](upstream/README.md#p5) `add_ef_state` ignores `xis`.
  - [ ] [P6](upstream/README.md#p6) ν = ±∞ on missing observations with HGF
        learning.
  - [ ] [P7](upstream/README.md#p7) stale `update_fn` in the posterior pass:
        a categorical or dp-state node after a continuous one crashes (both
        reproduced 2026-11-06).
  - [ ] [P8](upstream/README.md#p8) JAX and rshgf order independent
        predictions differently (cosmetic).
  - [ ] [P9](upstream/README.md#p9) licence: MIT vs GPL-3.0.
  - [ ] [P10](upstream/README.md#p10) `categorical-state` with `n_nodes > 1`
        builds a broken graph.
  - [ ] [P11](upstream/README.md#p11) `binary_parameters` ignored; docstring
        says `3n + 2` nodes (it's `2n + 2`).
  - [ ] [P12](upstream/README.md#p12) categorical α = ±∞ on missing
        observations.
  - [ ] [P13](upstream/README.md#p13) `dirichlet_kullback_leibler` NaN once
        Σα > 171 (Γ overflows).
  - [ ] [P14](upstream/README.md#p14) dp-state ignores `observed`: missing
        samples are clustered, a NaN one opens a NaN cluster. Fix ready
        (`gen_fixtures.py`); hgf.dart skips them.
  - [ ] [P15](upstream/README.md#p15) dp-state results depend on JAX's PRNG
        mode and precision; pyhgf's tests pin the legacy mode and expect
        float32 values. Proposals are redrawn every trial from the same key.
  - [ ] [P16](upstream/README.md#p16) dp-state `batch_size` ≠ number of
        clusters fails inside the scan (also with the default 10).
  - [ ] [P17](upstream/README.md#p17) the unbounded volatility update (the
        default) runs away on 3-level binary HGFs: μ₂ only ratchets up until
        μ₁ explodes (JAX and rshgf; found with the playground, 2026-11-06).
  - [ ] [P18](upstream/README.md#p18) `DeepNetwork`'s volatility levels run
        away (all three updates, frozen weights too), then NaN (2026-11-09).
- **rshgf**
  - [ ] [R1](upstream/README.md#r1) ef-state ignores `observed`; univariate
        fixed rate only.
  - [ ] [R2](upstream/README.md#r2) hot-path costs (string compares,
        per-call `Vec`s, clones when recording).
  - [ ] [R3](upstream/README.md#r3) GELU uses a 1.5e-7 `erfc` approximation.
  - [ ] [R4](upstream/README.md#r4) `DeepNetwork.batch_update` kernel: stale
        g″ weighting, ignores `predict_precision` (fix verified).
  - [ ] [R5](upstream/README.md#r5) `DeepNetwork(predict_precision=False)`
        panics.
- **dart2wasm**
  - [ ] [D1](upstream/README.md#d1) a typed-data view anywhere slows reads
        of non-view lists held in fields (~10%).
  - [ ] [D2](upstream/README.md#d2) `1 / 6` with int literals isn't folded.
  - [ ] [D3](upstream/README.md#d3) top-level `final` tables are null-checked
        at every use, with the initialiser inlined there.
  - [ ] [D4](upstream/README.md#d4) SIMD types emulated (10–15× slower).
  - [ ] [D5](upstream/README.md#d5) exp/log are JS imports.
  - [ ] [D6](upstream/README.md#d6) no way to drop `array.get` bounds checks.
  - [ ] [D7](upstream/README.md#d7) `Expando<int>` loses its values when it
        is the program's only Expando.

## Questions answered by the user

- Julia comparison: the new Julia HGF, once usable. Which scenarios and which
  version?
  - I think the Julia version can be tested against the current, latest:
    https://github.com/ComputationalPsychiatry/HierarchicalGaussianFiltering.jl
    juliaup is installed with the latest julia in:
    /home/zeyus/.juliaup/bin
- Which response models matter most for phase 6 (binary softmax, Gaussian,
  custom)?
  - I think Guassian is the most important, then binary softmax, then custom.
    everything should be implemented, but Gaussian is the most interesting.
- The playground's audience (students, workshops, experiment demos):
  this affects phase 5 scope.
  - All of the above, maybe a useful teaching tool in general for people to
    learn about the HGF and applications.
    I think it also serves as documentation for the engine.
- Repository and publishing: GitHub org (NexusDynamic?), and pub.dev name
  `hgf`.
  - GitHub org will likely be the one from the core team. Pub.dev likely hgf.
    Both are undecided yet, but the group's org is:
    https://github.com/ComputationalPsychiatry

## Starting a new session

Phase W is done apart from the optional SIMD variant and W5 (both AOT
only). Phase 4's node kinds are all done: ef-state (2026-11-05),
categorical-state and dp-state (2026-11-06). Left in Phase 4: per-child
coupling functions (only if needed), and batched kernels / `sample` for the
vector kinds (`BatchNetwork` rejects them). Phase 5's first version is
done (2026-11-06): `packages/hgf_widgets` and `apps/hgf_playground`, with
views and presets for every node kind; see its open items. CI runs on
every push, and the playground deploys to
https://zeyus-research.github.io/hgf.dart/ from `main`. Phase 6 has
started: `packages/hgf_fit` does MAP estimation, the Laplace
approximation, NUTS, exact gradients and hierarchical models
(2026-11-06), and runs them on worker pools (`FitWorkers`: isolates, and
Web Workers on the web, 2026-11-06). The playground has a fitting view on
those workers (2026-11-07). See Phase 6's open items: next are batched
kernels and tangents for the vector node kinds. Phase 7's GPU probe is
done and a go (2026-11-07, `packages/hgf_gpu`); its follow-ups (HGF neural
networks, GPU tangents and a GPU transport for hgf_fit) are listed there.
The HGF neural networks have a CPU baseline (`packages/hgf_deep`,
2026-11-09, parity with JAX f64) and run batched on the GPU
(`GpuDeepNetwork`, 2026-11-10). Open there: the sequential `fit` (and
ensembles of networks) on the GPU, tile sizes for small batches, and a
browser benchmark.

```sh
cd hgf.dart
melos bootstrap
melos run test && melos run test:parity           # should be all green
melos run test:wasm-release                       # parity at dart2wasm -O4
melos run bench:playground                        # playground frame times (Wasm)
melos run bench:fit                               # MAP fitting vs pyhgf + JAX
melos run bench:nuts                              # NUTS vs PyMC and NumPyro (~8 min)
melos run bench:hier                              # hierarchical NUTS vs NumPyro, PyMC (~4 min)
melos run bench:parallel                          # FitWorkers speed-ups (~15 min)
melos run test:web-workers                        # FitWorkers on Web Workers (Chrome)
melos run bench:web-parallel                      # ... and their speed-ups (~10 min)
melos run bench:playground-fit                    # the playground's fits (~3 min)
melos run fixtures:deep && melos run bench:deep   # hgf_deep: fixtures, speed vs pyhgf
melos run test:gpu && melos run bench:gpu-deep    # GPU kernels incl. GpuDeepNetwork
(cd apps/hgf_playground && flutter run -d chrome --wasm)   # try it
dart compile exe packages/hgf/benchmark/bin/profile_modes.dart -o /tmp/pm && \
  (cd packages/hgf && /tmp/pm)                    # ns/trial baseline per mode
```

Re-read this file and `docs/FEASIBILITY.md`, then pick the next unchecked
item. After pulling pyhgf, run `melos run fixtures` first.
