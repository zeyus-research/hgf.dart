# hgf.dart roadmap

Picks up after the feasibility spike (phases 1–3, see
[FEASIBILITY.md](FEASIBILITY.md)). The original plan lives at
`~/.claude/plans/so-i-m-associated-with-radiant-cloud.md`; this file
supersedes its phase list.

## Where things stand (2026-11-06, after W4, ef-state and categorical)

- `packages/hgf` covers:
  - continuous, binary, constant (bias), exponential-family (`ef-state`)
    nodes, the latter with fixed or HGF-implied (`hgf-1/2/3`) learning,
    and categorical nodes (`categorical-state`);
  - standard, eHGF and unbounded volatility updates, relaxed or mean-field;
  - named and custom coupling functions, missing observations, irregular
    time steps;
  - `inputData`, `predict`/`step`, `setInitial`, `Trajectories`,
    `NetworkSpec`;
  - `BatchNetwork`: lock-step batches with per-element parameters and
    data, bit-identical to `Network` (W4);
  - surprise and response functions (`Network.surprise`), generative
    sampling (`Network.sample`), `saveState`/`restoreState`.
- 414 tests pass. Parity with float64 pyhgf JAX holds to 1e-9 on 55 fixtures
  (3 of them against a patched pyhgf, see `ehgfMgfCorrection`; 11 ef-state,
  7 categorical),
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
  Wasm SIMD. Re-check each release.
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
  - [ ] Dirichlet process (`dp-state`)

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
      with `-O4` and runs it under Node: 34/34 match. For the playground
      (phase 5), check that `flutter build web --wasm` builds at `-O4`, and
      pass `-O4` if it doesn't.
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

- [ ] `packages/hgf_widgets`: trajectory plot (μ ± 1/√π bands, prediction
      errors) and a network graph view. Chart library vs CustomPainter:
      decide on performance for 10k points.
- [ ] `apps/hgf_playground`: sliders for ω/κ/μ₀/π₀ with a live re-run (the
      engine re-runs 600 trials in under 1.5 ms on Wasm), trial-by-trial
      stepping via `predict`/`step`, and loading CSV data. **Build with
      `flutter build web --wasm`** as the primary target; desktop and mobile
      too.
- [ ] Measure frame time with the engine running each frame.

### Phase 6: fitting probe (`packages/hgf_fit`)

- [ ] Objective: total surprise (phase 4) plus response models.
- [ ] Gradients, cheapest first:
  1. central finite differences on the batched engine (W4): 2·k extra
     forward passes for k parameters, which is cheap;
  2. hand-written forward-mode tangents (a JVP of each update step),
     if HMC needs exact gradients;
  3. a dual-number engine variant. A generic numeric type would cost the
     f64 fast path, so it would be a separate code path, perhaps generated.
- [ ] Optimisers: L-BFGS and Nelder–Mead for MAP; then a minimal HMC/NUTS.
- [ ] Parameter recovery on simulated data, compared with
      `docs/source/notebooks/4-Parameter_recovery.ipynb`.
- [ ] Export draws for ArviZ (netCDF is heavy; start with CSV/JSON).

### Phase 7: GPU probe (go/no-go)

- [ ] **First target: the batched filter, not deep networks.** One GPU
      thread per parameter set, in a WGSL compute shader (f32). It is
      embarrassingly parallel, matches W4's layout, and is where JAX
      currently wins.
- [ ] Routes:
  - WebGPU via `dart:js_interop` (Wasm/web builds);
  - Flutter GPU (check compute shader support at the time);
  - a native WebGPU (Dawn/wgpu) FFI binding for desktop.
- [ ] Then one vectorised layer step (`src/vectorised/layer.rs`) with a
      `Float32List` CPU baseline.
- [ ] Record platform coverage, upload/readback overhead, and the batch size
      at which the GPU starts to beat the CPU.

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
        a categorical node after a continuous one crashes (reproduced
        2026-11-06; recheck with dp-state's ef parents).
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
- **rshgf**
  - [ ] [R1](upstream/README.md#r1) ef-state ignores `observed`; univariate
        fixed rate only.
  - [ ] [R2](upstream/README.md#r2) hot-path costs (string compares,
        per-call `Vec`s, clones when recording).
- **dart2wasm**
  - [ ] [D1](upstream/README.md#d1) a typed-data view anywhere slows reads
        of non-view lists held in fields (~10%).
  - [ ] [D2](upstream/README.md#d2) `1 / 6` with int literals isn't folded.
  - [ ] [D3](upstream/README.md#d3) top-level `final` tables are null-checked
        at every use, with the initialiser inlined there.
  - [ ] [D4](upstream/README.md#d4) SIMD types emulated (10–15× slower).
  - [ ] [D5](upstream/README.md#d5) exp/log are JS imports.
  - [ ] [D6](upstream/README.md#d6) no way to drop `array.get` bounds checks.

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
only). Phase 4: ef-state (2026-11-05) and categorical-state (2026-11-06)
are done. Next: **dp-state**, which reads ef parents' `xis` and needs ef
nodes with value children (a spec ef node can't declare edges yet), and
changes which branches are active at run time (see the W constraints). It
reuses the vector side store; its steps go in `_dispatchVectorStep`, out of
the inlined hot switch. Watch for [P7](upstream/README.md#p7): pyhgf's
stale `update_fn` will hit ef parents with children too. New node kinds
need a batched kernel too (or `BatchNetwork` rejects them; see the W4
notes).

```sh
cd hgf.dart
melos bootstrap
melos run test && melos run test:parity           # should be all green
melos run test:wasm-release                       # parity at dart2wasm -O4
dart compile exe packages/hgf/benchmark/bin/profile_modes.dart -o /tmp/pm && \
  (cd packages/hgf && /tmp/pm)                    # ns/trial baseline per mode
```

Re-read this file and `docs/FEASIBILITY.md`, then pick the next unchecked
item. After pulling pyhgf, run `melos run fixtures` first.
