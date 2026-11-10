# How hgf.dart computes gradients

hgf.dart fits HGF models by MAP (L-BFGS), Laplace, NUTS and hierarchical
empirical Bayes. All of these need the gradient of the negative log
likelihood with respect to the model parameters. This note explains how
that gradient is computed without an automatic differentiation (AD)
package such as JAX, Zygote/Enzyme/ForwardDiff (Julia) or PyTorch, how the
results are checked, and where reverse mode stands.

**Short version.** The gradients are *exact* (not finite differences) and
come from **forward-mode AD written into the filter itself**:

| where | precision | how the derivative code is produced | status |
|---|---|---|---|
| CPU (`package:hgf`, `batch/batch_tangents.dart`) | f64 | written by hand: every update step has a twin that also propagates tangents | default since 2026-11-06 |
| GPU (`package:hgf_gpu`, `ir.dart` + `trial.dart`) | f32 | generated: a small expression IR with derivative rules, printed as WGSL | default on the GPU since 2026-11-09 |
| GPU, reverse mode (`fit/grad_kernel.dart`) | f32 | generated from the same IR: adjoint statements in reverse order | spike, 2026-11-11 (below) |

Finite differences remain only where they are the right tool: the Laplace
Hessian (second differences of values), and as an independent oracle in
the tests.

---

## 1. What is being differentiated

An HGF filter is a fixed recurrence. For each trial `t` it updates a state
vector `s` (the beliefs of every node) from the observation `u_t`, and a
response model scores the trial:

```
s_{t+1} = F(s_t, u_t; θ)          (prediction, prediction errors, posteriors)
L(θ)    = Σ_t  ℓ(s_t, r_t; θ)     (negative log likelihood of the responses)
```

`θ` covers the initial beliefs (μ₀, π₀), the tonic volatilities (ω),
drifts, autoconnections, coupling strengths (α, κ) and the response
parameters (e.g. the inverse temperature β). In the engines these are all
*slots* of the state, so "a parameter" and "an initial value of the state"
are the same thing to the differentiation code.

The fitting code works on an unconstrained scale, `x = T(y)`. Both engines
seed each parameter's tangent with `dx/dy`, so they return `∂L/∂y`
directly. The prior and the Jacobian terms are then added in f64 on the
host (`HgfModel.completeNegLogPosterior`).

## 2. Why no AD package

- **There isn't one to use.** Dart has no mature AD library. Its macro
  system, which could have done source transformation, was cancelled in
  2025.
- **Operator overloading is too slow here.** A `Dual` class with
  overloaded `+ − × ÷` works in Dart, but every intermediate becomes a heap
  object. The filter's inner loop is a few hundred floating-point
  operations per trial and element, at a few ns each. Boxing every one of
  them costs far more than the arithmetic.
- **The program is fixed and small.** The update equations are a closed
  set (a handful of step types, about 17 elementary functions). Their
  derivatives can be written, or generated, once. The engine then keeps
  its batched, cache-friendly layout for the tangents too.

What JAX does, and what hgf.dart does instead:

| JAX | hgf.dart, CPU | hgf.dart, GPU |
|---|---|---|
| trace Python into a jaxpr | no trace: the step functions are known | Dart operators on `Ex` objects build expression trees (the "trace") |
| apply the `jvp` / `vjp` transformation | tangent code written next to each step | `Tangents.d` (JVP) or `Adjoints` (VJP) rewrite the trees using a rule table |
| XLA compiles the result | the Dart AOT compiler | the trees are printed as WGSL; the driver compiles them |
| general Python, `lax.cond`, `scan` | — | straight-line code, `select` for branches, one loop over trials |

The GPU path is therefore a small, domain-specific JAX: trace, transform,
compile, for exactly the operations an HGF uses. The CPU path is what that
pipeline would emit, written out by hand and tuned.

## 3. Forward mode (JVP) in one paragraph

A tangent `ẋ` travels alongside every value `x`, where `ẋ = ∂x/∂θ · v` for
a direction `v`. Every operation updates both: `z = a·b` gives
`ż = ȧ·b + a·ḃ`, and `z = exp(a)` gives `ż = z·ȧ`. Seed `v = e_i` (one
parameter) and run the filter once. The tangent of `L` is `∂L/∂θ_i`. With
`k` parameters you carry `k` tangents (a `k`-column Jacobian-vector
product) and get the whole gradient in one pass. The cost grows with `k`,
which is fine for HGF models: `k` is typically 2–10.

## 4. CPU: hand-written tangents (`package:hgf`)

`BatchNetwork(tangents: k)` runs `k` directions for a whole batch of
parameter sets. Each batched update kernel `_bX` has a tangent twin `_tX`:

- the twin computes **exactly** the primal values of `_bX`
  (bit-identical, tested) **plus** the tangents;
- it works stage-wise. First the per-element partial derivatives go into
  coefficient arrays, once per element. Then a tight loop per direction
  multiplies them into the tangents. Tangents are laid out
  `[node][direction][element]`, so the loops vectorise.

The binary prediction step shows the pattern (shortened from
`_tPredictionBinary`). The primal is `μ̂ = clamp(σ(x))` and `π̂ = μ̂(1 − μ̂)`,
where `x` is the sum of the parents' expected means:

```dart
for (var j = 0; j < m; j++) {               // per element: primal + partials
  final raw = sigmoid(x[j]);
  var p = raw; var clipped = false;
  if (p < v) { p = v; clipped = true; } else if (p > 1.0 - v) { p = 1.0 - v; clipped = true; }
  expectedMean[base + j] = p;
  expectedPrecision[base + j] = p * (1.0 - p);
  final slope = clipped ? 0.0 : raw * (1.0 - raw);   // dμ̂/dx
  c0[j] = slope;
  c1[j] = (1.0 - 2.0 * p) * slope;                    // dπ̂/dx
}
for (var ad = 0; ad < _nActive; ad++) {     // per direction: tangents
  ...
  for (var j = 0; j < m; j++) {
    dExpectedMean[ti + j]      = c0[j] * dx[so + j];
    dExpectedPrecision[ti + j] = c1[j] * dx[so + j];
  }
}
```

Other details:
- **Response models** report partial derivatives `∂ℓ/∂(node field)`
  (`batchSurprisePartials`). The engine contracts them with the tangents
  it already has. Asking each response for full tangents made
  `TotalGaussianSurprise` cost more than the filter.
- **Directions nothing moves are skipped.** A response parameter such as β
  costs nothing inside the filter.
- **Cost** (AOT, per element and trial): the first direction costs about
  one more primal (the `exp`s are shared), and each further one about half
  to two-thirds of a primal. With two directions on the binary 2-level
  model that is 155 ns, against 195 ns for forward finite differences.

## 5. The derivative conventions (shared by CPU and GPU)

These are the places where "the derivative" has to be decided rather than
just written down. The rule throughout is to differentiate **the function
as computed**, not an idealised one:

- **Branches** pass the derivative of the side taken. For example, eHGF's
  floored precision has zero derivative where the floor is active.
- **Active clamps pass nothing.** `clamp(x, lo, hi)` has derivative 1
  strictly inside and 0 at the bounds (the `slope = clipped ? 0 : …`
  above).
- **Lambert W** (the unbounded update) is evaluated by Newton iterations,
  but it is differentiated *implicitly*: `W′(z) = W / (z (1 + W))`. AD
  through the iterations would also work, but it would be slower and less
  accurate.
- **Quotients** are formed as `(ȧ − (a/b)·ḃ) / b`, never through `1/b²`.
  Variances in runaway regimes reach 1e300, where `1/V²` underflows to 0.
  That gave wrong tangents on two fixtures before it was fixed. For
  the same reason the expected precision's tangent is
  `π̂̇ = −π̂ (π̂ V̇)`, not `−V̇/V²`.
- **Non-linear value couplings** need `g‴` for the tangent of the `g″`
  term in the precision update. Built-in couplings supply it. A
  `CustomCoupling` supplies `g, g′, g″` and optionally `g‴` (otherwise
  `g‴` comes from central differences of `g″`, about 1e-10 relative), and
  `CustomCoupling.check()` compares the given derivatives with finite
  differences.

## 6. GPU: generated tangents (`package:hgf_gpu`)

WGSL has no AD, and the GPU kernels are *generated* per network anyway (one
GPU thread filters one element, its whole state in registers). So the
generator differentiates while it generates:

1. **Expression IR** (`ir.dart`). The generator writes the update equations
   with Dart operators on `Ex` objects (`a * b`, `exp(x)`, `sel(f, t, c)`),
   which build trees instead of computing numbers. This is the tracing
   step. Control flow becomes `select`, so each trial is straight-line
   code.
2. **JVP rules** (`Tangents.d`). For `let x = e`, the generator also emits
   the tangent of `e`: the sum rule, product rule, the quotient rule above,
   `select` passing the tangent of the side taken, and a table of partials
   for each function (`exp`, `log`, `sigm`, `tanh`, `clamp`, `max`,
   `lambertW0`, `logAddExp`, …).
3. **Activity analysis.** A fixed point works out which slots carry
   tangents. It starts from the parameters and adds every slot an active
   value is written to, recompiling until nothing changes. Everything else
   stays scalar.
4. **Vectorised directions.** Tangents are `vec4<f32>` chunks, so 5
   directions are one `vec4` plus one `f32`.

The binary prediction from §4, as generated with 2 directions:

```wgsl
let pr = clamp(sigm(em1), 0.000001f, (1.0f - 0.000001f));
let pr_t0 = (select(0.0f, 1.0f, ((sigm(em1) > 0.000001f) && (sigm(em1) < (1.0f - 0.000001f))))
             * (dsigm(em1) * em1_t0));
```

The clamp rule (zero derivative when active) and the sigmoid rule
(`dsigm`) appear directly. `em1_t0` is a `vec2<f32>` holding both
directions.

## 7. Reverse mode (VJP): the 2026-11-11 spike

Forward mode costs about `1 + c·k` primal evaluations, with `c` ≈ 0.5–0.7
on the CPU. Reverse mode costs a fixed multiple of the primal whatever `k`
is, but it must remember (or recompute) the forward values. The spike
built it on the GPU from the same IR and measured where it pays.

### How it works

1. **SSA trial** (`TrialCompiler.compileSsa`). The same generator runs in a
   second mode. It emits one flat block in which every local and every
   write to a slot is a fresh `let` (`em0_v17`, `ep0_v18`, …), and it keeps
   the statements as trees: `[(name, expression)]`.
2. **Adjoint emission** (`Adjoints` in `ir.dart`). Walk the statements in
   reverse order. For `let x = e`, push `x_b` (the adjoint `∂L/∂x`) into
   the adjoints of the variables `e` reads, using **the same partials
   table** as forward mode. `select` routes the adjoint to the branch
   taken, active clamps pass nothing, and the quotient uses `−(a/b)/b`.
   Generated for the binary prediction:

   ```wgsl
   pr_v15_b += (ep0_v18_b * qr_v16);         // ep0 = pr · qr
   qr_v16_b += (ep0_v18_b * pr_v15);
   pr_v15_b += em0_v17_b;                    // em0 = pr
   em1_v11_b += ((pr_v15_b * select(0.0f, 1.0f, (sigm(em1_v11) > 0.000001f) && …)) * dsigm(em1_v11));
   ```
3. **Across trials.** One GPU thread runs one element's trial loop. The
   backward sweep needs each trial's starting state, which comes from one
   of two kinds of tape:
   - **full tape:** the carried state of every trial, `[t][slot][element]`
     (coalesced). Memory is `T × slots × B × 4` bytes;
   - **√T checkpoints:** the state every `C = ⌈√T⌉` trials. On the way
     back, each segment is recomputed into a small segment tape. This costs
     one extra forward pass and needs `(T/C + C) × slots × B × 4` bytes.

   Each reverse step reloads that trial's state, recomputes the trial's
   primal from it, then runs the adjoint statements. Parameters never
   written in a trial (ω, β, couplings) simply accumulate their adjoint
   across trials. A written slot hands its adjoint from the end of trial
   `t` to its start.

### Correctness

- The new tests (`test/native_reverse_test.dart`) compare the reverse
  gradient with the forward tangents on four models, for every element and
  parameter: binary 2-level with softmax decisions, continuous 3-level
  standard and eHGF, and an 8-input binary network with 33 parameters.
  They agree to 2e-4 relative over 200 trials.
- **Against f64** (`benchmark/bin/reverse_f64.dart`, 65,536 elements, 320
  trials): for typical elements, forward and reverse both match the f64
  CPU gradient to about 1e-6. For the elements where the two f32 gradients
  disagree most, both are off by 1e-4 to 3e-2. These elements have
  |∂L/∂θ| of about 1e4, and neither method is consistently closer
  (forward was closer on 21/23/22 of 40 elements per model, reverse on
  19/16/17). The disagreements are f32 conditioning, not a defect of
  either mode.
- The kernels the library already generates are byte-identical to before
  (the SSA mode is additive).

### Cost

RX 7600 XT via RADV with wgpu-native, 320 trials, best of 15. Times are
multiples of the primal (no gradient) at the same batch size `B`:

| model (parameters, carried slots) | B | fwd k=2 | k=5 | k=10 | k=20 | k=all | rev, tape | rev, √T ckpt |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| binary 2-level (6, 7) | 2,000 | 1.4 | 1.5 | 1.9 | 2.7 | 1.6 | 2.8 | 2.7 |
| | 16,384 | 1.5 | 2.2 | 3.2 | 5.4 | 2.4 | 4.6 | 2.9 |
| | 65,536 | 1.7 | 2.3 | 2.5 | 3.5 | 1.5 | 6.7 | 2.7 |
| continuous 3-level (10, 17) | 2,000 | 1.2 | 1.4 | 2.1 | 3.1 | 2.1 | 2.4 | 3.0 |
| | 16,384 | 1.7 | 2.7 | 4.3 | 7.8 | 4.3 | 8.6 | 3.3 |
| | 65,536 | 1.8 | 2.4 | 3.2 | 6.3 | 3.2 | 16.4 | 6.1 |
| continuous 3-level eHGF (10, 17) | 16,384 | 2.0 | 3.1 | 5.2 | 10.1 | 5.2 | 8.8 | 3.5 |
| | 65,536 | 1.7 | 2.3 | 3.5 | 7.6 | 3.5 | 13.1 | 5.4 |
| 8 binary inputs (33, 61) | 2,000 | 2.9 | 7.7 | 56 | 251 | 687 | 4.0 | 5.5 |
| | 16,384 | 2.1 | 7.2 | 57 | 268 | 968 | 9.2 | 4.7 |
| | 65,536 | 2.4 | 14 | 105 | — | — | (4.9 GB) | 9.6 |

(k=all: one direction per parameter, which is what fitting uses. "—":
not run, because longer dispatches trigger the driver's GPU-hang reset.
At B = 2,000 every kernel is latency-bound at 0.2–0.7 ms. To rerun:
`dart compile exe benchmark/bin/reverse_gpu.dart -o .dart_tool/reverse_gpu
&& .dart_tool/reverse_gpu` in `packages/hgf_gpu`.)

What the numbers say:

- **Small models (k ≤ 10): forward mode stays.** Checkpointed reverse costs
  about 2.7–6× the primal, which is what forward costs at k ≈ 5–10. The
  models fitted today have k = 2–6.
- **Large models: reverse mode is necessary on the GPU.** On the
  33-parameter network, forward mode does not just grow linearly; it falls
  off a cliff, from 7–14× the primal at k = 5 to 56–105× at k = 10 and ~700–970×
  at k = 33. Each thread carries `61 slots × (1 + k)` floats, far beyond
  the register file, and the jump fits register spilling. Checkpointed
  reverse mode costs 4.7–9.6× the primal. At B = 16,384 that is 6.4 ms
  against 1.3 s, about 200× faster. Splitting forward mode into
  dispatches of ≤ 4 directions would avoid the spills, but it would still
  cost roughly k/4 passes (an estimated ~50× the primal at k = 33).
- **Checkpointing beats the full tape** once `B` is large. Recomputing a
  segment is cheaper than streaming the full tape through memory (1.4 GB
  for the 3-level model at B = 65,536). It also stays within WebGPU's
  default buffer limits (128 MB per binding in browsers), which the full
  tape does not.

### Not done yet

- The **unbounded update** (pyhgf's default) has no SSA form yet. Its
  mutable accumulators and `if` blocks need rewriting as `select`s first,
  as was done for forward mode.
- Reverse mode is not wired into `GpuFitter`, and it runs each series in
  one dispatch. Browsers need long series split across dispatches, as
  `FitKernel` already does.
- **No reverse mode on the CPU.** Dart AOT cannot load generated code at
  runtime, so a CPU reverse mode would mean hand-written adjoints for
  every batched kernel, or generating Dart from the IR at build time. The
  CPU's f64 forward mode does not have the GPU's register problem.
- Second order: an exact Laplace Hessian could come from forward-over-
  reverse (`k` reverse sweeps of a tangent kernel). Today it uses second
  differences of values in f64, which is accurate and cheap for small `k`.

## 8. Where finite differences are still used

- **Laplace Hessian:** central second differences of function values on
  the unconstrained scale, `2k² + 1` evaluations per subject, all subjects
  in one pass, in f64 (`Pass.precise` keeps these on the CPU even when the
  GPU fits).
- **Tests:** Richardson-extrapolated central differences are the oracle
  for the tangents.
- **History:** before 2026-11-06, gradients were central differences in the
  same batched pass (`2k + 1` elements per gradient). Exact gradients took
  MAP from 29 to 13.7 ms (20 agents) and NUTS from 64 to 43.7 s, with
  higher ESS/s.

## 9. How the gradients are validated

- **CPU tangents against finite differences:** every tangent field of
  every node, plus the total surprise, against Richardson central
  differences on the 37 batchable pyhgf fixtures. That is 6 directions
  each (ω, drift, μ₀, π₀, α, κ), on the Dart VM, dart2js and dart2wasm.
  The same check covers the gradient of every built-in response model and
  `hgf_fit`'s objective gradient (all priors and transforms, with and
  without Jacobian, missing data, time steps, response parameters).
  - Two fixtures need care. `continuous_2level_eHGF` turns chaotic after
    about 150 trials, where finite differences disagree even with each
    other, so it is checked on its first 120 trials. The binary 3-level
    fixtures sit in a runaway regime (tangents around 1e8), where finite
    differences only converge onto the tangents below h = 1e-5.
- **The primal itself** matches pyhgf's outputs to 1e-9 on 61 fixtures,
  so the function being differentiated is pyhgf's.
- **GPU tangents against the f64 CPU engine:** values within 3.6e-7
  relative and gradients within 1.1e-5 of the gradient's scale (mostly
  1e-6), on six models, on native Vulkan and on Chromium's
  WebGPU/SwiftShader.
- **GPU reverse mode** against GPU forward mode and the f64 CPU engine
  (§7).
- **Custom coupling fixtures** (sin, softplus) were generated with JAX
  autodiff.

## 10. Files

| | |
|---|---|
| `packages/hgf/lib/src/batch/batch_tangents.dart` | CPU forward-mode kernels (`_t*`) |
| `packages/hgf/lib/src/coupling.dart` | coupling functions and their derivatives up to `g‴` |
| `packages/hgf_fit/lib/src/model.dart` | seeding with `dx/dy`, prior and Jacobian terms |
| `packages/hgf_gpu/lib/src/ir.dart` | expression IR, `Tangents` (JVP), `Adjoints` (VJP), the rule table |
| `packages/hgf_gpu/lib/src/trial.dart` | trial code generation (`compile`, `compileSsa`) |
| `packages/hgf_gpu/lib/src/fit/fit_kernel.dart` | production GPU likelihood + gradient kernel (forward mode) |
| `packages/hgf_gpu/lib/src/fit/grad_kernel.dart` | reverse-mode spike kernels (forward/reverse harness) |
| `packages/hgf_gpu/benchmark/bin/reverse_gpu.dart`, `reverse_f64.dart` | the spike's benchmarks |
