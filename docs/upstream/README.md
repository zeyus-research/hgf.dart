# Upstream issues

Problems found in pyhgf (JAX and rshgf) and in dart2wasm while porting and
optimising hgf.dart, each with a repro. None is reported upstream yet; tick
**Reported** with the issue link when it is.

Run the pyhgf/rshgf repros from `tool/parity`, whose uv environment builds
pyhgf and rshgf from `../pyhgf`:

```sh
cd tool/parity && uv run python ../../docs/upstream/pyhgf/<script>.py
```

Run the dart2wasm repros with `docs/upstream/dart2wasm/run.sh` (needs `dart`
and `node`; `taskset -c 2 ./run.sh` gives steadier timings).

Checked against pyhgf `6be12eb4` (0.3.2) and Dart 3.13.7.

| id | where | issue | severity | reported |
|---|---|---|---|---|
| [P1](#p1) | pyhgf JAX + rshgf | relaxed eHGF diverges (missing MGF terms) | **high**: NaN | [ ] |
| [P2](#p2) | pyhgf JAX + rshgf | `lambert_w0` NaN for z ≳ 2.7e151 | medium | [ ] |
| [P3](#p3) | pyhgf JAX | update-sequence move loop mutates the list it iterates | low | [ ] |
| [P4](#p4) | pyhgf JAX | a parent of two ef-state nodes crashes the scheduler | medium: crash | [ ] |
| [P5](#p5) | pyhgf JAX | `add_ef_state` ignores `xis` | low | [ ] |
| [P6](#p6) | pyhgf JAX | ν = ±∞ on missing observations (HGF learning) | low | [ ] |
| [P7](#p7) | pyhgf JAX | stale `update_fn` in the posterior pass (not reproduced) | unknown | [ ] |
| [P8](#p8) | pyhgf JAX vs rshgf | backends order independent predictions differently | cosmetic | [ ] |
| [P9](#p9) | pyhgf | licence: MIT in `LICENSE`, GPL-3.0 in `pyproject.toml` | to settle | [ ] |
| [R1](#r1) | rshgf | ef-state ignores `observed`; univariate fixed rate only | medium | [ ] |
| [R2](#r2) | rshgf | hot-path costs (string compares, allocations, clones) | perf | [ ] |
| [D1](#d1) | dart2wasm | a typed-data view anywhere slows reads of non-view lists | perf, ~10% | [ ] |
| [D2](#d2) | dart2wasm | `1 / 6` (int literals) isn't constant-folded | perf | [ ] |
| [D3](#d3) | dart2wasm | top-level `final` tables: null check + initialiser at each use | perf, ~1.5% | [ ] |
| [D4](#d4) | dart2wasm | `Float32x4`/`Float64x2` emulated, 10–15× slower than scalar | perf | [ ] |
| [D5](#d5) | dart2wasm | `dart:math` exp/log are JS imports | perf | [ ] |
| [D6](#d6) | dart2wasm | no way to drop `array.get` bounds checks | perf | [ ] |

## pyhgf (JAX backend, and rshgf where noted)

### P1

**Relaxed eHGF diverges on 3-level continuous HGFs** (JAX and rshgf).
Repro: [`pyhgf/ehgf_relaxed_divergence.py`](pyhgf/ehgf_relaxed_divergence.py).

```
JAX eHGF, mean_field=False         first NaN: 44    x2 gains precision on 28/44 finite trials
JAX eHGF, mean_field=True          first NaN: None  x2 gains precision on 569/614 finite trials
rshgf eHGF, relaxed                first NaN: 44    x2 gains precision on 28/44 finite trials
JAX eHGF, relaxed + MGF correction first NaN: None  x2 gains precision on 586/614 finite trials
```

The eHGF precision update re-predicts the volatility child's volatility as
`t·exp(κ μ_b + ω̃)`, with `ω̃ = ω + Σ_{p≠b} κ_p μ̂_p`. The relaxed prediction
of that child used `Ω = t·exp(ω + Σ_p κ_p μ̂_p + Σ_p κ_p²/(2 π̂_p))`. Without
the `κ²/(2π̂)` terms, the re-prediction at `μ_b = μ̂_b` is below the child's
predicted precision, contrary to the docstring. The increment is then floored
at 0 on most trials (~95%), the volatility parent never gains precision, and
the network goes NaN (trial 44 on the USD/CHF example).

**Fix:** add `Σ_p κ_p²/(2 π̂_p)` over all of the child's volatility parents to
`ω̃`, in relaxed mode only (mean-field has no such terms). The patch is
`_ehgf_increment_mgf_corrected` in
[`tool/parity/gen_fixtures.py`](../../tool/parity/gen_fixtures.py), in the
shape of an upstream PR; rshgf needs the same change.
**hgf.dart:** opt-in `Network.ehgfMgfCorrection` (off by default, for
parity), checked against the patched JAX (`*_mgf_correction` fixtures); demo
`packages/hgf/example/ehgf_mgf_correction.dart`. If upstream adopts the
fix, flip the default and regenerate the fixtures.

### P2

**`lambert_w0` returns NaN for z ≳ 2.7e151** (`pyhgf/math.py:451`; rshgf
the same).
Repro: [`pyhgf/lambert_w0_overflow.py`](pyhgf/lambert_w0_overflow.py).

```
z = 2.0e+151   lambert_w0 = 342.5470870938422      residual 5.7e-14
z = 1.0e+152   lambert_w0 = nan                    residual nan
NaN from z ≈ 2.719e+151
```

Each Halley step divides by `2 f1² − f f2` with `f1 = (w + 1) eʷ`; `f1²`
overflows, and the step becomes `inf/inf`. The unbounded update then drops
the expansion that needed it.

**Fix:** divide `f`, `f1`, `f2` by `eʷ` first (`w − z e⁻ʷ`, `w + 1`, `w + 2`):
the step is unchanged and nothing overflows. Starting from
`log z − log log z` for large z also saves iterations.
**hgf.dart:** `lambertW0` keeps the same behaviour (NaN from ~2.7e151) for
parity.

### P3

**The update-sequence move loop mutates the list it iterates over**
(`pyhgf/utils/get_update_sequence.py:308`).
Repro: [`pyhgf/update_sequence_move_loop.py`](pyhgf/update_sequence_move_loop.py).

```
2 ef nodes: deferred posteriors run for nodes [1, 0]
3 ef nodes: deferred posteriors run for nodes [1, 0, 2]
loop as written: ['a', 'b', 'c', 'D1', 'D0', 'D2']
stable partition: ['a', 'b', 'c', 'D0', 'D1', 'D2']
```

`for step in update_sequence: ... remove(step); append(step)` skips the
element after each moved step and moves some steps twice, so the deferred
steps (ef-state posteriors, `categorical_state_update`) end up in a scrambled
order. For ef nodes this changes no values (their posteriors are
independent); it would matter if two deferred steps ever depend on each
other.

**Fix:** a stable partition:
`seq = [s for s in seq if not deferred(s)] + [s for s in seq if deferred(s)]`.
**hgf.dart:** replays the loop literally (`_moveDeferredSteps` in
`packages/hgf/lib/src/update_sequence.dart`) to keep the same schedule;
switch to the partition if upstream fixes it.

### P4

**A continuous node that is the value parent of two ef-state nodes crashes
`get_update_sequence`** (`get_update_sequence.py:94`).
Repro: [`pyhgf/ef_two_ef_children.py`](pyhgf/ef_two_ef_children.py)
→ `ValueError: list.remove(x): x not in list`.

The loop calls `nodes_without_posterior_update.remove(i)` once per ef child.
**Fix:** remove once (`break` after the first ef child, or filter with
`any(...)`). **hgf.dart:** builds the schedule with `any`, so it doesn't
crash, but there is no pyhgf reference for this structure.

### P5

**`add_ef_state` ignores a user `xis`** (`pyhgf/model/add_nodes.py:312`).
Repro: [`pyhgf/ef_xis_ignored.py`](pyhgf/ef_xis_ignored.py)
→ `asked for xis = [1, 2], got [0. 1.]`.

The defaults are written over the merged parameters. **Fix:** compute the
default `xis` (N(0, 1) or N(0, I) for the dimension) only when the user gave
none. **hgf.dart:** `addEfState(xis: ...)` sets the prior (documented as a
difference).

### P6

**With HGF learning, a missing observation records ν = ±∞**
(`pyhgf/updates/posterior/exponential.py`).
Repro: [`pyhgf/ef_missing_nus_inf.py`](pyhgf/ef_missing_nus_inf.py).

```
observed: [1, 1, 1, 0, 1, 1, 1, 1]
nus:      [1.988, 2.279, 2.518, -inf, 2.626, 2.801, 2.943, 3.058]
```

ν = mean(δ/Δ) is computed before `observed` is checked; on a missing trial
the tracking node's parent doesn't move (Δ = 0). `xis` is protected by the
`jnp.where`, so only the recorded `nus` (and `to_pandas`) shows it.
**Fix:** `nus = jnp.where(observed, new_nus, old_nus)`. **hgf.dart:** same
as pyhgf (the `ef_normal_hgf2_missing` fixture has the -inf values).

### P7

**Stale `update_fn` in the posterior pass** (from reading
`get_update_sequence.py:145`, not reproduced). `update_fn = None` is set once
per pass, not per node, and the posterior branch only handles node types 2
and 4. A node of another type with children (an ef-state node that is a
value parent, as dp-state networks use) gets either no posterior step or the
previous node's update function, depending on what ran earlier in the same
pass. **Fix:** reset `update_fn` per node and handle every node type
explicitly. Check again when porting dp-state.

### P8

**JAX and rshgf schedule independent predictions in different orders.**
Repro: [`pyhgf/prediction_order_backends.py`](pyhgf/prediction_order_backends.py)
→ `JAX: [4, 2, 3, 0, 1]`, `rshgf: [4, 2, 0, 3, 1]`.

JAX takes all ready nodes in one pass; rshgf takes the first ready node and
restarts. Values are equal (the nodes are independent), but the schedules
can't be compared step by step. **hgf.dart:** follows rshgf; its parity
tests compare predictions to JAX as a set, and the update steps in order.

### P9

**Licence mismatch:** `LICENSE` is MIT, `pyproject.toml` says
`license = "GPL-3.0"`. hgf.dart is MIT, as in liblsl.dart. Worth settling
before hgf.dart is published.

## rshgf

P1 and P2 apply to rshgf too.

### R1

**ef-state ignores `observed`, and is univariate with a fixed rate only**
(`src/updates/nodalised/prediction_error/exponential.rs`).
Repro: [`pyhgf/rshgf_ef_ignores_observed.py`](pyhgf/rshgf_ef_ignores_observed.py).

```
observed: [1, 0, 1]
rshgf xis: [[0.25, 1.0], [0.6875, 1.75], [1.265625, 3.5625]]
JAX   xis: [[0.25, 1.0], [0.25, 1.0], [0.9375, 3.0]]
```

**Fix:** skip the update when `observed == 0`; add `dimension` (multivariate
normal) and the `hgf-1/2/3` learning modes as in JAX. **hgf.dart:** follows
JAX.

### R2

**Hot-path costs** (performance suggestions, no wrong values):

- `relaxed_factors` compares `node_type` strings per call
  (`src/updates/nodalised/posterior/continuous.rs:34`);
- the unbounded update allocates three `Vec`s per call
  (`continuous.rs:478-482`);
- recording clones each node's vectors per trial
  (`src/model/network.rs:225-241`).

hgf.dart avoids all three (int kind codes, preallocated scratch, a staged
recorder) and runs single series ~1.9× faster than rshgf on AOT
(`docs/benchmark_results.md`); not all of that gap comes from these three.

## dart2wasm (Dart SDK)

### D1

**A typed-data view anywhere in the program slows reads of lists that are
never views.** Repro:
[`dart2wasm/typed_data_views.dart`](dart2wasm/typed_data_views.dart), via
`run.sh`.

```
-- view in program: false; kernel reads _offsetInElements 0 time(s)
-- view in program: true; kernel reads _offsetInElements 2 time(s)
view in program: false  13.39 ns/call (median of 7)
view in program: true  15.00 ns/call (median of 7)
```

`Engine`'s fields only ever hold `Float64List(n)`, but once an unrelated
function receives a `Float64List.sublistView`, `Engine.kernel` also loads
`_offsetInElements` for its lists. The effect needs the lists in (mutable)
fields: passed straight as arguments, they keep their exact type. On the
hgf.dart engine, removing every view gained 8–10%; that isn't possible in a
real app (views come from `sublistView`, `buffer.as*List` and packages).

### D2

**Int-literal division isn't constant-folded:** `x * (1 / 6)` calls
`BoxedInt.toDouble` at run time; `1.0 / 6.0` becomes `f64.const
0.16666666666666666`. Repro:
[`dart2wasm/codegen.dart`](dart2wasm/codegen.dart), via `run.sh`.
**hgf.dart:** writes float coefficients as double literals.

### D3

**Top-level `final` tables are null-checked at every use, with the
initialiser inlined there:** one `table[i & 3]` read compiles to ~130 lines
(`br_on_non_null` on the global, then the whole `Float64List.fromList`). Same
repro. In hgf.dart the exp tables (`_pow2`, `_exp2Frac`) cost ~1.5% of the
eHGF profile. A side-effect-free initialiser could run eagerly.

### D4

**`Float32x4` / `Float64x2` are emulated**: 10–15× slower than scalar code,
while on AOT `Float32x4` mul-add is 2.5× faster than scalar. Measured with
`packages/hgf/benchmark/bin/runtime_probe.dart`. hgf.dart keeps SIMD to AOT
until dart2wasm lowers these to Wasm SIMD (`v128`).

### D5

**`dart:math` exp/log are JS imports:** 11 ns per exp on Wasm vs 6 ns on AOT
(25 vs 10 ns when each call waits on the previous one), measured with
`runtime_probe.dart`. A table-driven exp in Dart (`tableExp`, 1 ulp) costs
8 ns (17 chained), so hgf.dart uses it on dart2wasm. Wasm-native
implementations in the SDK would help every numeric app.

### D6

**No way to drop bounds checks on `array.get`:** Wasm GC arrays are always
bounds-checked, and dart2wasm has no counterpart of
`vm:unsafe:no-bounds-checks` (35–45% on the VM's standard/eHGF updates).
Partly inherent to Wasm GC; a feature request at most.
