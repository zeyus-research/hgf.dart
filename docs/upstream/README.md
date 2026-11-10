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

Checked against pyhgf `6be12eb4` (0.3.2) and Dart 3.13.7; `hgf_deep` (P18,
P19) against pyhgf `1e6e98d7` (after #475–#477).

| id | where | issue | severity | reported |
|---|---|---|---|---|
| [P1](#p1) | pyhgf JAX + rshgf | relaxed eHGF diverges (missing MGF terms) | **high**: NaN | [ ] |
| [P2](#p2) | pyhgf JAX + rshgf | `lambert_w0` NaN for z ≳ 2.7e151 | medium | [ ] |
| [P3](#p3) | pyhgf JAX | update-sequence move loop mutates the list it iterates | low | [ ] |
| [P4](#p4) | pyhgf JAX | a parent of two ef-state nodes crashes the scheduler | medium: crash | [ ] |
| [P5](#p5) | pyhgf JAX | `add_ef_state` ignores `xis` | low | [ ] |
| [P6](#p6) | pyhgf JAX | ν = ±∞ on missing observations (HGF learning) | low | [ ] |
| [P7](#p7) | pyhgf JAX | stale `update_fn` in the posterior pass: a categorical or dp-state node after a continuous one crashes | medium: crash | [ ] |
| [P8](#p8) | pyhgf JAX vs rshgf | backends order independent predictions differently | cosmetic | [ ] |
| [P9](#p9) | pyhgf | licence: MIT in `LICENSE`, GPL-3.0 in `pyproject.toml` | to settle | [ ] |
| [P10](#p10) | pyhgf JAX | `categorical-state` with `n_nodes > 1` builds a broken graph | medium | [ ] |
| [P11](#p11) | pyhgf JAX | `add_categorical_state` ignores `binary_parameters`; docstring says `3n + 2` nodes | low | [ ] |
| [P12](#p12) | pyhgf JAX | categorical α = ±∞ on missing observations | low | [ ] |
| [P13](#p13) | pyhgf JAX | `dirichlet_kullback_leibler` NaN once Σα > 171 (Γ overflows) | low | [ ] |
| [P14](#p14) | pyhgf JAX | dp-state ignores `observed`: missing samples are clustered (NaN opens a NaN cluster) | medium | [ ] |
| [P15](#p15) | pyhgf JAX | dp-state results depend on JAX's PRNG mode and precision; the tests pin the legacy mode | low | [ ] |
| [P16](#p16) | pyhgf JAX | dp-state `batch_size` ≠ number of clusters fails inside the scan | low | [ ] |
| [P17](#p17) | pyhgf JAX + rshgf | unbounded volatility update (the default) runs away on 3-level binary HGFs | **high**: runaway beliefs | [ ] |
| [P18](#p18) | pyhgf JAX + rshgf | `DeepNetwork`'s volatility levels run away (all three updates), then NaN: the value posterior's precision floor lets the curvature term cancel the evidence | **high**: NaN | [ ] |
| [P19](#p19) | pyhgf JAX | `DeepNetwork`: exp overflow and a cancelling reconstruction turn a runaway volatility level into NaN; an improper 0/0 posterior after it | medium: NaN | [ ] |
| [R1](#r1) | rshgf | ef-state ignores `observed`; univariate fixed rate only | medium | [ ] |
| [R2](#r2) | rshgf | hot-path costs (string compares, allocations, clones) | perf | [ ] |
| [R3](#r3) | rshgf | GELU coupling uses a 1.5e-7 `erfc` approximation (called exact) | low | [ ] |
| [R4](#r4) | rshgf | `DeepNetwork.batch_update` kernel: stale g″ weighting (#456), ignores `predict_precision` | **high**: wrong values | [ ] |
| [R5](#r5) | rshgf | `DeepNetwork(predict_precision=False)` panics with volatility parents | medium: crash | [ ] |
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

**Stale `update_fn` in the posterior pass: a categorical node added after a
continuous one crashes** (`get_update_sequence.py:145`).
Repro: [`pyhgf/categorical_after_continuous.py`](pyhgf/categorical_after_continuous.py).

```
categorical alone, binary nodes 1-3: [(1, 'binary_state_node_prediction_error'), ...]
after a 2-level HGF, binary nodes 3-5: [(3, 'continuous_node_posterior_update'), (4, ...), (5, ...), (3, 'binary_state_node_prediction_error'), ...]
input_data: KeyError 'value_coupling_children'
```

`update_fn = None` is set once per pass, not per node, and the posterior
branch only assigns it for node types 2 and 4. A categorical node's binary
nodes have a child, so they enter the posterior pass; when a continuous node
was handled earlier in the same pass (the parent of a continuous input added
before the categorical node), they inherit its
`continuous_node_posterior_update`, which fails on the binary node's
attributes. With the categorical node added first the pass reaches the
binary nodes before any continuous one, and it works. The same would happen
to any other node type with children: the ef-state clusters of a dp-state
node do (repro: [`pyhgf/dp_after_continuous.py`](pyhgf/dp_after_continuous.py),
`KeyError 'expected_precision'`).
**Fix:** reset `update_fn` per node and handle every node type explicitly.
**hgf.dart:** only continuous nodes get posterior steps, so either order
works; the `categorical_mixed` and `dp_mixed` fixtures use the order pyhgf
can run, and `categorical_test.dart` / `dp_state_test.dart` check that a
continuous HGF before the node leaves its trajectories bit-identical.

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

### P10

**`add_nodes(kind="categorical-state", n_nodes=2)` builds a broken graph**
(`pyhgf/model/add_nodes.py:408`).
Repro: [`pyhgf/categorical_n_nodes.py`](pyhgf/categorical_n_nodes.py): of
16 nodes only node 0 is categorical; its binary parents 1-3 have no value
parents, and node 4 (meant to be the second categorical node) is a binary
node.

`binary_idxs` is computed once, from the node count before the categorical
nodes are inserted, and every categorical node gets the same list.
**Fix:** compute the indexes per categorical node (or reject
`n_nodes > 1`). **hgf.dart:** `addCategoricalState` adds one node per call,
and any number of calls work; `NetworkSpec` rejects `n_nodes > 1`.

### P11

**`add_categorical_state` ignores `binary_parameters`**
(`pyhgf/model/add_nodes.py:397`).
Repro: [`pyhgf/categorical_binary_parameters.py`](pyhgf/categorical_binary_parameters.py)
→ `asked for tonic_volatility_2 = -2, got [-4.0, -4.0, -4.0]`.

The implied binary HGFs are built from a local dict of defaults; the user's
dict is only stored on the categorical node, although the `add_nodes`
docstring says it parametrises them. The same docstring gives the node count
as `3n + 2`; it is `2n + 2`. **Fix:** merge the user's dict over the
defaults. **hgf.dart:** `addCategoricalState` takes the parameters, and
`NetworkSpec` reads `binary_parameters` (documented as a difference).

### P12

**A missing observation makes a categorical node's α ±∞**
(`pyhgf/updates/posterior/categorical.py`).
Repro: [`pyhgf/categorical_missing_alpha.py`](pyhgf/categorical_missing_alpha.py).

```
observed:      [1 1 1 0 1 1 1 1]
alpha[3]:      [inf -inf -inf]
kl_divergence: [0.754 0.086 0.102 nan nan 0.1 0.046 0.067]
```

`categorical_state_update` doesn't check `observed`. On a missing trial the
binary HGFs get no prediction error, so Δ = σ(μ_p) − μ̂_b is exactly 0 and
ν = δ/Δ − 1 = ±∞. The KL divergence is NaN on that trial and the next. The
binary HGFs are unaffected, as is the next α (it doesn't depend on the old
one). Like [P6](#p6). **Fix:** keep α, the mean and the surprise terms where
`observed` is 0. **hgf.dart:** same as pyhgf (fixture
`categorical_3_missing`).

### P13

**`dirichlet_kullback_leibler` is NaN once the concentrations sum past
~171** (`pyhgf/math.py:337`).
Repro: [`pyhgf/dirichlet_kl_overflow.py`](pyhgf/dirichlet_kl_overflow.py).

```
sum 180:  pyhgf nan  gammaln 0.04435376604050223
omega2 = -8, precision 100: alpha sums up to 429; 21 of 500 trials past 171.6; KL NaN on 20
```

It takes `log(gamma(Σα₁) / gamma(Σα₂))`, and Γ overflows float64 above
171.6. With pyhgf's default parameters a categorical node's α sums stay well
below that (15-120 in the runs tried), but slowly learning binary HGFs (a
confident prior on their value parents) get there. **Fix:** differences of
`gammaln` (with `gammasgn` if negative α must give NaN as now).
**hgf.dart:** computes it in log space with the sign of Γ, so it matches
pyhgf wherever pyhgf is finite and stays finite here.

### P14

**A dp-state node clusters missing observations**
(`pyhgf/updates/prediction_error/dirichlet.py`).
Repro: [`pyhgf/dp_missing_observed.py`](pyhgf/dp_missing_observed.py)
(trial 5 unobserved):

```
placeholder 0.0: n_total at trials 4-6 [5 6 7], n at trial 5 [6. 0. 0.], final expected_means [-2.58   3.775  0.   ]
placeholder nan: n_total at trials 4-6 [5 6 7], n at trial 5 [5. 1. 0.], final expected_means [-2.934    nan    nan]
```

`dirichlet_node_prediction_error` never reads `observed`. An unobserved
trial still picks or opens a cluster, counts in `n` and `n_total`, and the
chosen cluster learns from the placeholder value. With NaN as the
placeholder, the cluster likelihoods are NaN, a NaN cluster opens, and every
later candidate is NaN too. **Fix:** on `observed == 0`, set the clusters'
`observed` to 0 and leave everything else as it is (the patch in
`tool/parity/gen_fixtures.py`, `_dp_prediction_error_skip_missing`).
**hgf.dart:** skips missing observations that way; the `dp_missing`
fixture comes from pyhgf JAX with that patch.

### P15

**dp-state results depend on JAX's PRNG mode and float precision**
(`get_candidate`, `likely_cluster_proposal`).
Repro: [`pyhgf/dp_prng_mode.py`](pyhgf/dp_prng_mode.py):

```
pyhgf's test_get_candidate expects mean 5.026636, sigma 1.2752448
partitionable=True  x64=True  mean 5.051089  sigma 1.290296
partitionable=True  x64=False mean 5.028831  sigma 1.329850
partitionable=False x64=True  mean 5.024141  sigma 1.271072
partitionable=False x64=False mean 5.026635  sigma 1.275245
```

The new-cluster candidate averages the best of 20000 proposals drawn from
`random.key(42)`. Which numbers that key gives depends on
`jax_threefry_partitionable` (True by default since JAX 0.5) and on
`jax_enable_x64`, so the same network clusters differently across JAX
versions and precisions. pyhgf's `tests/conftest.py` pins the legacy mode,
so `test_get_candidate` checks values (float32, legacy) that users on a
current JAX don't get. The proposals are also redrawn on every trial from
the same key: 40000 threefry hashes and `erf_inv`s, and an argsort of 20000,
for the same numbers each time. **Fix:** draw the proposals once (or from
the network's `rng_key`, if they are meant to be random), and test against
the default PRNG mode. **hgf.dart:** reproduces the default mode's float64
draws (threefry and XLA's `erf_inv`, `lib/src/jax_random.dart`) to about
1 ulp, once per process; the fixtures use the default mode.

### P16

**A dp-state node whose `batch_size` differs from its number of clusters
fails inside the scan** (`add_dp_state`).
Repro: [`pyhgf/dp_batch_size.py`](pyhgf/dp_batch_size.py):

```
batch_size=3, 3 clusters: ran
batch_size=4, 3 clusters: TypeError mul got incompatible shapes for broadcasting: (3,), (4,).
batch_size=None, 3 clusters: TypeError mul got incompatible shapes for broadcasting: (3,), (10,).
```

`batch_size` (default 10) sizes `n` and `activated`, but the prediction step
replaces `expected_means` with one entry per ef-state parent. **Fix:** check
the parent count when the network is built, or derive `batch_size` from it.
**hgf.dart:** `build()` throws a `StateError` naming both counts.

### P17

**The unbounded volatility update, pyhgf's default, runs away on 3-level
binary HGFs** (JAX and rshgf, identically). Found through the playground's
binary preset (ω₂ = −3, ω₃ = −4, μ₃ = 1, reversal learning).
Repro: [`pyhgf/binary_unbounded_divergence.py`](pyhgf/binary_unbounded_divergence.py):

```
reversal, 2000 trials (2000 trials)
  w2=-3.0  w3=-4.0  standard   JAX: bounded            max mu2     1.0, min mu2     0.3   rshgf: bounded
  w2=-3.0  w3=-4.0  eHGF       JAX: bounded            max mu2     1.0, min mu2   -18.2   rshgf: bounded
  w2=-3.0  w3=-4.0  unbounded  JAX: |mu1|>50 at 361    max mu2    30.3, min mu2     1.0   rshgf: |mu1|>50 at 361
  w2=-3.0  w3=-6.0  unbounded  JAX: bounded            max mu2     2.2, min mu2     1.0   rshgf: bounded
pyhgf binary data x6 (1920 trials)
  w2=-3.0  w3=-4.0  unbounded  JAX: |mu1|>50 at 368    max mu2    30.3, min mu2     1.0   rshgf: |mu1|>50 at 368
```

Under the unbounded update the volatility parent's mean never falls below
its initial value (min μ₂ = 1.0 in every run): it only ratchets up, until
x₁'s predicted precision collapses and μ₁ jumps to ~1e5 and oscillates in
sign. A lower ω₃ only delays it: hgf.dart (bit-for-bit the same) runs away
at trials 2840–3052 with ω₃ = −6 and 4347–4413 with (−4, −6) on 10 000
reversal trials, seeds 1–3. Under the standard update μ₂ instead never
rises above its initial value; eHGF drifts down without bound on long
series (μ₂ ≈ −100 after 10 000 trials). Not diagnosed further: the
asymmetry suggests the unbounded update's mean step for a volatility
parent of a binary-input node's value parent keeps one sign.
**hgf.dart:** matches pyhgf. The playground's binary preset uses ω₂ = −4,
ω₃ = −6 and flags runaway beliefs.

**Also the categorical node** (`add_nodes(kind="categorical-state")`),
whose implied network has the same shape: binary nodes, their continuous
value parents, and one shared volatility parent. With pyhgf's defaults
(ω = −4 throughout), 3 categories and a likely outcome (p = 0.8) that
switches every 50 trials, a logit runs away at trials 290–320 (seeds 1–3,
hgf.dart; pyhgf not run on this case yet, but the update is the same
code). With the shared parent at ω = −6 it runs away at trials 2377–2564
(seeds 1–6); at −8 it stays bounded for 10 000 trials. Standard and eHGF
stay bounded at all three. The playground's categorical preset uses −8.

### P18

**`DeepNetwork`'s implied volatility levels run away, under all three
volatility updates, even with the weights frozen** (JAX, and rshgf up to the
first NaN). Found while porting the vectorised backend (hgf_deep, which
reproduces it bit for bit).
Repro: [`pyhgf/deep_volatility_runaway.py`](pyhgf/deep_volatility_runaway.py)
(a 2-4-3 tanh network, pyhgf's defaults otherwise, fixed N(0, 0.5²) weights,
N(0, 1) predictors and targets, `weight_update=False`):

```
unbounded hidden mean_vol max at t=0,10,20,30: 0.0466, 12, 60.7, 115   first NaN at sample: 38
eHGF      hidden mean_vol max at t=0,10,20,30: 0.00739, 6.01, 2.77, nan   first NaN at sample: 10
standard  hidden mean_vol max at t=0,10,20,30: 0.00615, 7.43, 11.3, 12   first NaN at sample: 16
```

The hidden layer's `mean_vol` ratchets up in steps (in one run 17 → 56 →
106 → 204 within 27 samples), its predicted precision collapses
(π̂ ≈ e^−μ_vol, down to 1e−185), one posterior mean step `msg·g′/π` then
moves the node by orders of magnitude (to −4e183), and the volatility
posterior goes NaN. Weight learning is not the driver: frozen weights run away sooner.

**Cause.** A hidden layer's value posterior precision is
`π = clip(π̂ + Σ w²·γ_a·g′² − g″·Σ w·g_a·δ_a, π̂, max)`
(`vectorised_layer_posterior_update`): the information term plus a
second-order curvature term, floored at the prior precision π̂. The
curvature term is often negative and larger than the information (tanh
nodes away from 0), so π falls back to π̂, while the mean still moves by
the full message, `Δμ = g′·Σ w·g_a·δ_a / π`. Once π̂ is small (Ω ≥ 1 per
sample with pyhgf's default ω), that step is 10³–10¹⁵: in one hgf_deep trace
of the moons task π̂ = 6e−4, information 0.02, curvature −0.40, and the node
moves by 2.2e3 where the information alone would give 69. The volatility
prediction error `π̂/π + π̂·(μ − μ̂)² − 1` then takes values of 10³–10⁴,
μ_vol jumps, π̂ = e^−μ_vol collapses further, and the loop closes.

**Fix (proposed):** floor π at `π̂ + Σ w²·γ_a·g′²` instead of π̂, i.e. drop
a negative curvature term (a Gauss–Newton step when the curvature would cut
the evidence). The step can then never exceed the one the information alone
makes. With it, the repro's three schemes stay finite (JAX, patched):

```
floor unbounded hidden mean_vol max at t=0,10,20,30: 0.031, 0.229, 0.46, 0.461   first NaN at sample: None
floor eHGF      hidden mean_vol max at t=0,10,20,30: -0.0112, 0.105, 0.363, 0.189   first NaN at sample: None
floor standard  hidden mean_vol max at t=0,10,20,30: -0.00938, 0.0881, 0.301, 0.208   first NaN at sample: None
```

and in hgf_deep, 500 samples on five seeds of the same setting stay finite
under every scheme (μ_vol ≤ 1.7; pyhgf goes NaN at samples 5–270).
**hgf.dart:** `DeepFixes.curvatureFloor`, on by default
(`DeepFixes.recommended`; `DeepFixes.none`, `DeepNetwork.defaultFixes` or
`-DHGF_DEEP_UPSTREAM=true` give pyhgf). The `floor_*` fixtures check it
against the patched JAX (`_CurvatureFloor` in
[`tool/parity/gen_deep_fixtures.py`](../../tool/parity/gen_deep_fixtures.py)).

**Still open: learning.** With weight learning on 2-D classification (moons,
circles; 2-16-16-1 tanh, Adam 0.01) the floor removes most NaN (moons
0.91 → 0.99 test accuracy with the standard update) but not all: hidden
nodes still drift to μ_vol ≈ 10–15 (Ω ≈ 10⁵ per sample), where their
pseudo-observations are effectively ignored, and the standard update's
volatility precision `π̂_v + ½γ² + γ²δ − ½γδ` goes negative once γ ≈ ½ and δ
is large (the scheme's known limitation). Networks without a volatility
level, or with a value-level tonic volatility, learn every toy task to
0.99–1.00. This is the model's own feedback (large steps read as
volatility), not a numerical fault; a principled damping would be a
modelling change.

**rshgf diverges from JAX silently after the first NaN:** Rust's
`f64::max`/`min` return the other operand when one is NaN, where
`jnp.maximum`/`clip` propagate it, so rshgf keeps finite beliefs that JAX
has already lost. Every per-sample `fit` gap between rshgf f64 and JAX on
the hgf_deep fixtures traces back to such a NaN.
**hgf.dart:** matches JAX, NaN included (`hgf_deep`'s fixtures
`deep4_tanh_*`, `fit_eHGF_tanh`, `fit_adam_continued` contain it).

**How common (2026-11-10):** with Xavier-scale random weights, N(0, 0.5²)
data, Adam(1e-3) and pyhgf's defaults otherwise (carried precision), most
random networks run away within a few hundred samples. Of six 3-12-9-4 tanh
networks, 4 left f32's range or went NaN within 120 samples under each update
(after 8–117 samples; hgf_gpu `native_deep_test.dart`, "random members").
Over 200 samples, 1–3 of six 3-12-4 networks and 3–6 of six 3-8-8-4 ones
did, depending on update and data scale. Without carried
precision (`carryPrecision: false`, no volatility levels) none did over 400
samples. This bounds any sequential `fit` comparison, and what an ensemble of
deep HGF networks can be trained for, until it is fixed.


### P19

**`DeepNetwork`: a runaway volatility level becomes NaN through overflow
and cancellation** (JAX), after P18 has let μ_vol grow past ~700.

- The value-level prediction's effective precision is `γ = Ω·π̂` with
  `Ω = t·exp(μ_vol + …)`: `∞·0 = NaN` once Ω overflows, where its value
  `Ω/(a + Ω)` tends to 1.
- The eHGF and unbounded volatility posteriors rebuild the known variance
  as `max(1/π̄ − Ω, 1e-128)`, which cancels catastrophically once Ω ≫ 1/π
  (P18's "ruled out" note) and is `∞ − ∞` once Ω overflows.
- The eHGF re-prediction `r = t·exp(ω + μ_vol)` overflows the same way:
  `r/(p + r) = ∞/∞`.

**Fix (proposed):** store the known variance (the carried `1/π`) at
prediction time; write `γ = 1/(1 + a/Ω)`; and evaluate the eHGF terms as
`σ(ln r − ln p)`, `2σ(ln r − ln p) − 1` and
`exp(−logaddexp(ln p, ln r))`. These are the same values wherever pyhgf is
finite (hgf_deep checks every fixture to 1e-7 relative up to pyhgf's first
NaN), and a runaway node keeps finite beliefs with π̂ → 0, so it drops out
of its parents' updates instead of turning the layers above it NaN.
**hgf.dart:** `DeepFixes.logSpaceVolatility`, on by default.

**Open: the improper posterior.** A node whose volatility has run away has
π̂ = 0 exactly; if its tanh is also saturated (g′ = 0 in double), its
posterior is `π = 0 + 0` and its mean step `msg·g′/π = 0/0`. A flat prior
meeting no evidence has no posterior, so this is a modelling question,
not a numerical one (hgf_deep leaves the NaN). One option: no update when
π = 0 (the limit of a fixed prior as the information vanishes).

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

### R3

**The GELU coupling is approximate** (`src/math.rs:313`): `erfc` is
Abramowitz–Stegun 7.1.26 (|error| ≤ 1.5e-7, 5e-5 relative in the tail),
though the doc comment calls it the exact erf-based GELU. JAX's
`jax.nn.gelu(approximate=False)` uses XLA's full-precision `erfc`.
Repro: [`pyhgf/rshgf_gelu_erfc.py`](pyhgf/rshgf_gelu_erfc.py): a value
parent with GELU coupling on the USD/CHF series, float64:

```
max |rshgf - JAX| expected_mean (float64): 8.737146961479425e-08
```

Fix: a full-precision `erfc` (`libm::erfc`, or the positive-term series
below 1.5 and the continued fraction above that hgf.dart now uses, 9e-15
relative). **hgf.dart:** had copied the approximation; fixed 2026-11-09
(`erfc` in `math.dart`). The f32 GPU kernel keeps it: 1.5e-7 is f32 rounding
level.

### R4

**rshgf's batched `DeepNetwork` step disagrees with JAX** (and with its own
per-sample path) from the first call, on any network with an interior layer
under a curved coupling (`src/vectorised/batched.rs`):

1. `batched_posterior_update` weights the g″ term of the posterior precision
   by the effective precision, `Wᵀ(eff∘δ)`. Since #456 the per-sample kernel
   and JAX weight it by the relaxed gain, `Wᵀ(gain∘δ)`. The two coincide on
   leaf children (gain = eff there), which is all
   `test_batch_update_matches_per_sample_reference` exercises (3 layers).
2. `batched_volatile_prediction` ignores `predict_precision`.

Repro: [`pyhgf/rshgf_deep_batched_kernel.py`](pyhgf/rshgf_deep_batched_kernel.py)
(2-5-4-3 tanh network, one `batch_update` of 32 samples):

```
predict_precision=True   max |rshgf - JAX| input errors: 2.61e-01  (rshgf float32)
predict_precision=False  max |rshgf - JAX| input errors: 5.15e-01  (rshgf float32)
```

Fix: [`pyhgf/rshgf_deep_batched_kernel.patch`](pyhgf/rshgf_deep_batched_kernel.patch)
(applies to `6be12eb4`; also fixes R5). Built with `--features f64`, rshgf
then matches JAX to ~1e-15 on every hgf_deep fixture (input errors 4.4e-16 /
6.7e-16 in the repro), except GELU networks (R3). Without the `f64` feature
the remaining gap is f32 rounding (JAX in float32 lands as far off).
**hgf.dart:** follows JAX.

### R5

**`DeepNetwork(predict_precision=False)` panics** as soon as a volatile layer
has a volatility parent (the default): `layer_prediction` sets
`expected_*_vol` to `None` and the prediction error unwraps them. JAX passes
the frozen volatility level through. Repro:
[`pyhgf/rshgf_deep_predict_precision_panic.py`](pyhgf/rshgf_deep_predict_precision_panic.py):

```
JAX: ok, predictions finite: True
rshgf: PanicException: called `Option::unwrap()` on a `None` value
```

Fix: in the patch of R4. **hgf.dart:** follows JAX.

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

### D7

**`Expando<int>` loses its values when it is the program's only `Expando`
instantiation:** reading back a stored int throws "The Wasm reference is
not a Dart object" (dart2wasm 3.13.7, every `-O` level, Chrome and Node).
Repro: [`dart2wasm/expando_int.dart`](dart2wasm/expando_int.dart), via
`run.sh`. dart2js and the VM are fine, and so is dart2wasm once some other
`Expando<T>` is instantiated anywhere (which is why `dart test`'s pages,
which use Expandos of their own, didn't show it). Found 2026-11-06 with
`FitWorkers`, whose id tables were `Expando<int>`s: they now hold a small
wrapper class.

