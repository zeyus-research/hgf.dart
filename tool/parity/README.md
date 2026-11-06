# pyhgf parity tooling

A [uv](https://docs.astral.sh/uv/) project that installs the sibling pyhgf
checkout (`../pyhgf`, next to `hgf.dart`). The install is non-editable,
builds the Rust extension `rshgf`, and needs `cargo`.

```sh
# once: build pyhgf + rshgf into tool/parity/.venv
# (CARGO_TARGET_DIR keeps cargo's build output out of the pyhgf checkout)
CARGO_TARGET_DIR=/tmp/cargo-target uv sync

uv run python gen_fixtures.py               # all scenarios
uv run python gen_fixtures.py binary_2level # just some
uv run python bench_python.py --repeat 7    # JAX (float64) + rshgf timings
JAX_ENABLE_X64=0 uv run python bench_python.py --jax-only   # JAX float32
```

## Fixtures

`gen_fixtures.py` writes `packages/hgf/test/fixtures/<scenario>.json`, each
containing:

- `spec`: the network in `NetworkSpec` JSON. The Dart tests rebuild it from
  this.
- `inputs`, `observed`, `time_steps`.
- `trajectories`: pyhgf JAX run with `JAX_ENABLE_X64=1`, the reference.
- `rshgf_update_sequence`, and `rshgf_max_rel_diff` (the gap between
  pyhgf's own two backends). Where that gap exceeds 1e-9, the fixture also
  stores `rshgf_trajectories`.
- `pyhgf_commit`, the pyhgf version the fixture came from.
- `reference`: what produced `trajectories`.

**Patched references.** The `*_mgf_correction` scenarios set
`"ehgf_mgf_correction": true` (hgf.dart's `Network.ehgfMgfCorrection`). For
them the generator swaps pyhgf's relaxed `precision_update_ehgf` for a
version whose volatility re-prediction includes the relaxed prediction's
`Σ κ²/(2π̂)` terms (`_precision_update_ehgf_mgf_corrected`). It then
clears JAX's caches before and after, and skips rshgf, which is unpatched.
These fixtures check the proposed upstream fix, not released pyhgf, and
their `reference` field says so.

The same goes for `dp_missing` (`"dp_skip_missing": true`): there the
generator wraps pyhgf's `dirichlet_node_prediction_error` so that an
unobserved trial only sets the clusters' `observed` to 0, which is what
hgf.dart does (pyhgf clusters missing observations, upstream P14).

`fixtures/dp/dp_reference.json` is not a scenario: it holds pyhgf's fixed
dp-state cluster proposals (every 50th, plus sums) and some `get_candidate`
results, which `dp_state_test.dart` checks on their own. They come from the
default JAX PRNG mode (`jax_threefry_partitionable`), in float64.

After pulling pyhgf, regenerate with `melos run fixtures` and rerun
`melos run test:parity`. Mismatches show which update equations changed
upstream.
