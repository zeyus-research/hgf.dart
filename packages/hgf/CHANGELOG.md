## Unreleased

- Read-only graph accessors: `Network.valueParentsOf`, `valueChildrenOf`,
  `volatilityParentsOf`, `volatilityChildrenOf`, their coupling strengths
  (`valueParentStrengthsOf`, `volatilityParentStrengthsOf`) and
  `couplingOf`, for drawing and inspecting a network.
- Dirichlet process nodes (`NodeKind.dpState`, `Network.addDpState`, and
  `Network.addEfState(valueChildren: [...])` for their clusters): each
  observation goes to the most likely open cluster or opens a new one, with
  pyhgf's new-cluster proposals (JAX's threefry PRNG and `erf_inv`,
  reproduced). New fields `NodeField.alpha`, `sensoryPrecision`, `nTotal`,
  `nActiveCluster`, and `VectorField.n`, `expectedMeans`, `expectedSigmas`,
  `activated`. Checked against pyhgf JAX on 6 new fixtures (VM, dart2js,
  dart2wasm, `-O4`). Unlike pyhgf, missing observations are skipped, the
  node may follow a continuous one, and a cluster count that doesn't match
  `batchSize` is reported at `build()`. About 1.3 ms per trial on AOT
  (pyhgf JAX: 3.8 ms). `BatchNetwork` and `sample` don't take them yet.
- Update order: an ef-state node without parents takes its step on the
  first pass even when it has children, as in pyhgf (only dp-state clusters
  are affected).

- Categorical state nodes (`NodeKind.categoricalState`,
  `Network.addCategoricalState`): one-hot observations over `n`
  categories, one binary HGF per category with a shared volatility parent
  (pyhgf's `2n + 2` nodes), and the implied Dirichlet distribution
  (`VectorField.alpha`), its KL divergence (`NodeField.klDivergence`) and
  surprise (`NodeField.surprise`). Checked against pyhgf JAX on 7 new
  fixtures (VM, dart2js, dart2wasm, `-O4`). Unlike pyhgf, several
  categorical nodes can share a network, a categorical node can follow a
  continuous one (pyhgf crashes), and the implied nodes take parameters.
  `BatchNetwork` and `sample` don't take them yet.
- `logGamma`, `gammaSign`, `digamma` and `dirichletKullbackLeibler` (in log
  space: finite where pyhgf's overflows to NaN).
- `Trajectories`: asking for a field that wasn't recorded now names it.

- Exponential-family state nodes (`NodeKind.efState`, `Network.addEfState`):
  univariate or multivariate normal sufficient statistics, with a fixed
  learning rate (`nus`) or HGF learning (`EfLearning.hgf1/2/3`, which adds
  pyhgf's implied continuous nodes). Checked against pyhgf JAX on 11 new
  fixtures (VM, dart2js, dart2wasm, `-O4`).
- Vector-valued attributes: `VectorField` (`mean`, `xis`,
  `observation_ss`), `Network.getVector` / `setVector` / `getInitialVector`
  / `setInitialVector` / `vectorLength`, recorded in `Trajectories`
  (`vectorFields`, `vectorSeries`, `vectorAt`, and in `toJson`).
  `NodeField.nus`.
- Inputs may span several columns: `Network.nInputColumns`. **Breaking for
  networks with multivariate inputs only:** `observed` is one flag per input
  node (`nTime × nInputs`), as in pyhgf; unchanged otherwise.
- `UpdateStep.pyhgfName`: the JAX function name of each step. The parity
  tests now also check the update order against pyhgf's JAX schedule.
- `BatchNetwork`: B copies of a `Network` run in lock step, with
  per-element initial values (`setInitial`, `setInitialAt`) and coupling
  strengths (`setValueCoupling`, `setVolatilityCoupling`), shared or
  per-element observations, `inputData` / `surprise` (per-element totals and
  per-trial) / `step`. Each element matches `Network` bit for bit (checked on
  every fixture on the VM, dart2js and dart2wasm, including `-O4`).
  The update steps run stage by stage over blocks of elements, so the CPU
  overlaps the elements' exps and divisions. Per element-trial that is
  1.4–1.9× faster than looping a `Network` on AOT, and 1.2–1.5× on Wasm.
  No trajectory recording yet.
- `ResponseFunction.batchSurprise` (implemented by every built-in model) and
  `ResponseFunction.fn(..., batch:)`.
- `Network.setValueCoupling` / `setVolatilityCoupling`, to change a coupling
  strength (e.g. a fitted κ) after building.
- Surprise and response functions: `gaussianSurprise`, `binarySurprise`,
  `binarySurpriseFinitePrecision`, `gaussianDensity`,
  `sigmoidInverseTemperature`; the `ResponseFunction` interface with ports of
  `pyhgf.response` and a `GaussianResponse` model; `Network.surprise` (no
  recording, optional per-trial output); `inputData(response:)`;
  `Trajectories.nodeSurprise` / `totalSurprise` / `responseSurprise`.
  Checked against pyhgf on every fixture (VM, dart2js, dart2wasm).
- `NodeKind.constantState` (pyhgf's bias node).
- Custom value-coupling functions: the `Coupling` interface (implemented by
  `CouplingFn`) and `CustomCoupling` with user g, g', g'' and a
  finite-difference `check()`; `NetworkSpec.build(couplings: ...)` resolves
  them by name. No cost on the named functions.
- Generative mode: `Network.sample` (pyhgf's `sample`), and
  `saveState`/`restoreState` (`NetworkState`).
- `Network.inputIdxs` no longer allocates on each call.
- `Network.ehgfMgfCorrection` (spec key `ehgf_mgf_correction`), opt-in and
  **diverging from pyhgf 0.3.2**: adds the relaxed prediction's
  `Σ κ²/(2π̂)` terms to the eHGF volatility re-prediction. Without them,
  relaxed eHGF diverges on 3-level continuous HGFs. Checked against pyhgf
  JAX patched with the same change; see the README and
  `example/ehgf_mgf_correction.dart`.
- `inputData(into: traj)` re-records into an existing `Trajectories` of the
  same shape. Recording is staged in blocks: on AOT about 2× cheaper, and 6×
  with `into:`.
- Faster on Wasm: the unbounded update does fewer exp/log calls, dart2wasm
  builds use a table-driven exp, and hot-path columns are no longer boxed.
  On the 3-level 10k series, AOT went 13.8 → 9.2 ms and Wasm 24.2 → 13.6 ms.

## 0.1.0-dev.1

- Initial feasibility spike: nodalised continuous and binary HGF engine
  (standard, eHGF and unbounded volatility updates; relaxed and mean-field
  value coupling; non-linear coupling functions; missing observations;
  irregular time steps), checked against pyhgf to 1e-9 on the VM, dart2js
  and dart2wasm.
- `Network.inputData` / `step` / `predict` / `reset` / `setInitial`,
  `Trajectories` (CSV/JSON export), `NetworkSpec` (JSON networks).
