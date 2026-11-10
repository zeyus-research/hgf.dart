## Unreleased

- Choice tasks. `SoftmaxReadout`: choices among options valued at
  weighted sums of beliefs (`pₖ ∝ exp(β vₖ)`; one node per option with
  `SoftmaxReadout.nodes`, or e.g. `W − L` for pyhgf's two-armed bandit),
  with exact gradients, sampling and `FitCodec` support.
  `HgfModel.simulateChoices(…, reveals:)` simulates the closed loop where
  the choice decides which inputs are observed, and returns fit-ready
  subjects. `IowaGamblingTask`: the decks' payoff schedule, a network and
  readout, and `data(choices, payoffs)` for recorded data
  (`example/iowa_gambling_task.dart`, `test/choice_task_test.dart`).
- `DpSurprise` is serialisable (`ObservationSurprise(DpSurprise())`), so dp
  networks can be fitted (α, sensory precision); without exact gradients,
  use `NelderMead` for MAP (`test/dp_fit_test.dart`).
- `FitCodec.encodeTarget` / `decodeTarget` are public (a parameter's
  target as JSON on its own).
- `Lbfgs` and `NelderMead` run as explicit state machines instead of
  `sync*` bodies, with bit-identical iterates. On the JIT VM (`dart run`,
  `dart test`, Flutter debug) every live `sync*` run compiled its own
  on-stack-replacement copy of the loop once the function got hot,
  10–25 ms each: `fitMap` on 200 subjects took 4.2 s instead of 0.14 s.
  AOT, dart2wasm and dart2js were unaffected.
- Categorical models: `CategoricalSoftmaxReadout` (choices among a
  categorical node's categories, `pₖ ∝ μ̂ₖ^β`, with exact gradients and
  sampling) and `ObservationSurprise(CategoricalSurprise())`; both in
  `FitCodec`.
- `Subject.inputs` has `network.nInputColumns` columns per trial (one-hot
  rows for categorical inputs); `observed` one flag per trial and input.

- `FitWorkers.defaultCount`: the logical processors `spawn` uses by
  default (`Platform.numberOfProcessors` / `navigator.hardwareConcurrency`).

- `NutsEnsemble`: the NUTS drivers now advance all chains as a state
  machine over flat per-chain buffers (Stan's tree recursion as an
  explicit stack) instead of one `sync*` generator per chain. The draws
  are bit-identical to `Nuts.chain`'s; the driver's own cost per chain and
  leapfrog step drops from 1.7–5.8 µs (400–8000 chains) to 0.7–0.9 µs and
  no longer grows with the number of chains. GPU NUTS on 2000 agents
  (8000 chains) goes from 139 s to 30 s.
- `BatchedObjective.evaluateGradientPasses`: points and gradients in flat
  arrays (one pass over the points themselves with exact gradients).
- `PassRunner`: the fitting functions' extension methods (`fitMap`,
  `laplace`, `sampleNuts`, `sampleHierarchical`, `fitEmpiricalBayes`) are
  now on this interface, which `FitWorkers` implements, so other
  evaluators (`hgf_gpu`'s `GpuFitter`) run them too.
  `FitWorkers.evaluatePass` answers a single pass.
- `Pass.precise` marks passes that need f64 (finite-difference gradients,
  Hessians, `fitMap`'s reported values); lower-precision runners answer
  them on the CPU.
- `HgfModel.completeNegLogPosterior`: the prior and Jacobian terms (and
  their gradient) from likelihoods computed elsewhere.
- `benchmark/bin/nuts_overhead.dart`: the NUTS driver's own cost per chain
  and step.
- For apps: `FitWorkers.cancel()` (running jobs fail with `FitCancelled`
  before their next pass), `FitWorkers.yieldInterval` (a pool without
  workers gives way to the event loop that often, so a UI keeps drawing),
  and `SamplingProgress` for `sampleNuts` (transitions done / total;
  `ChainOutput.iterations` counts each chain's).
- `FitWorkers` on the web: Web Workers running a separately compiled worker
  (`package:hgf_fit/fit_worker.dart`, `runFitWorker()`), compiled with
  `dart compile js` or `wasm`; `FitWorkers.spawn(script:
  FitWorkerScript.wasm('fit_worker.wasm'))`. Results are bit-identical to
  the main thread when the worker is compiled like the page. The pool,
  its slicing and streams are shared with the isolate version (now one
  `FitWorkers` class over a transport).
- `FitCodec`: models as JSON (`encodeModel`/`decodeModel`, parameters,
  priors, transforms, response models), with decoders for custom types.
  The built-in priors, transforms and response models have `toJson()`;
  custom ones override it. A decoded model evaluates bit for bit like the
  original.
- Errors raised in a worker now arrive as an error carrying the worker's
  message and stack trace (no longer `RemoteError`, which the web lacks).
- Isolate-parallel fitting: `FitWorkers` is a pool of worker isolates
  (`FitWorkers.spawn()`, one per processor by default), and
  `workers.fitMap`, `.laplace`, `.sampleNuts`, `.sampleHierarchical`,
  `.fitEmpiricalBayes` and `.evaluate` give the same results as the
  synchronous functions, bit for bit. Each batched pass is split into
  slices across the workers, and drivers with independent tasks run them as
  concurrent streams, so the slowest chain doesn't set every pass's barrier.
  On the web the pool has no workers and runs inline.
- The drivers are now `Program`s: `sync*` generators yielding `Pass`es
  (one `HgfModel.negLogPosterior` call each) and `Concurrent` streams,
  answered by `runInline` or a `FitWorkers` pool, and public so that custom
  drivers can run on the pool too. `BatchedObjective.evaluatePasses` /
  `hessianPasses`, `HierarchicalObjective.evaluatePasses` and
  `minimizeAllPasses` are the program forms of the existing methods.

- Hierarchical models: `HierarchicalModel` over a subject-level `HgfModel`,
  with pooled parameters (each subject's value, on the parameter's
  unconstrained scale, drawn from a `Population` `N(μ, σ²)` with hyperpriors
  on `μ` and `σ`), shared parameters (one value for all subjects) and
  independent ones; centred or non-centred (default). `sampleHierarchical`
  samples the joint posterior by NUTS, all subjects of all chains in one
  batched pass per leapfrog step, with exact gradients; `HierarchicalFit`
  gives summaries by group or subject and ArviZ JSON with a `subject`
  dimension. `fitEmpiricalBayes`: type-II maximum likelihood of the
  populations by Laplace-EM, with shrunken subject estimates and the
  evidence.
- `HgfModel.withPriors` (a copy with some priors replaced, sharing the
  compiled network and batches), `negLogPosterior(includePrior:)` and
  `BatchedObjective(includePrior:)`, and `minimizeAll` (the lock-step
  optimiser driver behind `fitMap`).

- Exact gradients: `HgfModel.negLogPosterior(gradient:)` differentiates the
  negative log posterior with forward-mode tangents through the filter
  (`BatchNetwork.tangents`, one direction per parameter), the priors
  (`Prior.dLogDensity`) and the transforms (`Transform.dToConstrained`,
  `dLogAbsJacobian`). `HgfModel.hasExactGradients` says whether the model
  allows it (the response model reports partials and every parameter
  target carries tangents). `Gradients` (`exact`, `central`, `forward`,
  replacing `Differences`) picks the method for `BatchedObjective`,
  `fitMap` and `sampleNuts`; exact is the default where available.
- `ResponseModel.surprisePartials` / `supportsGradients` and
  `ResponsePartials` (`add` for beliefs, `addParameter` for the model's own
  parameters); implemented by `GaussianReadout`, `BinarySoftmaxReadout`
  and `ObservationSurprise`, and by `ResponseModel.fn(surprisePartials:)`.

- First version: `HgfModel`, `Subject`, `Parameter` (node fields, couplings,
  response parameters), priors (`Normal`, `TruncatedNormal`, `Uniform`,
  `HalfNormal`, `LogNormal`, `GammaPrior`, `Flat`) with matching transforms,
  response models (`GaussianReadout`, `BinarySoftmaxReadout`,
  `ObservationSurprise`, `ResponseModel.fn`), batched objectives with
  finite-difference gradients (`BatchedObjective`), `Lbfgs` and `NelderMead`,
  and `fitMap`, which fits every subject in lock step.
- `laplace`: Laplace approximation around MAP estimates (Hessian by finite
  differences in one batched pass, `BatchedObjective.hessians`), with
  covariance, standard errors, intervals and log evidence. `normalQuantile`.
- `sampleNuts`: posterior draws by the No-U-Turn Sampler (Stan's algorithm:
  multinomial trajectories, generalised no-U-turn criterion, dual-averaging
  step size, windowed unit/diagonal/dense metric adaptation), every chain of
  every subject advanced in lock step with one batched pass per leapfrog
  step. `PosteriorDraws` with summaries, rank-normalised split R̂, bulk and
  tail ESS, MCSE and E-BFMI (the same estimators as ArviZ), CSV export in
  Stan's column layout, and `NutsFit.toJson()` for ArviZ's `from_dict`.
  `sampleFunction` runs the sampler on any log density.
