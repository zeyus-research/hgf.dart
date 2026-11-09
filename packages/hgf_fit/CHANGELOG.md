## Unreleased

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
