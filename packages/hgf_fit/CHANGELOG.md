## Unreleased

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
