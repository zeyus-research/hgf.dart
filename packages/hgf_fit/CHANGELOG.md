## Unreleased

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
