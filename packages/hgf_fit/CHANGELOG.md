## Unreleased

- First version: `HgfModel`, `Subject`, `Parameter` (node fields, couplings,
  response parameters), priors (`Normal`, `TruncatedNormal`, `Uniform`,
  `HalfNormal`, `LogNormal`, `GammaPrior`, `Flat`) with matching transforms,
  response models (`GaussianReadout`, `BinarySoftmaxReadout`,
  `ObservationSurprise`, `ResponseModel.fn`), batched objectives with
  finite-difference gradients (`BatchedObjective`), `Lbfgs` and `NelderMead`,
  and `fitMap`, which fits every subject in lock step.
