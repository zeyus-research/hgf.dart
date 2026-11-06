/// Parameter estimation for `package:hgf` networks.
///
/// Describe the model with an [HgfModel] (network, [ResponseModel], free
/// [Parameter]s with [Prior]s), the data with [Subject]s, and fit with
/// [fitMap]. Every subject is fitted at once: each optimiser step of every
/// subject, with its exact gradient (forward-mode tangents through the
/// filter), is one `BatchNetwork` pass.
/// [laplace] adds the Laplace approximation around the estimates (standard
/// errors, intervals, log evidence for model comparison) in one more pass.
/// [sampleNuts] draws full posteriors per subject. For group-level
/// inference, a [HierarchicalModel] pools parameters across subjects:
/// [sampleHierarchical] samples the joint posterior, [fitEmpiricalBayes]
/// estimates the populations by Laplace-EM.
///
/// ```dart
/// final model = HgfModel(
///   network: Network()
///     ..addNodes(kind: NodeKind.binaryState)
///     ..addNodes(valueChildren: [0]),
///   response: const BinarySoftmaxReadout(),
///   parameters: [
///     Parameter.node('omega2', 1, NodeField.tonicVolatility,
///         prior: TruncatedNormal(-3, 5, lower: -8, upper: 2)),
///     Parameter.response('inverseTemperature',
///         prior: const Uniform(0.2, 20)),
///   ],
/// );
/// final fit = fitMap(model, subjects);
/// for (final r in fit.results) print(r.parameters);
/// final lap = laplace(model, fit);
/// print(lap.logEvidence);
/// for (final r in lap.results) print(r.interval());
/// ```
library;

export 'src/empirical_bayes.dart';
export 'src/fit.dart';
export 'src/hierarchical.dart';
export 'src/laplace.dart';
export 'src/model.dart';
export 'src/optim/lbfgs.dart';
export 'src/optim/nelder_mead.dart';
export 'src/optim/optimizer.dart'
    show EvalRequest, OptimizeResult, Optimizer, ResultSlot, minimizeFunction;
export 'src/parameter.dart';
export 'src/prior.dart';
export 'src/response_model.dart' hide bindResponsePartials;
export 'src/sample/diagnostics.dart'
    show essBulk, essMean, essTail, mcseMean, quantileOf, rhat;
export 'src/sample/nuts.dart';
export 'src/sample/sample.dart' hide chainSeed, driveChains;
export 'src/special.dart' show erf, erfc, logNormalCdf, normalQuantile;
export 'src/transform.dart';
