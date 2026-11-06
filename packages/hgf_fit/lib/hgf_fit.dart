/// Parameter estimation for `package:hgf` networks.
///
/// Describe the model with an [HgfModel] (network, [ResponseModel], free
/// [Parameter]s with [Prior]s), the data with [Subject]s, and fit with
/// [fitMap]. Every subject is fitted at once: each optimiser step of every
/// subject, with its finite-difference gradient, is one `BatchNetwork` pass.
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
/// ```
library;

export 'src/fit.dart';
export 'src/model.dart';
export 'src/optim/lbfgs.dart';
export 'src/optim/nelder_mead.dart';
export 'src/optim/optimizer.dart'
    show EvalRequest, OptimizeResult, Optimizer, ResultSlot, minimizeFunction;
export 'src/parameter.dart';
export 'src/prior.dart';
export 'src/response_model.dart';
export 'src/special.dart' show erf, erfc, logNormalCdf;
export 'src/transform.dart';
