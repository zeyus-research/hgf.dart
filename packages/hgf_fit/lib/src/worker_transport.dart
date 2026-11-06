import 'dart:typed_data';

import 'codec.dart';
import 'model.dart';
import 'parameter.dart';

/// Where a web page finds the compiled worker script that `FitWorkers`
/// runs its passes on.
///
/// Web Workers can't run the page's own code: compile a worker entry point
/// separately (see `package:hgf_fit/fit_worker.dart`) and serve it with the
/// page. Ignored on the platforms with isolates.
///
/// ```dart
/// final workers = await FitWorkers.spawn(
///   script: FitWorkerScript.wasm('fit_worker.wasm'),
/// );
/// ```
final class FitWorkerScript {
  /// A worker compiled with `dart compile js` (the `.js` file's URL).
  const FitWorkerScript.js(this.url, {this.codec = const FitCodec()})
    : wasm = false;

  /// A worker compiled with `dart compile wasm`: the `.wasm` file's URL,
  /// with the `.mjs` loader next to it (as the compiler writes them).
  const FitWorkerScript.wasm(this.url, {this.codec = const FitCodec()})
    : wasm = true;

  /// Relative URLs are resolved against the page's base URI.
  final String url;
  final bool wasm;

  /// Encodes models for the workers. The worker decodes them with the codec
  /// given to `runFitWorker`; both need the same custom types (only
  /// `FitCodec.responseFunctions` is used on this side).
  final FitCodec codec;
}

/// One slice of a pass, as sent to a worker.
typedef EvaluateRequest = ({
  int id,
  int modelId,
  Int32List subjectIds,
  Int32List subjectOf,
  Float64List y,
  bool jacobian,
  List<bool>? includePrior,
  bool negLogLikelihood,
  bool gradient,
});

/// The messages between `FitWorkers` and its workers, over isolates or Web
/// Workers. Package-internal.
abstract class WorkerTransport {
  /// Number of workers.
  int get count;

  /// Registers a model family (network, response model, parameters) on
  /// every worker. Throws, before sending anything, if the model can't be
  /// sent.
  void registerFamily(int id, HgfModel model);

  /// Registers [parameters] over family [familyId] on every worker.
  void registerModel(int id, int familyId, List<Parameter> parameters);

  /// Registers [subjects] under [ids] on every worker, in one message
  /// (series shared between subjects stay shared).
  void registerSubjects(List<int> ids, List<Subject> subjects);

  /// Forgets the family, model or subject [id] on every worker.
  void drop(int id);

  /// Sends [request] to [worker]; its answer comes to [onResult] or
  /// [onFailure].
  void evaluate(int worker, EvaluateRequest request);

  /// Stops the workers.
  void close();

  /// A slice's outputs.
  late void Function(
    int id,
    int worker,
    Float64List out,
    Float64List? negLogLikelihood,
    Float64List? gradient,
  )
  onResult;

  /// A slice whose evaluation threw.
  late void Function(int id, int worker, String error, String stack) onFailure;

  /// A worker died: the transport can't be used any more.
  late void Function(Object? reason) onExit;
}
