import 'dart:typed_data';

import 'model.dart';
import 'parameter.dart';
import 'pass.dart';

/// A worker's registry of models and subjects, and its evaluation of
/// slices: the worker side of `FitWorkers`, on isolates and Web Workers.
/// Package-internal.
///
/// Registration errors (e.g. a model the worker can't decode) are kept and
/// raised by the first slice that needs the model, so they reach the
/// caller of the fitting function.
final class WorkerHost {
  final _families = <int, Object>{};
  final _models = <int, Object>{};
  final _subjects = <int, Subject>{};

  void addFamily(int id, HgfModel Function() make) {
    try {
      _families[id] = make();
    } catch (e, st) {
      _families[id] = (e, st);
    }
  }

  void addModel(int id, int familyId, List<Parameter> Function() parameters) {
    try {
      final family = _families[familyId];
      _models[id] = switch (family) {
        final HgfModel f => modelWithParameters(f, parameters()),
        final (Object, StackTrace) failed => failed,
        _ => throw StateError('Unknown model family $familyId'),
      };
    } catch (e, st) {
      _models[id] = (e, st);
    }
  }

  void addSubjects(List<int> ids, List<Subject> subjects) {
    for (var i = 0; i < ids.length; i++) {
      _subjects[ids[i]] = subjects[i];
    }
  }

  void drop(int id) {
    _families.remove(id);
    _models.remove(id);
    _subjects.remove(id);
  }

  /// Evaluates a slice; throws what the evaluation (or the model's
  /// registration) threw.
  Pass evaluate({
    required int modelId,
    required Int32List subjectIds,
    required Int32List subjectOf,
    required Float64List y,
    required bool jacobian,
    required List<bool>? includePrior,
    required bool negLogLikelihood,
    required bool gradient,
  }) {
    final model = switch (_models[modelId]) {
      final HgfModel m => m,
      (final Object e, final StackTrace st) => Error.throwWithStackTrace(e, st),
      _ => throw StateError('Unknown model $modelId'),
    };
    return Pass(
      model,
      [
        for (final id in subjectIds)
          _subjects[id] ?? (throw StateError('Unknown subject $id')),
      ],
      subjectOf,
      y,
      jacobian: jacobian,
      includePrior: includePrior,
      negLogLikelihood: negLogLikelihood,
      gradient: gradient,
    )..run();
  }
}
