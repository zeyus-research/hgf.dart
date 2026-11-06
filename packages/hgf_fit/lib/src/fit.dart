import 'dart:math' as math;
import 'dart:typed_data';

import 'model.dart';
import 'optim/lbfgs.dart';
import 'optim/optimizer.dart';

/// Evaluates [model]'s objective (negative log posterior, on the
/// unconstrained scale) for many (subject, point) pairs in one batched pass,
/// with gradients by central finite differences in the same pass.
///
/// A point with a gradient costs `2k + 1` batch elements (`k` parameters).
/// The step along `yᵢ` is `fdStep · max(1, |yᵢ|)`; where one side of the
/// stencil isn't finite, the one-sided difference is used.
final class BatchedObjective {
  BatchedObjective(
    this.model,
    this.subjects, {
    this.fdStep = 1e-5,
    this.jacobian = false,
  });

  final HgfModel model;
  final List<Subject> subjects;

  /// Relative finite-difference step.
  final double fdStep;

  /// Add the transforms' Jacobian terms (a density on the unconstrained
  /// scale, as samplers need). MAP estimation leaves them out, so the mode
  /// is that of the posterior on the parameters' natural scale.
  final bool jacobian;

  /// Batched passes run so far.
  int passes = 0;

  /// Batch elements evaluated so far.
  int elements = 0;

  /// The objective at each of [points] (`points[j]` belongs to subject
  /// `subjectOf[j]`), and the gradients where [gradient] says so.
  (Float64List values, List<Float64List?> gradients) evaluate(
    List<int> subjectOf,
    List<Float64List> points,
    List<bool> gradient,
  ) {
    final k = model.dimension, nP = points.length;
    var nE = 0;
    for (var j = 0; j < nP; j++) {
      nE += gradient[j] ? 2 * k + 1 : 1;
    }
    final subj = Int32List(nE);
    final y = Float64List(nE * k);
    final steps = Float64List(nE);
    var e = 0;
    for (var j = 0; j < nP; j++) {
      final p = points[j];
      if (p.length != k) throw ArgumentError('Points must have $k entries');
      subj[e] = subjectOf[j];
      y.setRange(e * k, (e + 1) * k, p);
      e++;
      if (!gradient[j]) continue;
      for (var i = 0; i < k; i++) {
        final h = fdStep * math.max(1.0, p[i].abs());
        for (final sign in const [1.0, -1.0]) {
          subj[e] = subjectOf[j];
          y.setRange(e * k, (e + 1) * k, p);
          final v = p[i] + sign * h;
          y[e * k + i] = v;
          steps[e] = (v - p[i]).abs(); // the step actually taken
          e++;
        }
      }
    }
    final out = Float64List(nE);
    model.negLogPosterior(subjects, subj, y, out, jacobian: jacobian);
    passes++;
    elements += nE;
    final values = Float64List(nP);
    final grads = List<Float64List?>.filled(nP, null);
    e = 0;
    for (var j = 0; j < nP; j++) {
      final f0 = out[e];
      values[j] = f0;
      e++;
      if (!gradient[j]) continue;
      final g = Float64List(k);
      for (var i = 0; i < k; i++) {
        final fp = out[e], hp = steps[e];
        final fm = out[e + 1], hm = steps[e + 1];
        e += 2;
        g[i] = switch ((fp.isFinite, fm.isFinite)) {
          (true, true) => (fp - fm) / (hp + hm),
          (true, false) => (fp - f0) / hp,
          (false, true) => (f0 - fm) / hm,
          _ => double.nan,
        };
      }
      grads[j] = g;
    }
    return (values, grads);
  }
}

/// One subject's MAP estimate.
final class FitResult {
  FitResult({
    required this.subject,
    required this.names,
    required this.x,
    required this.y,
    required this.negLogPosterior,
    required this.negLogLikelihood,
    required this.optimization,
  });

  final Subject subject;
  final List<String> names;

  /// The estimate on the parameters' natural scale.
  final Float64List x;

  /// The estimate on the unconstrained scale.
  final Float64List y;

  /// `−log p(responses | x) − log p(x)` at the estimate.
  final double negLogPosterior;

  /// `−log p(responses | x)` at the estimate.
  final double negLogLikelihood;

  /// `log p(x)` at the estimate.
  double get logPrior => negLogLikelihood - negLogPosterior;

  /// The run that found it (the best of the starts).
  final OptimizeResult optimization;

  bool get converged => optimization.converged;

  /// The estimate by parameter name.
  Map<String, double> get parameters => {
    for (var i = 0; i < names.length; i++) names[i]: x[i],
  };

  @override
  String toString() =>
      'FitResult(${subject.id ?? ''} $parameters, '
      '−log post = ${negLogPosterior.toStringAsFixed(4)}, '
      '${optimization.message})';
}

/// The estimates of a [fitMap] call, with what they cost.
final class MapFit {
  MapFit(this.results, this.passes, this.elements);

  /// One per subject, in order.
  final List<FitResult> results;

  /// Batched passes over the data.
  final int passes;

  /// Batch elements evaluated in total (each a full filter of one series).
  final int elements;
}

/// Maximum a posteriori estimates of [model]'s parameters for each of
/// [subjects], fitted independently but in lock step: each pass evaluates
/// the pending points (and finite-difference stencils) of every subject's
/// optimiser in one `BatchNetwork` run.
///
/// Each subject starts from `start[s]` (constrained scale; default the
/// parameters' starting values), plus [restarts] starts drawn from the
/// priors with [random]; the best run is kept.
MapFit fitMap(
  HgfModel model,
  List<Subject> subjects, {
  Optimizer optimizer = const Lbfgs(),
  List<List<double>>? start,
  int restarts = 0,
  math.Random? random,
  double fdStep = 1e-5,
}) {
  if (start != null && start.length != subjects.length) {
    throw ArgumentError('start must have one vector per subject');
  }
  if (restarts < 0) throw ArgumentError.value(restarts, 'restarts');
  final rng = random ?? math.Random(0);
  final objective = BatchedObjective(model, subjects, fdStep: fdStep);
  final tasks = <_Task>[];
  for (var s = 0; s < subjects.length; s++) {
    final starts = [
      start?[s] ?? model.start(),
      for (var r = 0; r < restarts; r++) model.samplePrior(rng),
    ];
    for (final x0 in starts) {
      if (x0.length != model.dimension) {
        throw ArgumentError(
          'Start vectors must have ${model.dimension} values',
        );
      }
      final slot = ResultSlot();
      final it = optimizer.minimize(model.toUnconstrained(x0), slot).iterator;
      tasks.add(_Task(s, it, slot));
    }
  }
  var active = [
    for (final t in tasks)
      if (t.iterator.moveNext()) t,
  ];
  while (active.isNotEmpty) {
    final subjectOf = <int>[], points = <Float64List>[], grad = <bool>[];
    for (final t in active) {
      final r = t.iterator.current;
      for (final p in r.points) {
        subjectOf.add(t.subject);
        points.add(p);
        grad.add(r.gradient);
      }
    }
    final (values, grads) = objective.evaluate(subjectOf, points, grad);
    var j = 0;
    for (final t in active) {
      final r = t.iterator.current;
      final n = r.points.length;
      r.values = Float64List.sublistView(values, j, j + n);
      if (r.gradient) {
        r.gradients = [for (var i = j; i < j + n; i++) grads[i]!];
      }
      j += n;
    }
    active = [
      for (final t in active)
        if (t.iterator.moveNext()) t,
    ];
  }

  // The best run per subject, then its likelihood and prior parts.
  final best = List<OptimizeResult?>.filled(subjects.length, null);
  for (final t in tasks) {
    final r = t.slot.result!;
    final b = best[t.subject];
    if (b == null || r.value < b.value || (b.value.isNaN && !r.value.isNaN)) {
      best[t.subject] = r;
    }
  }
  final k = model.dimension, n = subjects.length;
  final y = Float64List(n * k);
  for (var s = 0; s < n; s++) {
    y.setRange(s * k, (s + 1) * k, best[s]!.x);
  }
  final post = Float64List(n), nll = Float64List(n);
  model.negLogPosterior(
    subjects,
    Int32List.fromList([for (var s = 0; s < n; s++) s]),
    y,
    post,
    negLogLikelihood: nll,
  );
  final names = model.names;
  return MapFit(
    [
      for (var s = 0; s < n; s++)
        FitResult(
          subject: subjects[s],
          names: names,
          x: model.toConstrained(best[s]!.x),
          y: best[s]!.x,
          negLogPosterior: post[s],
          negLogLikelihood: nll[s],
          optimization: best[s]!,
        ),
    ],
    objective.passes + 1,
    objective.elements + n,
  );
}

final class _Task {
  _Task(this.subject, this.iterator, this.slot);

  final int subject;
  final Iterator<EvalRequest> iterator;
  final ResultSlot slot;
}
