import 'dart:math' as math;
import 'dart:typed_data';

import 'model.dart';
import 'optim/lbfgs.dart';
import 'optim/optimizer.dart';
import 'pass.dart';

/// How [BatchedObjective] computes gradients.
enum Gradients {
  /// Exact: forward-mode tangents through the filter, one direction per
  /// parameter (`HgfModel.hasExactGradients`). One batch element per
  /// gradient, each costing about `1 + k/2` plain ones (`k` parameters).
  exact,

  /// `(f(y + hᵢ) − f(y − hᵢ)) / 2hᵢ`: `2k + 1` elements per gradient,
  /// error `O(h²)`. Where one side isn't finite, the one-sided difference
  /// is used.
  central,

  /// `(f(y + hᵢ) − f(y)) / hᵢ`: `k + 1` elements per gradient, error
  /// `O(h)`. Enough for HMC, whose exactness doesn't depend on the gradient
  /// (see `Nuts`). NaN where `f(y + hᵢ)` isn't finite.
  forward,
}

/// Evaluates [model]'s objective (negative log posterior, on the
/// unconstrained scale) for many (subject, point) pairs in one batched pass,
/// with gradients in the same pass.
///
/// Gradients are exact by default where the model allows it
/// ([HgfModel.hasExactGradients]), otherwise central differences.
/// A point with a finite-difference gradient costs `2k + 1` batch elements
/// with [Gradients.central], `k + 1` with [Gradients.forward] (`k`
/// parameters); the step along `yᵢ` is `fdStep · max(1, |yᵢ|)`.
final class BatchedObjective {
  BatchedObjective(
    this.model,
    this.subjects, {
    this.fdStep = 1e-5,
    this.jacobian = false,
    this.includePrior,
    Gradients? gradients,
  }) : gradients =
           gradients ??
           (model.hasExactGradients ? Gradients.exact : Gradients.central) {
    if (this.gradients == Gradients.exact && !model.hasExactGradients) {
      throw ArgumentError(
        'The model has no exact gradients (see HgfModel.hasExactGradients)',
      );
    }
  }

  /// How gradients are computed.
  final Gradients gradients;

  final HgfModel model;
  final List<Subject> subjects;

  /// Relative finite-difference step.
  final double fdStep;

  /// Add the transforms' Jacobian terms (a density on the unconstrained
  /// scale, as samplers need). MAP estimation leaves them out, so the mode
  /// is that of the posterior on the parameters' natural scale.
  final bool jacobian;

  /// Which parameters' priors (and Jacobian terms) to include; all if null
  /// (see `HgfModel.negLogPosterior`).
  final List<bool>? includePrior;

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
  ) => runInline((o) => evaluatePasses(subjectOf, points, gradient, o));

  /// [evaluate] as a program: yields the batched passes it needs and puts
  /// the result in [result].
  Iterable<Step> evaluatePasses(
    List<int> subjectOf,
    List<Float64List> points,
    List<bool> gradient,
    Out<(Float64List, List<Float64List?>)> result,
  ) sync* {
    if (gradients == Gradients.exact) {
      yield* _evaluateExact(subjectOf, points, gradient, result);
      return;
    }
    final k = model.dimension, nP = points.length;
    final central = gradients == Gradients.central;
    final sides = central ? const [1.0, -1.0] : const [1.0];
    var nE = 0;
    for (var j = 0; j < nP; j++) {
      nE += gradient[j] ? sides.length * k + 1 : 1;
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
        for (final sign in sides) {
          subj[e] = subjectOf[j];
          y.setRange(e * k, (e + 1) * k, p);
          final v = p[i] + sign * h;
          y[e * k + i] = v;
          steps[e] = (v - p[i]).abs(); // the step actually taken
          e++;
        }
      }
    }
    final pass = Pass(
      model,
      subjects,
      subj,
      y,
      jacobian: jacobian,
      includePrior: includePrior,
      precise: nE > nP,
    );
    yield pass;
    final out = pass.out;
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
        if (!central) {
          final fp = out[e], hp = steps[e];
          e++;
          g[i] = fp.isFinite ? (fp - f0) / hp : double.nan;
          continue;
        }
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
    result.value = (values, grads);
  }

  /// [evaluatePasses] for points that all need a gradient, in flat arrays:
  /// [y] is row-major `n x k`, and so are the result's gradients. With
  /// exact gradients this is one pass over [y] itself (the NUTS drivers'
  /// path).
  Iterable<Step> evaluateGradientPasses(
    Int32List subjectOf,
    Float64List y,
    Out<(Float64List values, Float64List gradients)> result,
  ) sync* {
    final k = model.dimension, n = subjectOf.length;
    if (y.length != n * k) throw ArgumentError('Points must have $k entries');
    if (gradients != Gradients.exact) {
      final inner = Out<(Float64List, List<Float64List?>)>();
      yield* evaluatePasses(
        subjectOf,
        [
          for (var j = 0; j < n; j++)
            Float64List.sublistView(y, j * k, (j + 1) * k),
        ],
        List.filled(n, true),
        inner,
      );
      final (v, g) = inner.value;
      final flat = Float64List(n * k);
      for (var j = 0; j < n; j++) {
        flat.setRange(j * k, (j + 1) * k, g[j]!);
      }
      result.value = (v, flat);
      return;
    }
    final pass = Pass(
      model,
      subjects,
      subjectOf,
      y,
      jacobian: jacobian,
      includePrior: includePrior,
      gradient: true,
    );
    yield pass;
    passes++;
    elements += n;
    result.value = (pass.out, pass.gradient!);
  }

  /// [evaluate] with exact gradients: the points that need one in a pass
  /// with tangents, the others (if any) in a plain pass.
  Iterable<Step> _evaluateExact(
    List<int> subjectOf,
    List<Float64List> points,
    List<bool> gradient,
    Out<(Float64List, List<Float64List?>)> result,
  ) sync* {
    final k = model.dimension, nP = points.length;
    final values = Float64List(nP);
    final grads = List<Float64List?>.filled(nP, null);
    for (final withGradient in [true, false]) {
      final idx = [
        for (var j = 0; j < nP; j++)
          if (gradient[j] == withGradient) j,
      ];
      if (idx.isEmpty) continue;
      final n = idx.length;
      final subj = Int32List(n);
      final y = Float64List(n * k);
      for (var q = 0; q < n; q++) {
        final p = points[idx[q]];
        if (p.length != k) throw ArgumentError('Points must have $k entries');
        subj[q] = subjectOf[idx[q]];
        y.setRange(q * k, (q + 1) * k, p);
      }
      final pass = Pass(
        model,
        subjects,
        subj,
        y,
        jacobian: jacobian,
        includePrior: includePrior,
        gradient: withGradient,
      );
      yield pass;
      final out = pass.out, g = pass.gradient;
      passes++;
      elements += n;
      for (var q = 0; q < n; q++) {
        values[idx[q]] = out[q];
        if (g != null) {
          grads[idx[q]] = Float64List.sublistView(g, q * k, (q + 1) * k);
        }
      }
    }
    result.value = (values, grads);
  }

  /// The objective and its Hessian (row-major `k x k`) at each of [points],
  /// all in one pass.
  ///
  /// Central second differences with step `hᵢ = step · max(1, |yᵢ|)`: the
  /// diagonal from `f(y ± hᵢeᵢ)`, each off-diagonal entry from the four
  /// points `f(y ± hᵢeᵢ ± hⱼeⱼ)`; `2k² + 1` elements per point. An entry
  /// whose stencil touches a non-finite value is NaN. The default step
  /// balances the `O(h²)` truncation error against rounding (`~ε/h²`).
  (Float64List values, List<Float64List> hessians) hessians(
    List<int> subjectOf,
    List<Float64List> points, {
    double step = 1e-4,
  }) => runInline((o) => hessianPasses(subjectOf, points, o, step: step));

  /// [hessians] as a program.
  Iterable<Step> hessianPasses(
    List<int> subjectOf,
    List<Float64List> points,
    Out<(Float64List, List<Float64List>)> result, {
    double step = 1e-4,
  }) sync* {
    final k = model.dimension, nP = points.length;
    final per = 2 * k * k + 1;
    final nE = nP * per;
    final subj = Int32List(nE);
    final y = Float64List(nE * k);
    final h = Float64List(nP * k);
    var e = 0;
    void add(
      int j,
      Float64List p, [
      int i = -1,
      double si = 0,
      int l = -1,
      double sl = 0,
    ]) {
      subj[e] = subjectOf[j];
      y.setRange(e * k, (e + 1) * k, p);
      if (i >= 0) y[e * k + i] = p[i] + si * h[j * k + i];
      if (l >= 0) y[e * k + l] = p[l] + sl * h[j * k + l];
      e++;
    }

    for (var j = 0; j < nP; j++) {
      final p = points[j];
      if (p.length != k) throw ArgumentError('Points must have $k entries');
      for (var i = 0; i < k; i++) {
        // Make the steps exactly representable offsets from p.
        final hi = step * math.max(1.0, p[i].abs());
        h[j * k + i] = (p[i] + hi) - p[i];
      }
      add(j, p);
      for (var i = 0; i < k; i++) {
        add(j, p, i, 1);
        add(j, p, i, -1);
      }
      for (var i = 0; i < k; i++) {
        for (var l = i + 1; l < k; l++) {
          add(j, p, i, 1, l, 1);
          add(j, p, i, 1, l, -1);
          add(j, p, i, -1, l, 1);
          add(j, p, i, -1, l, -1);
        }
      }
    }
    final pass = Pass(
      model,
      subjects,
      subj,
      y,
      jacobian: jacobian,
      includePrior: includePrior,
      precise: true,
    );
    yield pass;
    final out = pass.out;
    passes++;
    elements += nE;
    final values = Float64List(nP);
    final hess = <Float64List>[];
    for (var j = 0; j < nP; j++) {
      final base = j * per;
      final f0 = out[base];
      values[j] = f0;
      final hm = Float64List(k * k);
      double fin(double v) => v.isFinite ? v : double.nan;
      var o = base + 1;
      for (var i = 0; i < k; i++) {
        final hi = h[j * k + i];
        hm[i * k + i] = fin((out[o] - 2 * f0 + out[o + 1]) / (hi * hi));
        o += 2;
      }
      for (var i = 0; i < k; i++) {
        for (var l = i + 1; l < k; l++) {
          final v = fin(
            (out[o] - out[o + 1] - out[o + 2] + out[o + 3]) /
                (4 * h[j * k + i] * h[j * k + l]),
          );
          hm[i * k + l] = v;
          hm[l * k + i] = v;
          o += 4;
        }
      }
      if (!f0.isFinite) hm.fillRange(0, k * k, double.nan);
      hess.add(hm);
    }
    result.value = (values, hess);
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
/// the pending points (with their gradients) of every subject's optimiser
/// in one `BatchNetwork` run. [gradients] defaults to exact gradients where
/// the model allows them (see [BatchedObjective]).
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
  Gradients? gradients,
}) => runInline(
  (o) => _fitMap(
    model,
    subjects,
    optimizer,
    start,
    restarts,
    random,
    fdStep,
    gradients,
    1,
    o,
  ),
);

/// [fitMap] with each batched pass split across the pool's isolates.
extension FitMapOnWorkers on PassRunner {
  /// [fitMap] on this pool; the same result, bit for bit.
  Future<MapFit> fitMap(
    HgfModel model,
    List<Subject> subjects, {
    Optimizer optimizer = const Lbfgs(),
    List<List<double>>? start,
    int restarts = 0,
    math.Random? random,
    double fdStep = 1e-5,
    Gradients? gradients,
  }) => run(
    (o) => _fitMap(
      model,
      subjects,
      optimizer,
      start,
      restarts,
      random,
      fdStep,
      gradients,
      streams,
      o,
    ),
  );

  /// [BatchedObjective.evaluate] on this pool.
  Future<(Float64List values, List<Float64List?> gradients)> evaluate(
    BatchedObjective objective,
    List<int> subjectOf,
    List<Float64List> points,
    List<bool> gradient,
  ) => run((o) => objective.evaluatePasses(subjectOf, points, gradient, o));
}

Iterable<Step> _fitMap(
  HgfModel model,
  List<Subject> subjects,
  Optimizer optimizer,
  List<List<double>>? start,
  int restarts,
  math.Random? random,
  double fdStep,
  Gradients? gradients,
  int streams,
  Out<MapFit> result,
) sync* {
  if (start != null && start.length != subjects.length) {
    throw ArgumentError('start must have one vector per subject');
  }
  if (restarts < 0) throw ArgumentError.value(restarts, 'restarts');
  final rng = random ?? math.Random(0);
  final objective = BatchedObjective(
    model,
    subjects,
    fdStep: fdStep,
    gradients: gradients,
  );
  final tasks = <({int subject, Float64List y0})>[];
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
      tasks.add((subject: s, y0: model.toUnconstrained(x0)));
    }
  }
  final minimized = Out<List<OptimizeResult>>();
  yield* minimizeAllPasses(
    objective,
    optimizer,
    tasks,
    minimized,
    streams: streams,
  );
  final results = minimized.value;

  // The best run per subject, then its likelihood and prior parts.
  final best = List<OptimizeResult?>.filled(subjects.length, null);
  for (var t = 0; t < tasks.length; t++) {
    final r = results[t], s = tasks[t].subject;
    final b = best[s];
    if (b == null || r.value < b.value || (b.value.isNaN && !r.value.isNaN)) {
      best[s] = r;
    }
  }
  final k = model.dimension, n = subjects.length;
  final y = Float64List(n * k);
  for (var s = 0; s < n; s++) {
    y.setRange(s * k, (s + 1) * k, best[s]!.x);
  }
  final pass = Pass(
    model,
    subjects,
    Int32List.fromList([for (var s = 0; s < n; s++) s]),
    y,
    negLogLikelihood: true,
    precise: true,
  );
  yield pass;
  final post = pass.out, nll = pass.negLogLikelihood!;
  final names = model.names;
  result.value = MapFit(
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

/// Runs [optimizer] from each task's start `y0` (unconstrained) on its
/// subject's objective, every run advanced in lock step: each round
/// answers all pending requests with one [BatchedObjective.evaluate] pass.
/// Returns one result per task, in order.
List<OptimizeResult> minimizeAll(
  BatchedObjective objective,
  Optimizer optimizer,
  List<({int subject, Float64List y0})> tasks,
) => runInline((o) => minimizeAllPasses(objective, optimizer, tasks, o));

/// [minimizeAll] as a program; with [streams] > 1, the tasks are split
/// into that many [Concurrent] streams (each in lock step).
Iterable<Step> minimizeAllPasses(
  BatchedObjective objective,
  Optimizer optimizer,
  List<({int subject, Float64List y0})> tasks,
  Out<List<OptimizeResult>> result, {
  int streams = 1,
}) sync* {
  if (streams <= 1 || tasks.length < 2) {
    yield* _minimizeAll(objective, optimizer, tasks, result);
    return;
  }
  final groups = splitTasks(tasks.length, streams);
  final outs = [for (final _ in groups) Out<List<OptimizeResult>>()];
  yield Concurrent([
    for (var g = 0; g < groups.length; g++)
      _minimizeAll(objective, optimizer, [
        for (final t in groups[g]) tasks[t],
      ], outs[g]),
  ]);
  final results = List<OptimizeResult?>.filled(tasks.length, null);
  for (var g = 0; g < groups.length; g++) {
    for (var i = 0; i < groups[g].length; i++) {
      results[groups[g][i]] = outs[g].value[i];
    }
  }
  result.value = [for (final r in results) r!];
}

Iterable<Step> _minimizeAll(
  BatchedObjective objective,
  Optimizer optimizer,
  List<({int subject, Float64List y0})> tasks,
  Out<List<OptimizeResult>> result,
) sync* {
  final slots = [for (final _ in tasks) ResultSlot()];
  final its = [
    for (var t = 0; t < tasks.length; t++)
      optimizer.minimize(tasks[t].y0, slots[t]).iterator,
  ];
  var active = [
    for (var t = 0; t < tasks.length; t++)
      if (its[t].moveNext()) t,
  ];
  final evaluated = Out<(Float64List, List<Float64List?>)>();
  while (active.isNotEmpty) {
    final subjectOf = <int>[], points = <Float64List>[], grad = <bool>[];
    for (final t in active) {
      final r = its[t].current;
      for (final p in r.points) {
        subjectOf.add(tasks[t].subject);
        points.add(p);
        grad.add(r.gradient);
      }
    }
    yield* objective.evaluatePasses(subjectOf, points, grad, evaluated);
    final (values, grads) = evaluated.value;
    var j = 0;
    for (final t in active) {
      final r = its[t].current;
      final n = r.points.length;
      r.values = Float64List.sublistView(values, j, j + n);
      if (r.gradient) {
        r.gradients = [for (var i = j; i < j + n; i++) grads[i]!];
      }
      j += n;
    }
    active = [
      for (final t in active)
        if (its[t].moveNext()) t,
    ];
  }
  result.value = [for (final s in slots) s.result!];
}
