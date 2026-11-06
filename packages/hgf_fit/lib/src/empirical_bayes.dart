import 'dart:math' as math;
import 'dart:typed_data';

import 'codec.dart' show jsonDouble;
import 'fit.dart';
import 'hierarchical.dart';
import 'linalg.dart';
import 'model.dart';
import 'optim/lbfgs.dart';
import 'optim/optimizer.dart';
import 'pass.dart';
import 'prior.dart';
import 'special.dart';
import 'transform.dart';
import 'workers.dart';

const double _logSqrt2Pi = 0.9189385332046727;

/// `N(μ, σ²)` on a parameter's unconstrained scale `y = T(x)`, as a prior
/// for an objective that adds the transform's log Jacobian (so the two
/// together give exactly `−log N(y | μ, σ²)`).
final class _PopulationPrior extends Prior {
  const _PopulationPrior(this.transform, this.mu, this.sd);

  final Transform transform;
  final double mu, sd;

  @override
  double logDensity(double x) {
    final y = transform.toUnconstrained(x), z = (y - mu) / sd;
    return -0.5 * z * z -
        math.log(sd) -
        _logSqrt2Pi -
        transform.logAbsJacobian(y);
  }

  @override
  double dLogDensity(double x) {
    final y = transform.toUnconstrained(x);
    return (-(y - mu) / (sd * sd) - transform.dLogAbsJacobian(y)) /
        transform.dToConstrained(y);
  }

  @override
  double sample(math.Random rng) =>
      transform.toConstrained(mu + sd * standardNormal(rng));

  @override
  double get typical => transform.toConstrained(mu);

  @override
  String toString() => 'N($mu, $sd) on the ${transform.runtimeType} scale';

  @override
  Map<String, Object?> toJson() => {
    'type': 'Population',
    'transform': transform.toJson(),
    'mu': jsonDouble(mu),
    'sd': jsonDouble(sd),
  };
}

/// `N(mu, sd²)` on [transform]'s unconstrained scale as a prior (what
/// `fitEmpiricalBayes` refits with). Package-internal, for `FitCodec`.
Prior populationPrior(Transform transform, double mu, double sd) =>
    _PopulationPrior(transform, mu, sd);

/// One subject's estimate in an [EmpiricalBayesFit]: the mode of its
/// posterior on the unconstrained scale under the fitted population, with
/// the Laplace approximation around it.
final class SubjectEstimate {
  SubjectEstimate._(
    this.subject,
    this.names,
    this.y,
    this.x,
    this.covariance,
    this.negLogPosterior,
    this.logEvidence,
    this.optimization,
  );

  final Subject subject;
  final List<String> names;

  /// The mode on the unconstrained scale and the same point on the natural
  /// scale.
  final Float64List y, x;

  /// The Laplace covariance on the unconstrained scale (`k x k`
  /// row-major), or null if the Hessian wasn't positive definite.
  final Float64List? covariance;

  /// The objective at the mode (density on the unconstrained scale).
  final double negLogPosterior;

  /// The subject's Laplace log evidence under the fitted population (NaN
  /// without a positive definite Hessian).
  final double logEvidence;

  final OptimizeResult optimization;

  /// Posterior sds on the unconstrained scale.
  Float64List get unconstrainedStandardErrors {
    final c = covariance, k = names.length;
    return Float64List.fromList([
      for (var i = 0; i < k; i++)
        c == null ? double.nan : math.sqrt(c[i * k + i]),
    ]);
  }

  /// The estimate by parameter name (natural scale).
  Map<String, double> get parameters => {
    for (var i = 0; i < names.length; i++) names[i]: x[i],
  };

  @override
  String toString() => 'SubjectEstimate(${subject.id ?? ''} $parameters)';
}

/// The result of [fitEmpiricalBayes].
final class EmpiricalBayesFit {
  EmpiricalBayesFit._(
    this.model,
    this.populations,
    this.subjects,
    this.logEvidence,
    this.history,
    this.converged,
    this.passes,
    this.elements,
  );

  final HierarchicalModel model;

  /// The fitted population of each pooled parameter: mean and sd on the
  /// parameter's unconstrained scale.
  final Map<String, ({double mean, double sd})> populations;

  /// One per subject, in order.
  final List<SubjectEstimate> subjects;

  /// The Laplace approximation of `log p(data | populations)` (summed over
  /// subjects), which EM maximises.
  final double logEvidence;

  /// [logEvidence] after each iteration.
  final List<double> history;

  int get iterations => history.length;

  /// Whether the change in [logEvidence] fell below the tolerance.
  final bool converged;

  /// Batched passes and batch elements over all iterations.
  final int passes, elements;

  @override
  String toString() {
    final b = StringBuffer(
      'EmpiricalBayesFit($iterations iterations, '
      '${converged ? 'converged' : 'not converged'}, log evidence '
      '${logEvidence.toStringAsFixed(3)})\n',
    );
    for (final MapEntry(:key, :value) in populations.entries) {
      b.writeln(
        '  $key: N(${value.mean.toStringAsFixed(4)}, '
        '${value.sd.toStringAsFixed(4)}²)',
      );
    }
    return b.toString();
  }
}

/// Empirical Bayes for a [HierarchicalModel]: point estimates of the
/// populations (mean and sd of each pooled parameter) that maximise the
/// marginal likelihood, by expectation-maximisation with the Laplace
/// approximation (as in Huys et al. 2011, or the "MAP-EM" of many
/// computational psychiatry toolboxes). Much cheaper than
/// [sampleHierarchical], and often enough for group summaries and
/// individual estimates with shrinkage; the hyperpriors are not used
/// (type-II maximum likelihood), and the group-level uncertainty isn't
/// quantified.
///
/// Each iteration fits every subject's mode under the current populations
/// (on the unconstrained scale, every subject's optimiser in lock step,
/// warm-started from the last iteration), takes the Laplace covariance `Σ`
/// there (one Hessian pass), and sets each population to the moments of
/// the subjects' approximate posteriors: `μ = mean(ŷ)`,
/// `σ² = mean((ŷ − μ)² + Σᵢᵢ)` (at least [minSd]²). It stops when the log
/// evidence changes by less than [tolerance] nats.
///
/// Independent parameters keep their priors. Shared parameters aren't
/// supported (use [sampleHierarchical]). Starting populations come from
/// [start] (natural-scale vectors per subject, e.g. `fitMap` estimates) as
/// in [HierarchicalModel.initialPoint], otherwise from the parameters'
/// starting values with sd 1.
EmpiricalBayesFit fitEmpiricalBayes(
  HierarchicalModel model,
  List<Subject> subjects, {
  Optimizer optimizer = const Lbfgs(),
  List<List<double>>? start,
  int maxIterations = 200,
  double tolerance = 1e-4,
  double minSd = 1e-3,
  Gradients? gradients,
}) => runInline(
  (o) => _fitEmpiricalBayes(
    model,
    subjects,
    optimizer,
    start,
    maxIterations,
    tolerance,
    minSd,
    gradients,
    1,
    o,
  ),
);

/// [fitEmpiricalBayes] with each batched pass split across the pool's
/// isolates.
extension EmpiricalBayesOnWorkers on FitWorkers {
  /// [fitEmpiricalBayes] on this pool; the same result, bit for bit.
  Future<EmpiricalBayesFit> fitEmpiricalBayes(
    HierarchicalModel model,
    List<Subject> subjects, {
    Optimizer optimizer = const Lbfgs(),
    List<List<double>>? start,
    int maxIterations = 200,
    double tolerance = 1e-4,
    double minSd = 1e-3,
    Gradients? gradients,
  }) => run(
    (o) => _fitEmpiricalBayes(
      model,
      subjects,
      optimizer,
      start,
      maxIterations,
      tolerance,
      minSd,
      gradients,
      streams,
      o,
    ),
  );
}

Iterable<Step> _fitEmpiricalBayes(
  HierarchicalModel model,
  List<Subject> subjects,
  Optimizer optimizer,
  List<List<double>>? start,
  int maxIterations,
  double tolerance,
  double minSd,
  Gradients? gradients,
  int streams,
  Out<EmpiricalBayesFit> result,
) sync* {
  if (model.shared.isNotEmpty) {
    throw UnsupportedError(
      'Shared parameters need sampleHierarchical, not empirical Bayes',
    );
  }
  if (subjects.isEmpty) throw ArgumentError('No subjects');
  final base = model.model, k = base.dimension, n = subjects.length;
  final names = base.names;
  final pooled = [
    for (var i = 0; i < k; i++)
      if (model.pooled.containsKey(names[i])) i,
  ];

  // Starting populations and subject points.
  final theta0 = model.initialPoint(n, start: start);
  final x0 = model.toConstrained(theta0);
  final mu = Float64List(pooled.length), sd = Float64List(pooled.length);
  for (var p = 0; p < pooled.length; p++) {
    mu[p] = x0[2 * p];
    sd[p] = x0[2 * p + 1];
  }
  final ys = [
    for (var s = 0; s < n; s++)
      base.toUnconstrained([
        for (var i = 0; i < k; i++) x0[model.indexOf(names[i], subject: s)],
      ]),
  ];

  var passes = 0, elements = 0;
  final history = <double>[];
  var converged = false;
  late List<OptimizeResult> results;
  late Float64List values;
  late List<Float64List?> covs;
  late List<double> evidence;
  for (var it = 0; it < maxIterations; it++) {
    final m = base.withPriors({
      for (var p = 0; p < pooled.length; p++)
        names[pooled[p]]: _PopulationPrior(
          base.parameters[pooled[p]].transform,
          mu[p],
          sd[p],
        ),
    });
    final objective = BatchedObjective(
      m,
      subjects,
      jacobian: true,
      gradients: gradients,
    );
    final minimized = Out<List<OptimizeResult>>();
    yield* minimizeAllPasses(
      objective,
      optimizer,
      [for (var s = 0; s < n; s++) (subject: s, y0: ys[s])],
      minimized,
      streams: streams,
    );
    results = minimized.value;
    for (var s = 0; s < n; s++) {
      ys[s] = Float64List.fromList(results[s].x);
    }
    final hess = Out<(Float64List, List<Float64List>)>();
    yield* objective.hessianPasses([for (var s = 0; s < n; s++) s], ys, hess);
    final (v, hessians) = hess.value;
    values = v;
    passes += objective.passes;
    elements += objective.elements;

    // E-step summaries and the evidence.
    covs = List.filled(n, null);
    evidence = List.filled(n, double.nan);
    var total = 0.0;
    for (var s = 0; s < n; s++) {
      final l = cholesky(hessians[s], k);
      if (l == null) continue;
      covs[s] = inverseFromCholesky(l, k);
      evidence[s] =
          -v[s] + k * math.log(2 * math.pi) / 2 - logDetFromCholesky(l, k) / 2;
      total += evidence[s];
    }
    history.add(total);

    // M-step.
    for (var p = 0; p < pooled.length; p++) {
      final i = pooled[p];
      var m1 = 0.0;
      for (var s = 0; s < n; s++) {
        m1 += ys[s][i];
      }
      m1 /= n;
      var m2 = 0.0;
      for (var s = 0; s < n; s++) {
        final d = ys[s][i] - m1;
        // Without a usable Hessian the subject adds no posterior variance.
        m2 += d * d + (covs[s]?[i * k + i] ?? 0.0);
      }
      mu[p] = m1;
      sd[p] = math.max(math.sqrt(m2 / n), minSd);
    }

    final h = history.length;
    if (h >= 2 && (history[h - 1] - history[h - 2]).abs() < tolerance) {
      converged = true;
      break;
    }
  }

  result.value = EmpiricalBayesFit._(
    model,
    {
      for (var p = 0; p < pooled.length; p++)
        names[pooled[p]]: (mean: mu[p], sd: sd[p]),
    },
    [
      for (var s = 0; s < n; s++)
        SubjectEstimate._(
          subjects[s],
          names,
          ys[s],
          base.toConstrained(ys[s]),
          covs[s],
          values[s],
          evidence[s],
          results[s],
        ),
    ],
    history.last,
    history,
    converged,
    passes,
    elements,
  );
}
