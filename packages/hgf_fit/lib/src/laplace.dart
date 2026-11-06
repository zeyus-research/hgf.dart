import 'dart:math' as math;
import 'dart:typed_data';

import 'fit.dart';
import 'linalg.dart';
import 'model.dart';
import 'special.dart';

/// The Laplace approximation around one subject's MAP estimate: a Gaussian
/// with the inverse Hessian of the negative log posterior as covariance,
/// and the log evidence (marginal likelihood) it implies.
final class LaplaceResult {
  LaplaceResult._(this.map, this.model, this.hessian, this._chol);

  /// The MAP estimate the approximation is centred on.
  final FitResult map;

  final HgfModel model;

  /// The Hessian of the negative log posterior on the unconstrained scale,
  /// row-major `k x k`.
  final Float64List hessian;

  final Float64List? _chol;

  int get dimension => model.dimension;

  /// Whether [hessian] is positive definite. If not (the mode is on a
  /// ridge, at a bound, or the optimiser stopped short), [covariance] is
  /// null and the derived quantities are NaN.
  bool get positiveDefinite => _chol != null;

  /// The posterior covariance on the unconstrained scale, `H⁻¹`.
  late final Float64List? covariance = switch (_chol) {
    final l? => inverseFromCholesky(l, dimension),
    null => null,
  };

  /// `log det H`.
  double get logDetHessian => switch (_chol) {
    final l? => logDetFromCholesky(l, dimension),
    null => double.nan,
  };

  /// The posterior standard deviations on the unconstrained scale.
  Float64List get unconstrainedStandardErrors {
    final c = covariance, k = dimension;
    return Float64List.fromList([
      for (var i = 0; i < k; i++)
        c == null ? double.nan : math.sqrt(c[i * k + i]),
    ]);
  }

  /// The posterior standard deviations on the natural scale, by the delta
  /// method (`|dx/dy| · sd(y)`).
  Float64List get standardErrors {
    final s = unconstrainedStandardErrors;
    return Float64List.fromList([
      for (var i = 0; i < dimension; i++)
        s[i] * math.exp(model.parameters[i].transform.logAbsJacobian(map.y[i])),
    ]);
  }

  /// The posterior correlation matrix, row-major `k x k`.
  Float64List get correlation {
    final c = covariance, k = dimension;
    final out = Float64List(k * k);
    if (c == null) return out..fillRange(0, k * k, double.nan);
    for (var i = 0; i < k; i++) {
      for (var j = 0; j < k; j++) {
        out[i * k + j] = c[i * k + j] / math.sqrt(c[i * k + i] * c[j * k + j]);
      }
    }
    return out;
  }

  /// Central [level] intervals on the natural scale: `y ± z · sd(y)` mapped
  /// through each parameter's transform, so they stay inside its support
  /// (and are asymmetric around the estimate where the transform bends).
  (Float64List lower, Float64List upper) interval([double level = 0.95]) {
    if (!(level > 0 && level < 1)) {
      throw ArgumentError.value(level, 'level', 'Must be in (0, 1)');
    }
    final z = normalQuantile(0.5 + level / 2);
    final s = unconstrainedStandardErrors;
    final lo = Float64List(dimension), hi = Float64List(dimension);
    for (var i = 0; i < dimension; i++) {
      final t = model.parameters[i].transform;
      final a = t.toConstrained(map.y[i] - z * s[i]);
      final b = t.toConstrained(map.y[i] + z * s[i]);
      lo[i] = math.min(a, b);
      hi[i] = math.max(a, b);
    }
    return (lo, hi);
  }

  /// The Laplace approximation of the log evidence
  /// `log p(responses) = log ∫ p(responses | x) p(x) dx`:
  ///
  /// `−f(x̂) + (k/2) log 2π − ½ log det H_x`,
  ///
  /// with `f` the negative log posterior and `H_x` its Hessian on the
  /// natural scale. At a mode `H_y = J H_x J` (`J = diag(dx/dy)`), so this
  /// is computed as `−f + (k/2) log 2π − ½ log det H_y + Σ log |dxᵢ/dyᵢ|`.
  /// NaN if the Hessian isn't positive definite.
  double get logEvidence {
    if (!positiveDefinite) return double.nan;
    var jac = 0.0;
    for (var i = 0; i < dimension; i++) {
      jac += model.parameters[i].transform.logAbsJacobian(map.y[i]);
    }
    return -map.negLogPosterior +
        dimension / 2 * math.log(2 * math.pi) -
        logDetHessian / 2 +
        jac;
  }

  /// The estimate by parameter name, with its natural-scale standard error.
  Map<String, ({double estimate, double standardError})> get summary {
    final se = standardErrors;
    return {
      for (var i = 0; i < dimension; i++)
        map.names[i]: (estimate: map.x[i], standardError: se[i]),
    };
  }

  @override
  String toString() {
    final se = standardErrors;
    final parts = [
      for (var i = 0; i < dimension; i++)
        '${map.names[i]} = ${map.x[i].toStringAsFixed(4)} '
            '± ${se[i].toStringAsFixed(4)}',
    ];
    return 'LaplaceResult(${map.subject.id ?? ''} ${parts.join(', ')}, '
        'log evidence = ${logEvidence.toStringAsFixed(4)})';
  }
}

/// The Laplace approximations of a [laplace] call.
final class LaplaceFit {
  LaplaceFit(this.results, this.passes, this.elements);

  /// One per subject, in the order of the [MapFit].
  final List<LaplaceResult> results;

  /// Batched passes this step took (one).
  final int passes;

  /// Batch elements evaluated (`2k² + 1` per subject).
  final int elements;

  /// The summed log evidence of all subjects (the model's log evidence for
  /// the whole data set, subjects being independent). NaN if any subject's
  /// Hessian isn't positive definite.
  double get logEvidence => results.fold(0.0, (s, r) => s + r.logEvidence);
}

/// Laplace approximations around the MAP estimates in [fit] (from
/// `fitMap(model, ...)`): the Hessian of every subject's negative log
/// posterior, by finite differences in one batched pass (`2k² + 1`
/// filters per subject), with the covariance, standard errors and log
/// evidence it implies.
///
/// The Hessian is taken on the unconstrained scale with relative step
/// [step] (see `BatchedObjective.hessians`). The approximation assumes the
/// estimate is a mode: check `FitResult.converged`.
LaplaceFit laplace(HgfModel model, MapFit fit, {double step = 1e-4}) {
  final objective = BatchedObjective(model, [
    for (final r in fit.results) r.subject,
  ]);
  final n = fit.results.length, k = model.dimension;
  final (_, hessians) = objective.hessians(
    [for (var s = 0; s < n; s++) s],
    [for (final r in fit.results) r.y],
    step: step,
  );
  return LaplaceFit(
    [
      for (var s = 0; s < n; s++)
        LaplaceResult._(
          fit.results[s],
          model,
          hessians[s],
          cholesky(hessians[s], k),
        ),
    ],
    objective.passes,
    objective.elements,
  );
}
