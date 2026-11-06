import 'dart:math' as math;
import 'dart:typed_data';

import 'optimizer.dart';

/// Limited-memory BFGS with a backtracking (Armijo) line search that
/// interpolates with the gradient at each trial point.
///
/// Every trial point is evaluated with its gradient, so an accepted step
/// needs no extra round; with finite-difference gradients a trial costs
/// `2k + 1` batch elements but one pass. Curvature pairs with
/// `sᵀy ≤ 1e-10 ‖s‖‖y‖` are skipped, which keeps the update positive
/// definite without a Wolfe line search.
///
/// Stops when `max |gᵢ| ≤ gradientTolerance`, or when an accepted step
/// lowers the objective by at most `functionTolerance · max(|f|, 1)`
/// (SciPy's L-BFGS-B defaults).
final class Lbfgs extends Optimizer {
  const Lbfgs({
    this.memory = 10,
    this.maxIterations = 500,
    this.gradientTolerance = 1e-5,
    this.functionTolerance = 2.220446049250313e-9,
    this.maxLineSearch = 30,
  });

  /// Number of curvature pairs kept.
  final int memory;
  final int maxIterations;
  final double gradientTolerance;
  final double functionTolerance;

  /// Trial points per line search before giving up.
  final int maxLineSearch;

  static const double _c1 = 1e-4;

  @override
  bool get usesGradient => true;

  @override
  Iterable<EvalRequest> minimize(Float64List x0, ResultSlot slot) sync* {
    final n = x0.length;
    var x = Float64List.fromList(x0);
    var req = EvalRequest([x], gradient: true);
    yield req;
    var f = req.values[0];
    var g = req.gradients[0];
    var evaluations = 1;

    OptimizeResult result(int it, bool converged, String message) =>
        OptimizeResult(
          x: x,
          value: f,
          gradient: g,
          iterations: it,
          evaluations: evaluations,
          converged: converged,
          message: message,
        );

    if (!f.isFinite || !_finite(g)) {
      slot.result = result(0, false, 'objective not finite at the start');
      return;
    }
    final sHist = <Float64List>[], yHist = <Float64List>[];
    final rho = <double>[];
    var retried = false;
    for (var it = 1; it <= maxIterations; it++) {
      if (maxAbs(g) <= gradientTolerance) {
        slot.result = result(it - 1, true, 'gradient below tolerance');
        return;
      }
      var d = _direction(g, sHist, yHist, rho);
      var gd = dot(g, d);
      if (!(gd < 0)) {
        sHist.clear();
        yHist.clear();
        rho.clear();
        d = Float64List.fromList([for (final v in g) -v]);
        gd = dot(g, d);
      }
      var alpha = 1.0;
      if (sHist.isEmpty) {
        var norm = 0.0;
        for (final v in g) {
          norm += v * v;
        }
        alpha = math.min(1.0, 1.0 / math.sqrt(norm));
      }

      Float64List? xNew, gNew;
      var fNew = double.infinity;
      var accepted = false;
      for (var ls = 0; ls < maxLineSearch; ls++) {
        final trial = Float64List(n);
        for (var i = 0; i < n; i++) {
          trial[i] = x[i] + alpha * d[i];
        }
        req = EvalRequest([trial], gradient: true);
        yield req;
        evaluations++;
        final fa = req.values[0], ga = req.gradients[0];
        if (fa.isFinite && fa <= f + _c1 * alpha * gd && _finite(ga)) {
          xNew = trial;
          fNew = fa;
          gNew = ga;
          accepted = true;
          break;
        }
        alpha = _backtrack(alpha, f, gd, fa, _finite(ga) ? dot(ga, d) : null);
        if (alpha * maxAbs(d) < 1e-16 * (1 + maxAbs(x))) break;
      }
      if (!accepted) {
        if (sHist.isNotEmpty && !retried) {
          // The quasi-Newton direction may be stale: retry from scratch.
          sHist.clear();
          yHist.clear();
          rho.clear();
          retried = true;
          continue;
        }
        slot.result = result(it, false, 'line search failed');
        return;
      }
      retried = false;
      final s = Float64List(n), y = Float64List(n);
      for (var i = 0; i < n; i++) {
        s[i] = xNew![i] - x[i];
        y[i] = gNew![i] - g[i];
      }
      final sy = dot(s, y);
      if (sy > 1e-10 * math.sqrt(dot(s, s) * dot(y, y))) {
        if (sHist.length == memory) {
          sHist.removeAt(0);
          yHist.removeAt(0);
          rho.removeAt(0);
        }
        sHist.add(s);
        yHist.add(y);
        rho.add(1 / sy);
      }
      final fOld = f;
      x = xNew!;
      f = fNew;
      g = gNew!;
      if (fOld - f <=
          functionTolerance * math.max(math.max(fOld.abs(), f.abs()), 1)) {
        slot.result = result(it, true, 'function change below tolerance');
        return;
      }
    }
    slot.result = result(maxIterations, false, 'iteration limit reached');
  }

  /// `−H g` by the two-loop recursion, with `H₀ = (sᵀy / yᵀy) I`.
  static Float64List _direction(
    Float64List g,
    List<Float64List> s,
    List<Float64List> y,
    List<double> rho,
  ) {
    final m = s.length;
    final q = Float64List.fromList(g);
    final a = Float64List(m);
    for (var i = m - 1; i >= 0; i--) {
      a[i] = rho[i] * dot(s[i], q);
      for (var j = 0; j < q.length; j++) {
        q[j] -= a[i] * y[i][j];
      }
    }
    if (m > 0) {
      final gamma = dot(s[m - 1], y[m - 1]) / dot(y[m - 1], y[m - 1]);
      for (var j = 0; j < q.length; j++) {
        q[j] *= gamma;
      }
    }
    for (var i = 0; i < m; i++) {
      final b = rho[i] * dot(y[i], q);
      for (var j = 0; j < q.length; j++) {
        q[j] += s[i][j] * (a[i] - b);
      }
    }
    for (var j = 0; j < q.length; j++) {
      q[j] = -q[j];
    }
    return q;
  }

  /// A smaller step after a rejected one: the minimiser of the cubic (or,
  /// without the slope at the trial, the quadratic) through `φ(0) = f`,
  /// `φ'(0) = gd`, `φ(α) = fa`, `φ'(α) = ga`, kept within `[0.1α, 0.5α]`.
  static double _backtrack(
    double alpha,
    double f,
    double gd,
    double fa,
    double? ga,
  ) {
    final lo = 0.1 * alpha, hi = 0.5 * alpha;
    if (!fa.isFinite) return 0.25 * alpha;
    double next = double.nan;
    if (ga != null) {
      final d1 = gd + ga - 3 * (fa - f) / alpha;
      final disc = d1 * d1 - gd * ga;
      if (disc >= 0) {
        final d2 = math.sqrt(disc);
        next = alpha - alpha * (ga + d2 - d1) / (ga - gd + 2 * d2);
      }
    }
    if (!next.isFinite) {
      next = -gd * alpha * alpha / (2 * (fa - f - gd * alpha));
    }
    if (!next.isFinite) return hi;
    return next.clamp(lo, hi);
  }

  static bool _finite(Float64List v) {
    for (final x in v) {
      if (!x.isFinite) return false;
    }
    return true;
  }
}
