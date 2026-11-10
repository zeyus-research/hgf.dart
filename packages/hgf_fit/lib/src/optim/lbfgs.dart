import 'dart:math' as math;
import 'dart:typed_data';

import 'optimizer.dart';

/// Limited-memory BFGS with a backtracking (Armijo) line search that
/// interpolates with the gradient at each trial point.
///
/// Every trial point is evaluated with its gradient, so an accepted step
/// needs no extra round: a trial costs one batch element with exact
/// gradients (`2k + 1` with central differences) and one pass. Curvature pairs with
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

  /// The run is an explicit state machine ([_LbfgsRun]) rather than a
  /// `sync*` body: on the JIT VM, a suspended `sync*` frame resumes in the
  /// code it started in, so with many runs alive at once (one per subject
  /// and start) each one compiled its own on-stack-replacement copy of the
  /// loop once the function got hot, 10–25 ms apiece (a 200-subject
  /// `fitMap` took 4.2 s instead of 0.14 s). AOT, dart2wasm and dart2js
  /// have no OSR; the iterates are the same on every target.
  @override
  Iterable<EvalRequest> minimize(Float64List x0, ResultSlot slot) =>
      _LbfgsIterable(this, x0, slot);

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

final class _LbfgsIterable extends Iterable<EvalRequest> {
  _LbfgsIterable(this.options, this.x0, this.slot);

  final Lbfgs options;
  final Float64List x0;
  final ResultSlot slot;

  @override
  Iterator<EvalRequest> get iterator => _LbfgsRun(options, x0, slot);
}

/// One L-BFGS run: [moveNext] consumes the answer to [current] and poses
/// the next request, or ends the run (filling the slot).
final class _LbfgsRun implements Iterator<EvalRequest> {
  _LbfgsRun(this.o, Float64List x0, this.slot)
    : n = x0.length,
      x = Float64List.fromList(x0);

  final Lbfgs o;
  final ResultSlot slot;
  final int n;

  // 0: not started; 1: start point posed; 2: a line-search trial posed;
  // 3: finished.
  int _state = 0;
  EvalRequest? _req;

  Float64List x;
  double f = double.nan;
  Float64List g = Float64List(0);
  int evaluations = 0;
  int it = 0;
  final sHist = <Float64List>[], yHist = <Float64List>[];
  final rho = <double>[];
  bool retried = false;

  // The current line search.
  Float64List d = Float64List(0);
  double gd = 0, alpha = 1;
  int ls = 0;

  @override
  EvalRequest get current => _req!;

  OptimizeResult _result(int it, bool converged, String message) =>
      OptimizeResult(
        x: x,
        value: f,
        gradient: g,
        iterations: it,
        evaluations: evaluations,
        converged: converged,
        message: message,
      );

  bool _finish(int it, bool converged, String message) {
    slot.result = _result(it, converged, message);
    _state = 3;
    _req = null;
    return false;
  }

  @override
  bool moveNext() {
    switch (_state) {
      case 0:
        _req = EvalRequest([x], gradient: true);
        _state = 1;
        return true;
      case 1:
        final req = _req!;
        f = req.values[0];
        g = req.gradients[0];
        evaluations = 1;
        if (!f.isFinite || !Lbfgs._finite(g)) {
          return _finish(0, false, 'objective not finite at the start');
        }
        it = 1;
        return _iteration();
      case 2:
        return _trialAnswered();
      default:
        return false;
    }
  }

  /// Starts iteration [it] (at most [Lbfgs.maxIterations]): poses its first
  /// line-search trial, or finishes.
  bool _iteration() {
    if (it > o.maxIterations) {
      return _finish(o.maxIterations, false, 'iteration limit reached');
    }
    if (maxAbs(g) <= o.gradientTolerance) {
      return _finish(it - 1, true, 'gradient below tolerance');
    }
    d = Lbfgs._direction(g, sHist, yHist, rho);
    gd = dot(g, d);
    if (!(gd < 0)) {
      sHist.clear();
      yHist.clear();
      rho.clear();
      d = Float64List.fromList([for (final v in g) -v]);
      gd = dot(g, d);
    }
    alpha = 1.0;
    if (sHist.isEmpty) {
      var norm = 0.0;
      for (final v in g) {
        norm += v * v;
      }
      alpha = math.min(1.0, 1.0 / math.sqrt(norm));
    }
    ls = 0;
    return _trial();
  }

  /// Poses the line-search trial at [alpha], or gives up on the search.
  bool _trial() {
    if (ls >= o.maxLineSearch) return _searchFailed();
    final trial = Float64List(n);
    for (var i = 0; i < n; i++) {
      trial[i] = x[i] + alpha * d[i];
    }
    _req = EvalRequest([trial], gradient: true);
    _state = 2;
    return true;
  }

  bool _trialAnswered() {
    final req = _req!;
    evaluations++;
    final fa = req.values[0], ga = req.gradients[0];
    if (fa.isFinite && fa <= f + Lbfgs._c1 * alpha * gd && Lbfgs._finite(ga)) {
      return _accept(req.points[0], fa, ga);
    }
    alpha = Lbfgs._backtrack(
      alpha,
      f,
      gd,
      fa,
      Lbfgs._finite(ga) ? dot(ga, d) : null,
    );
    if (alpha * maxAbs(d) < 1e-16 * (1 + maxAbs(x))) return _searchFailed();
    ls++;
    return _trial();
  }

  bool _searchFailed() {
    if (sHist.isNotEmpty && !retried) {
      // The quasi-Newton direction may be stale: retry from scratch.
      sHist.clear();
      yHist.clear();
      rho.clear();
      retried = true;
      it++;
      return _iteration();
    }
    return _finish(it, false, 'line search failed');
  }

  bool _accept(Float64List xNew, double fNew, Float64List gNew) {
    retried = false;
    final s = Float64List(n), y = Float64List(n);
    for (var i = 0; i < n; i++) {
      s[i] = xNew[i] - x[i];
      y[i] = gNew[i] - g[i];
    }
    final sy = dot(s, y);
    if (sy > 1e-10 * math.sqrt(dot(s, s) * dot(y, y))) {
      if (sHist.length == o.memory) {
        sHist.removeAt(0);
        yHist.removeAt(0);
        rho.removeAt(0);
      }
      sHist.add(s);
      yHist.add(y);
      rho.add(1 / sy);
    }
    final fOld = f;
    x = xNew;
    f = fNew;
    g = gNew;
    if (fOld - f <=
        o.functionTolerance * math.max(math.max(fOld.abs(), f.abs()), 1)) {
      return _finish(it, true, 'function change below tolerance');
    }
    it++;
    return _iteration();
  }
}
