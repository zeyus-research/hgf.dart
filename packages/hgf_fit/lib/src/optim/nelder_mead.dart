import 'dart:typed_data';

import 'optimizer.dart';

/// Nelder–Mead simplex search, with Gao & Han's (2012) dimension-adaptive
/// coefficients (as SciPy's `adaptive=True`).
///
/// Gradient-free. By default each iteration asks for the reflection,
/// expansion and both contraction points in one request, so an iteration is
/// one batched pass however it resolves; the unused points cost batch
/// elements, not passes. Set [speculative] to false to evaluate only the
/// points the classic algorithm needs (more passes, fewer elements).
///
/// Stops when every vertex is within [xTolerance] of the best one
/// (max-norm) and their values within [fTolerance].
final class NelderMead extends Optimizer {
  const NelderMead({
    this.initialStep = 0.5,
    this.maxIterations,
    this.xTolerance = 1e-6,
    this.fTolerance = 1e-8,
    this.speculative = true,
  });

  /// Offset of the initial simplex's vertices from the start, along each
  /// unconstrained axis.
  final double initialStep;

  /// Default `200 · dimension`.
  final int? maxIterations;
  final double xTolerance;
  final double fTolerance;
  final bool speculative;

  @override
  bool get usesGradient => false;

  /// An explicit state machine, not a `sync*` body: see [Lbfgs.minimize]
  /// (on the JIT VM, many live `sync*` runs each compile their own OSR
  /// copy of the loop).
  @override
  Iterable<EvalRequest> minimize(Float64List x0, ResultSlot slot) =>
      _NelderMeadIterable(this, x0, slot);
}

final class _NelderMeadIterable extends Iterable<EvalRequest> {
  _NelderMeadIterable(this.options, this.x0, this.slot);

  final NelderMead options;
  final Float64List x0;
  final ResultSlot slot;

  @override
  Iterator<EvalRequest> get iterator => _NelderMeadRun(options, x0, slot);
}

/// One simplex search; [moveNext] consumes the answer to [current] and
/// poses the next request, or ends the run (filling the slot).
final class _NelderMeadRun implements Iterator<EvalRequest> {
  _NelderMeadRun(this.o, Float64List x0, this.slot)
    : n = x0.length,
      // The adaptive coefficients degenerate in one dimension (σ = 0), so
      // it uses the classic ones.
      chi = x0.length > 1 ? 1 + 2 / x0.length : 2.0,
      psi = x0.length > 1 ? 0.75 - 1 / (2 * x0.length) : 0.5,
      sigma = x0.length > 1 ? 1 - 1 / x0.length : 0.5,
      maxIt = o.maxIterations ?? 200 * x0.length,
      xs = <Float64List>[
        Float64List.fromList(x0),
        for (var i = 0; i < x0.length; i++)
          Float64List.fromList(x0)..[i] += o.initialStep,
      ];

  final NelderMead o;
  final ResultSlot slot;
  final int n, maxIt;
  static const double rho = 1.0;
  final double chi, psi, sigma;
  final List<Float64List> xs;
  Float64List fs = Float64List(0);
  int evaluations = 0, it = 0;

  // What the posed request answers.
  static const _start = 0,
      _initial = 1,
      _speculative = 2,
      _reflection = 3,
      _expansion = 4,
      _outside = 5,
      _inside = 6,
      _shrink = 7,
      _done = 8;
  int _state = _start;
  EvalRequest? _req;

  // The current iteration's candidate points and their values.
  Float64List xr = Float64List(0),
      xe = Float64List(0),
      xoc = Float64List(0),
      xic = Float64List(0);
  double fr = double.nan, fe = double.nan, foc = double.nan, fic = double.nan;

  @override
  EvalRequest get current => _req!;

  bool _pose(List<Float64List> points, int state) {
    _req = EvalRequest(points);
    _state = state;
    return true;
  }

  @override
  bool moveNext() {
    final req = _req;
    switch (_state) {
      case _start:
        return _pose(xs, _initial);
      case _initial:
        fs = Float64List.fromList(req!.values);
        evaluations = n + 1;
        return _iterate();
      case _speculative:
        evaluations += 4;
        fr = req!.values[0];
        fe = req.values[1];
        foc = req.values[2];
        fic = req.values[3];
        return _decide();
      case _reflection:
        evaluations++;
        fr = req!.values[0];
        if (fr < fs[0]) return _pose([xe], _expansion);
        if (fr < fs[n - 1]) return _decide();
        if (fr < fs[n]) return _pose([xoc], _outside);
        return _pose([xic], _inside);
      case _expansion:
        evaluations++;
        fe = req!.values[0];
        return _decide();
      case _outside:
        evaluations++;
        foc = req!.values[0];
        return _decide();
      case _inside:
        evaluations++;
        fic = req!.values[0];
        return _decide();
      case _shrink:
        evaluations += n;
        fs.setRange(1, n + 1, req!.values);
        return _iterate();
      default:
        return false;
    }
  }

  Float64List _lerp(Float64List c, Float64List w, double t) {
    // c + t (c − w)
    final out = Float64List(n);
    for (var j = 0; j < n; j++) {
      out[j] = c[j] + t * (c[j] - w[j]);
    }
    return out;
  }

  void _sort() {
    final order = List<int>.generate(n + 1, (i) => i)
      ..sort((a, b) => fs[a].compareTo(fs[b]));
    final xs2 = [for (final i in order) xs[i]];
    final fs2 = Float64List.fromList([for (final i in order) fs[i]]);
    xs.setAll(0, xs2);
    fs = fs2;
  }

  /// The next iteration's request, or the end of the run.
  bool _iterate() {
    if (it >= maxIt) return _finish(false, 'iteration limit reached');
    _sort();
    var dx = 0.0, df = 0.0;
    for (var i = 1; i <= n; i++) {
      final d = (fs[i] - fs[0]).abs();
      if (d > df || d.isNaN) df = d;
      for (var j = 0; j < n; j++) {
        final e = (xs[i][j] - xs[0][j]).abs();
        if (e > dx) dx = e;
      }
    }
    if (dx <= o.xTolerance && df <= o.fTolerance) {
      return _finish(true, 'simplex below tolerance');
    }
    it++;
    final c = Float64List(n);
    for (var i = 0; i < n; i++) {
      for (var j = 0; j < n; j++) {
        c[j] += xs[i][j] / n;
      }
    }
    final w = xs[n];
    xr = _lerp(c, w, rho);
    xe = _lerp(c, w, rho * chi);
    xoc = _lerp(c, w, rho * psi);
    xic = _lerp(c, w, -psi);
    fe = foc = fic = double.nan;
    return o.speculative
        ? _pose([xr, xe, xoc, xic], _speculative)
        : _pose([xr], _reflection);
  }

  /// Accepts a point or shrinks, given the values the branch needs.
  bool _decide() {
    Float64List? accept;
    double fAccept = double.nan;
    if (fr < fs[0]) {
      (accept, fAccept) = fe < fr ? (xe, fe) : (xr, fr);
    } else if (fr < fs[n - 1]) {
      (accept, fAccept) = (xr, fr);
    } else if (fr < fs[n]) {
      if (foc <= fr) (accept, fAccept) = (xoc, foc);
    } else {
      if (fic < fs[n]) (accept, fAccept) = (xic, fic);
    }
    if (accept != null) {
      xs[n] = accept;
      fs[n] = fAccept;
      return _iterate();
    }
    // Shrink towards the best vertex.
    for (var i = 1; i <= n; i++) {
      for (var j = 0; j < n; j++) {
        xs[i][j] = xs[0][j] + sigma * (xs[i][j] - xs[0][j]);
      }
    }
    return _pose(xs.sublist(1), _shrink);
  }

  bool _finish(bool converged, String message) {
    _sort();
    slot.result = OptimizeResult(
      x: xs[0],
      value: fs[0],
      iterations: it,
      evaluations: evaluations,
      converged: converged,
      message: message,
    );
    _state = _done;
    _req = null;
    return false;
  }
}
