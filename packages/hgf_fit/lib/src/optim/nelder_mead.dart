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

  @override
  Iterable<EvalRequest> minimize(Float64List x0, ResultSlot slot) sync* {
    final n = x0.length;
    // The adaptive coefficients degenerate in one dimension (σ = 0), so it
    // uses the classic ones.
    final rho = 1.0;
    final chi = n > 1 ? 1 + 2 / n : 2.0;
    final psi = n > 1 ? 0.75 - 1 / (2 * n) : 0.5;
    final sigma = n > 1 ? 1 - 1 / n : 0.5;
    final maxIt = maxIterations ?? 200 * n;
    final xs = <Float64List>[
      Float64List.fromList(x0),
      for (var i = 0; i < n; i++) Float64List.fromList(x0)..[i] += initialStep,
    ];
    var req = EvalRequest(xs);
    yield req;
    var fs = Float64List.fromList(req.values);
    var evaluations = n + 1;

    Float64List lerp(Float64List c, Float64List w, double t) {
      // c + t (c − w)
      final out = Float64List(n);
      for (var j = 0; j < n; j++) {
        out[j] = c[j] + t * (c[j] - w[j]);
      }
      return out;
    }

    void sort() {
      final order = List<int>.generate(n + 1, (i) => i)
        ..sort((a, b) => fs[a].compareTo(fs[b]));
      final xs2 = [for (final i in order) xs[i]];
      final fs2 = Float64List.fromList([for (final i in order) fs[i]]);
      xs.setAll(0, xs2);
      fs = fs2;
    }

    var it = 0;
    var converged = false;
    var message = 'iteration limit reached';
    while (it < maxIt) {
      sort();
      var dx = 0.0, df = 0.0;
      for (var i = 1; i <= n; i++) {
        final d = (fs[i] - fs[0]).abs();
        if (d > df || d.isNaN) df = d;
        for (var j = 0; j < n; j++) {
          final e = (xs[i][j] - xs[0][j]).abs();
          if (e > dx) dx = e;
        }
      }
      if (dx <= xTolerance && df <= fTolerance) {
        converged = true;
        message = 'simplex below tolerance';
        break;
      }
      it++;
      final c = Float64List(n);
      for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
          c[j] += xs[i][j] / n;
        }
      }
      final w = xs[n];
      final xr = lerp(c, w, rho);
      final xe = lerp(c, w, rho * chi);
      final xoc = lerp(c, w, rho * psi);
      final xic = lerp(c, w, -psi);
      double fr, fe = double.nan, foc = double.nan, fic = double.nan;
      if (speculative) {
        req = EvalRequest([xr, xe, xoc, xic]);
        yield req;
        evaluations += 4;
        fr = req.values[0];
        fe = req.values[1];
        foc = req.values[2];
        fic = req.values[3];
      } else {
        req = EvalRequest([xr]);
        yield req;
        evaluations++;
        fr = req.values[0];
      }
      Float64List? accept;
      double fAccept = double.nan;
      if (fr < fs[0]) {
        if (!speculative) {
          req = EvalRequest([xe]);
          yield req;
          evaluations++;
          fe = req.values[0];
        }
        (accept, fAccept) = fe < fr ? (xe, fe) : (xr, fr);
      } else if (fr < fs[n - 1]) {
        (accept, fAccept) = (xr, fr);
      } else if (fr < fs[n]) {
        if (!speculative) {
          req = EvalRequest([xoc]);
          yield req;
          evaluations++;
          foc = req.values[0];
        }
        if (foc <= fr) (accept, fAccept) = (xoc, foc);
      } else {
        if (!speculative) {
          req = EvalRequest([xic]);
          yield req;
          evaluations++;
          fic = req.values[0];
        }
        if (fic < fs[n]) (accept, fAccept) = (xic, fic);
      }
      if (accept != null) {
        xs[n] = accept;
        fs[n] = fAccept;
        continue;
      }
      // Shrink towards the best vertex.
      for (var i = 1; i <= n; i++) {
        for (var j = 0; j < n; j++) {
          xs[i][j] = xs[0][j] + sigma * (xs[i][j] - xs[0][j]);
        }
      }
      req = EvalRequest(xs.sublist(1));
      yield req;
      evaluations += n;
      fs.setRange(1, n + 1, req.values);
    }
    sort();
    slot.result = OptimizeResult(
      x: xs[0],
      value: fs[0],
      iterations: it,
      evaluations: evaluations,
      converged: converged,
      message: message,
    );
  }
}
