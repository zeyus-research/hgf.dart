import 'dart:typed_data';

/// Points an optimiser wants evaluated, answered by its driver.
///
/// Optimisers are written as `sync*` generators that yield requests: the
/// driver collects the pending requests of many optimisers (one per
/// subject), evaluates them all in one batched pass, fills in [values]
/// (and [gradients] if [gradient]), and resumes each generator. This keeps
/// the optimisers ordinary sequential code while the expensive part runs
/// batched.
final class EvalRequest {
  EvalRequest(this.points, {this.gradient = false});

  /// Unconstrained parameter vectors to evaluate.
  final List<Float64List> points;

  /// Whether the gradient at each point is needed too.
  final bool gradient;

  /// The objective at each point (`+∞` where it isn't finite), set by the
  /// driver.
  late Float64List values;

  /// The gradient at each point, set by the driver when [gradient].
  late List<Float64List> gradients;
}

/// The outcome of one minimisation.
final class OptimizeResult {
  OptimizeResult({
    required this.x,
    required this.value,
    required this.iterations,
    required this.evaluations,
    required this.converged,
    required this.message,
    this.gradient,
  });

  /// The minimiser found (unconstrained scale).
  final Float64List x;

  /// The objective there.
  final double value;

  /// The gradient there, if the method uses gradients.
  final Float64List? gradient;

  final int iterations;

  /// Number of points evaluated (excluding finite-difference stencils).
  final int evaluations;

  /// Whether a convergence criterion was met (rather than an iteration
  /// limit or a failed line search).
  final bool converged;

  final String message;

  @override
  String toString() =>
      'OptimizeResult(value: $value, iterations: $iterations, '
      'converged: $converged, $message)';
}

/// Holds an optimiser's result once its generator finishes.
final class ResultSlot {
  OptimizeResult? result;
}

/// A minimiser on the unconstrained scale.
abstract class Optimizer {
  const Optimizer();

  /// Whether the requests ask for gradients.
  bool get usesGradient;

  /// Minimise from [x0], yielding evaluation requests; the result goes to
  /// [slot] when the iterable ends.
  Iterable<EvalRequest> minimize(Float64List x0, ResultSlot slot);
}

/// Minimise a plain function with [optimizer], answering requests directly
/// ([gradient] is required for gradient-based optimisers). Mainly for
/// testing optimisers; for HGF models use `fitMap`.
OptimizeResult minimizeFunction(
  Optimizer optimizer,
  double Function(Float64List x) f,
  Float64List x0, {
  Float64List Function(Float64List x)? gradient,
}) {
  final slot = ResultSlot();
  for (final req in optimizer.minimize(x0, slot)) {
    req.values = Float64List.fromList([
      for (final p in req.points)
        switch (f(p)) {
          final v when v.isNaN => double.infinity,
          final v => v,
        },
    ]);
    if (req.gradient) {
      if (gradient == null) {
        throw ArgumentError('${optimizer.runtimeType} needs a gradient');
      }
      req.gradients = [for (final p in req.points) gradient(p)];
    }
  }
  return slot.result!;
}

double dot(Float64List a, Float64List b) {
  var s = 0.0;
  for (var i = 0; i < a.length; i++) {
    s += a[i] * b[i];
  }
  return s;
}

double maxAbs(Float64List a) {
  var m = 0.0;
  for (final v in a) {
    final x = v.abs();
    if (x > m || x.isNaN) m = x;
  }
  return m;
}
