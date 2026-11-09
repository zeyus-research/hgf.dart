import 'dart:math' as math;
import 'dart:typed_data';

import 'matrix.dart';

/// A gradient-descent optimiser over a network's weight matrices, with the
/// arithmetic of the optax transform pyhgf uses (`optax.sgd`,
/// `optax.adam`).
///
/// An optimiser keeps its own state (Adam's moments and step count), so
/// passing the same instance to successive [DeepNetwork.fit] calls continues
/// where the last one stopped; a new instance starts afresh.
sealed class Optimiser {
  const Optimiser();

  /// Apply one step to [weights] from the descent gradients [grads]
  /// (matched by index; `null` entries are skipped).
  void step(List<Matrix?> weights, List<Float64List?> grads);

  /// Forget the state (the next step starts afresh).
  void reset();
}

/// Plain SGD: `w ← w + (−lr · g)` (`optax.sgd(lr)`).
final class Sgd extends Optimiser {
  const Sgd(this.learningRate);

  final double learningRate;

  @override
  void step(List<Matrix?> weights, List<Float64List?> grads) {
    final scale = -learningRate;
    for (var l = 0; l < weights.length; l++) {
      final w = weights[l];
      final g = grads[l];
      if (w == null || g == null) continue;
      final d = w.data;
      for (var k = 0; k < d.length; k++) {
        d[k] = d[k] + scale * g[k];
      }
    }
  }

  @override
  void reset() {}
}

/// Adam (Kingma & Ba, 2015), as `optax.adam(lr, b1, b2, eps, eps_root)`.
final class Adam extends Optimiser {
  Adam(
    this.learningRate, {
    this.b1 = 0.9,
    this.b2 = 0.999,
    this.eps = 1e-8,
    this.epsRoot = 0.0,
  });

  final double learningRate;
  final double b1;
  final double b2;
  final double eps;
  final double epsRoot;

  /// Steps taken so far.
  int get count => _count;
  int _count = 0;
  List<Float64List?> _mu = const [];
  List<Float64List?> _nu = const [];

  @override
  @pragma('vm:unsafe:no-bounds-checks')
  void step(List<Matrix?> weights, List<Float64List?> grads) {
    if (!_matches(weights)) {
      _mu = [
        for (final w in weights) w == null ? null : Float64List(w.data.length),
      ];
      _nu = [
        for (final w in weights) w == null ? null : Float64List(w.data.length),
      ];
      _count = 0;
    }
    _count++;
    final bc1 = 1 - math.pow(b1, _count).toDouble();
    final bc2 = 1 - math.pow(b2, _count).toDouble();
    for (var l = 0; l < weights.length; l++) {
      final w = weights[l];
      final g = grads[l];
      if (w == null || g == null) continue;
      _adamLayer(w.data, g, _mu[l]!, _nu[l]!, bc1, bc2);
    }
  }

  @pragma('vm:unsafe:no-bounds-checks')
  void _adamLayer(
    Float64List d,
    Float64List g,
    Float64List mu,
    Float64List nu,
    double bc1,
    double bc2,
  ) {
    final n = d.length;
    if (g.length < n || mu.length < n || nu.length < n) {
      throw ArgumentError('A gradient is shorter than its weights.');
    }
    final b1 = this.b1, b2 = this.b2, eps = this.eps, epsRoot = this.epsRoot;
    final c1 = 1 - b1, c2 = 1 - b2;
    final scale = -learningRate;
    for (var k = 0; k < n; k++) {
      final gk = g[k];
      final m = c1 * gk + b1 * mu[k];
      final v = c2 * (gk * gk) + b2 * nu[k];
      mu[k] = m;
      nu[k] = v;
      final update = (m / bc1) / (math.sqrt(v / bc2 + epsRoot) + eps);
      d[k] = d[k] + scale * update;
    }
  }

  bool _matches(List<Matrix?> weights) {
    if (_mu.length != weights.length) return false;
    for (var l = 0; l < weights.length; l++) {
      final w = weights[l];
      if ((w == null) != (_mu[l] == null)) return false;
      if (w != null && w.data.length != _mu[l]!.length) return false;
    }
    return true;
  }

  @override
  void reset() {
    _mu = const [];
    _nu = const [];
    _count = 0;
  }
}
