import 'dart:math' as math;

import 'package:hgf_fit/hgf_fit.dart';

/// A custom prior: a Laplace distribution, decoded in the worker through
/// [testCodec].
final class LaplacePrior extends Prior {
  const LaplacePrior(this.location, this.scale);

  final double location, scale;

  @override
  double logDensity(double x) =>
      -(x - location).abs() / scale - math.log(2 * scale);

  @override
  double sample(math.Random rng) {
    final u = rng.nextDouble() - 0.5;
    return location - scale * u.sign * math.log(1 - 2 * u.abs());
  }

  @override
  double get typical => location;

  @override
  Map<String, Object?> toJson() => {
    'type': 'Laplace',
    'location': location,
    'scale': scale,
  };
}

/// A prior that throws once evaluated in a worker (its decoder makes it
/// throw), to check errors reach the caller.
final class ThrowsInWorker extends Prior {
  const ThrowsInWorker({this.armed = false});

  final bool armed;

  @override
  double logDensity(double x) =>
      armed ? throw StateError('boom in the worker') : -0.5 * x * x;

  @override
  double sample(math.Random rng) => 0;

  @override
  double get typical => -4;

  @override
  Map<String, Object?> toJson() => {'type': 'ThrowsInWorker'};
}

/// A prior the worker has no decoder for.
final class UnknownToWorker extends Prior {
  const UnknownToWorker();

  @override
  double logDensity(double x) => -0.5 * x * x;

  @override
  double sample(math.Random rng) => 0;

  @override
  double get typical => -4;

  @override
  Map<String, Object?> toJson() => {'type': 'UnknownToWorker'};
}

const testCodec = FitCodec(
  priors: {'Laplace': _laplace, 'ThrowsInWorker': _throws},
);

Prior _laplace(Map<String, Object?> json, FitCodec codec) => LaplacePrior(
  (json['location'] as num).toDouble(),
  (json['scale'] as num).toDouble(),
);

Prior _throws(Map<String, Object?> json, FitCodec codec) =>
    const ThrowsInWorker(armed: true);
