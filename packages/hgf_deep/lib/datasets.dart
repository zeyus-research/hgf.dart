/// Toy datasets, splitting, scaling and scores for [DeepNetwork] examples
/// and the playground. Pure Dart (no `dart:io`), seeded and deterministic.
///
/// ```dart
/// import 'package:hgf_deep/datasets.dart';
///
/// final (:train, :test) = split(moons(400, noise: 0.15), testFraction: 0.25);
/// ```
library;

export 'src/csv.dart';
export 'src/datasets.dart';
export 'src/metrics.dart';
