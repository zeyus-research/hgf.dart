import 'package:hgf/hgf.dart';

/// The kind of nodes a layer holds.
enum LayerKind {
  /// Volatile state nodes: a value level plus an internal volatility level.
  volatile('volatile'),

  /// Binary (Bernoulli) state nodes, one independent belief per node.
  binary('binary'),

  /// One joint N-way choice over the layer's nodes (softmax). Only valid as
  /// the output (bottom) layer.
  categorical('categorical');

  const LayerKind(this.pyhgfName);

  /// The name pyhgf uses.
  final String pyhgfName;

  /// Parse a pyhgf layer kind (`volatile`/`volatile-state`, and so on).
  static LayerKind parse(String name) => switch (name) {
    'volatile' || 'volatile-state' => volatile,
    'binary' || 'binary-state' => binary,
    'categorical' || 'categorical-state' => categorical,
    _ => throw ArgumentError.value(
      name,
      'name',
      'Unknown layer kind. Choose from volatile, binary, categorical',
    ),
  };
}

/// The volatility-level posterior update applied network-wide.
enum DeepVolatilityUpdate {
  /// Precision first, then mean.
  standard('standard'),

  /// Mean first (expected precision), then the safe precision increment.
  ehgf('eHGF'),

  /// Two quadratic expansions blended by a variational softmax, with
  /// Gaussian moment matching (the uHGF update; pyhgf's default).
  unbounded('unbounded');

  const DeepVolatilityUpdate(this.pyhgfName);

  /// The name pyhgf uses.
  final String pyhgfName;

  /// Parse a pyhgf name (`standard`, `eHGF`, `unbounded`).
  static DeepVolatilityUpdate parse(String name) => switch (name) {
    'standard' => standard,
    'eHGF' || 'ehgf' => ehgf,
    'unbounded' => unbounded,
    _ => throw ArgumentError.value(
      name,
      'name',
      'Unknown volatility update. Choose from unbounded, eHGF, standard',
    ),
  };
}

/// Which weight gradient the learning step descends.
enum LearningKind {
  /// The raw squared-error gradient, `δ ⊗ g(μ_parent)`.
  standard('standard'),

  /// The free-energy gradient, `δ·π_child ⊗ g(μ_parent)` (binary children
  /// stay unweighted). pyhgf's default.
  precisionWeighted('precision_weighted');

  const LearningKind(this.pyhgfName);

  /// The name pyhgf uses.
  final String pyhgfName;

  /// Parse a pyhgf name.
  static LearningKind parse(String name) => switch (name) {
    'standard' => standard,
    'precision_weighted' => precisionWeighted,
    _ => throw ArgumentError.value(
      name,
      'name',
      'Unknown learning kind. Choose from standard, precision_weighted',
    ),
  };
}

/// A belief field of a layer, for initial-state overrides and recording.
enum LayerField {
  mean('mean'),
  precision('precision'),
  expectedMean('expected_mean'),
  expectedPrecision('expected_precision'),
  conditionalExpectedPrecision('conditional_expected_precision'),
  effectivePrecision('effective_precision'),
  valuePredictionError('value_prediction_error'),
  meanVol('mean_vol', volatility: true),
  precisionVol('precision_vol', volatility: true),
  expectedMeanVol('expected_mean_vol', volatility: true),
  expectedPrecisionVol('expected_precision_vol', volatility: true),
  effectivePrecisionVol('effective_precision_vol', volatility: true),
  volatilityPredictionError('volatility_prediction_error', volatility: true);

  const LayerField(this.pyhgfName, {this.volatility = false});

  /// The `LayerState` field name in pyhgf.
  final String pyhgfName;

  /// Whether the field belongs to the internal volatility level (absent on
  /// layers without a volatility parent).
  final bool volatility;

  /// Parse a pyhgf `LayerState` field name.
  static LayerField parse(String name) {
    for (final f in values) {
      if (f.pyhgfName == name) return f;
    }
    throw ArgumentError.value(name, 'name', 'Unknown LayerState field');
  }
}

/// The description of one layer, as given to [DeepNetwork.addLayer].
final class LayerConfig {
  const LayerConfig({
    required this.size,
    required this.kind,
    required this.addConstantInput,
    required this.fullyConnected,
    required this.coupling,
    required this.volatilityParent,
    required this.tonicVolatility,
    required this.tonicVolatilityVol,
    required this.initial,
  });

  /// Number of nodes.
  final int size;

  final LayerKind kind;

  /// Whether a bias column is appended to this layer's incoming weights
  /// (a constant parent with activation 1).
  final bool addConstantInput;

  /// Dense (`true`) or one-to-one diagonal (`false`) weights to the layer
  /// below.
  final bool fullyConnected;

  /// The coupling (activation) function applied to this layer's means when
  /// it predicts the layer below.
  final Coupling coupling;

  /// Whether the layer has an internal volatility level.
  final bool volatilityParent;

  /// Value-level tonic volatility ω, or `null` when the layer has none.
  final double? tonicVolatility;

  /// Tonic volatility of the volatility level.
  final double tonicVolatilityVol;

  /// Initial-belief overrides, broadcast to every node.
  final Map<LayerField, double> initial;
}

/// Corrections to upstream pyhgf behaviour that hgf_deep applies on top of
/// the port (see `docs/upstream/README.md`). [DeepFixes.recommended] is the
/// default of every [DeepNetwork]; [DeepFixes.none] reproduces pyhgf bit for
/// bit (the parity tests use it).
final class DeepFixes {
  const DeepFixes({
    this.curvatureFloor = false,
    this.logSpaceVolatility = false,
  });

  /// Every correction off: pyhgf's behaviour, NaN included.
  static const none = DeepFixes();

  /// Every correction on.
  static const recommended = DeepFixes(
    curvatureFloor: true,
    logSpaceVolatility: true,
  );

  /// P18: the value-level posterior precision of a hidden node is
  /// `π̂ + Σ w²·γ·g′² − g″·msg`. pyhgf clips it at π̂ from below, so when
  /// the curvature term `−g″·msg` cancels the information the precision
  /// falls back to the (possibly tiny) prior while the mean still moves by
  /// `msg·g′/π`: steps of 10³–10¹⁵ that feed the volatility level and run
  /// it away. With this fix the floor is `π̂ + Σ w²·γ·g′²` (negative
  /// curvature is dropped, a Gauss–Newton step), so the step never exceeds
  /// the one the information alone would make.
  final bool curvatureFloor;

  /// P18: evaluate the volatility terms where they overflow, without
  /// changing their value elsewhere.
  ///
  /// * The eHGF and unbounded volatility posteriors read the known variance
  ///   of the value-level prediction (the carried `1/π`), stored at
  ///   prediction time, instead of reconstructing it as
  ///   `max(1/π̄ − Ω, 1e-128)`, which cancels catastrophically once
  ///   Ω ≫ 1/π and is ∞ − ∞ once Ω overflows.
  /// * The eHGF re-prediction uses `r/(p + r) = σ(ln r − ln p)`,
  ///   `(r − p)/(p + r) = 2σ(ln r − ln p) − 1` and
  ///   `1/(p + r) = exp(−logaddexp(ln p, ln r))`.
  /// * The effective precision of the value-level prediction is
  ///   `γ = 1/(1 + a/Ω)` (`= Ω/(a + Ω)`), whose limit as Ω overflows is 1.
  ///
  /// A node whose volatility has run away then has finite beliefs (a
  /// vanishing predicted precision) and drops out of its parents' updates,
  /// instead of turning NaN and poisoning the layers above it.
  final bool logSpaceVolatility;

  bool get isNone => !curvatureFloor && !logSpaceVolatility;

  @override
  bool operator ==(Object other) =>
      other is DeepFixes &&
      other.curvatureFloor == curvatureFloor &&
      other.logSpaceVolatility == logSpaceVolatility;

  @override
  int get hashCode => Object.hash(curvatureFloor, logSpaceVolatility);

  @override
  String toString() =>
      'DeepFixes(curvatureFloor: $curvatureFloor, '
      'logSpaceVolatility: $logSpaceVolatility)';
}
