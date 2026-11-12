/// The kinds of node a [Network] can hold. Names match pyhgf's `kind` strings.
enum NodeKind {
  continuousState('continuous-state'),
  binaryState('binary-state'),

  /// A fixed bias (pyhgf's `constant-state`): `mean = expectedMean = 1` and
  /// `expectedPrecision = ∞`, never updated. A value parent only, always with
  /// linear coupling, so a child's drift gains `α · 1` and no variance.
  constantState('constant-state'),

  /// An exponential-family state node (pyhgf's `ef-state`): an input node
  /// that tracks the expected sufficient statistics `xis` of a (multivariate)
  /// normal distribution of its observations. Add it with
  /// `Network.addEfState`; see [EfLearning] for how its learning rate is set.
  efState('ef-state'),

  /// A categorical state node (pyhgf's `categorical-state`): an input node
  /// that observes one of `n` categories (one-hot, `n` input columns) and
  /// tracks the probability of each with its own binary HGF, plus a
  /// Dirichlet distribution ([VectorField.alpha]) implied by how much those
  /// HGFs learn. Add it with `Network.addCategoricalState`.
  categoricalState('categorical-state'),

  /// A Dirichlet process node (pyhgf's `dp-state`): an input node that
  /// assigns each observation to one of a fixed number of clusters, or opens
  /// a new one (a Chinese restaurant process with concentration
  /// [NodeField.alpha]). Each cluster is a univariate ef-state value parent
  /// that only learns from the observations assigned to it. Add it with
  /// `Network.addDpState`, then its clusters with
  /// `Network.addEfState(valueChildren: [dp])`.
  dpState('dp-state');

  const NodeKind(this.pyhgfName);

  /// The pyhgf `kind` string (e.g. `"continuous-state"`).
  final String pyhgfName;

  static NodeKind parse(String name) {
    for (final k in values) {
      if (k.pyhgfName == name) return k;
    }
    throw UnsupportedError(
      'Node kind "$name" is not supported yet. Supported: '
      '${values.map((v) => v.pyhgfName).join(', ')}.',
    );
  }
}

/// Which volatility-coupling posterior update a node with volatility children
/// uses (pyhgf's `volatility_updates`).
enum VolatilityUpdates {
  /// The original HGF update.
  standard('standard'),

  /// The enhanced HGF: mean first, then a precision update floored at zero.
  eHGF('eHGF'),

  /// The unbounded approximation (pyhgf's default; Mathys et al., 2026).
  unbounded('unbounded');

  const VolatilityUpdates(this.pyhgfName);
  final String pyhgfName;

  static VolatilityUpdates parse(String name) {
    for (final v in values) {
      if (v.pyhgfName == name) return v;
    }
    throw ArgumentError.value(name, 'name', 'Unknown volatility update');
  }
}

/// A scalar node attribute. Names match pyhgf's attribute keys.
enum NodeField {
  mean('mean'),
  expectedMean('expected_mean'),
  precision('precision'),
  expectedPrecision('expected_precision'),
  conditionalExpectedPrecision('conditional_expected_precision'),
  observed('observed'),
  tonicVolatility('tonic_volatility'),
  tonicDrift('tonic_drift'),
  autoconnectionStrength('autoconnection_strength'),
  currentVariance('current_variance'),
  effectivePrecision('effective_precision'),
  valuePredictionError('value_prediction_error'),
  volatilityPredictionError('volatility_prediction_error'),

  /// The learning-rate parameter ν of an ef-state node: `xis` moves by
  /// `1 / (1 + ν)` of its prediction error. Fixed, or implied by the
  /// continuous nodes tracking the sufficient statistics (see [EfLearning]).
  nus('nus'),

  /// A categorical-state node's Bayesian surprise: the KL divergence from
  /// the Dirichlet distribution before the last observation to the one
  /// after it ([VectorField.alpha]).
  klDivergence('kl_divergence'),

  /// A categorical-state node's surprise at the last observation: the sum
  /// over categories of the `binarySurprise` of each category's indicator
  /// under its binary node's prediction. (Not the surprise that
  /// `Trajectories.nodeSurprise` computes, which pyhgf leaves out for this
  /// kind.)
  surprise('surprise'),

  /// A dp-state node's concentration parameter α: the prior weight of a new
  /// cluster against the existing ones (`α / (α + N)` against `n_k / (α + N)`).
  alpha('alpha'),

  /// A dp-state node's sensory precision: the exponent of the observation's
  /// likelihood when weighing the new-cluster proposals.
  sensoryPrecision('sensory_precision'),

  /// A dp-state node's number of observations so far (`N`).
  nTotal('n_total'),

  /// A dp-state node's number of clusters opened so far. Keeps counting
  /// past the number of clusters the node has; openings beyond it are
  /// dropped (as in pyhgf).
  nActiveCluster('n_active_cluster');

  const NodeField(this.pyhgfName);

  /// The pyhgf attribute key (e.g. `"expected_precision"`).
  final String pyhgfName;

  static NodeField parse(String name) {
    for (final f in values) {
      if (f.pyhgfName == name) return f;
    }
    throw ArgumentError.value(name, 'name', 'Unknown node field');
  }
}

/// A vector-valued node attribute, held outside the scalar columns (one
/// variable-length vector per node that has it; other nodes have length 0).
/// Names match pyhgf's attribute keys.
enum VectorField {
  /// An ef-state node's observation (length `dimension`). For
  /// `dimension == 1` the scalar [NodeField.mean] holds the same value.
  ///
  /// A categorical-state node's observation (length `n`, one indicator per
  /// category) until its posterior step, which replaces it with each
  /// category's updated probability, `sigmoid` of the binary HGF's value
  /// parent (as pyhgf does). The scalar [NodeField.mean] mirrors the first
  /// entry.
  mean('mean'),

  /// An ef-state node's expected sufficient statistics ξ: for dimension `d`,
  /// `[x₁..x_d, (x xᵀ) lower triangle row by row]`, `d + d(d+1)/2` values.
  xis('xis'),

  /// The sufficient statistics of the last observation (same layout as
  /// [xis]).
  observationSs('observation_ss'),

  /// A categorical-state node's Dirichlet concentration parameters α, one
  /// per category (initially all 1).
  alpha('alpha'),

  /// A dp-state node's number of observations assigned to each cluster
  /// (pyhgf's `n`).
  n('n'),

  /// A dp-state node's expected mean of each cluster, from its ef-state
  /// parent's `xis` (the prediction step).
  expectedMeans('expected_means'),

  /// A dp-state node's expected standard deviation of each cluster.
  expectedSigmas('expected_sigmas'),

  /// A dp-state node's `1` for each cluster opened so far, `0` otherwise.
  activated('activated');

  const VectorField(this.pyhgfName);

  /// The pyhgf attribute key.
  final String pyhgfName;

  static VectorField? tryParse(String name) {
    for (final f in values) {
      if (f.pyhgfName == name) return f;
    }
    return null;
  }
}

/// How an ef-state node sets its learning rate ν ([NodeField.nus]). Names
/// match pyhgf's `learning` strings.
enum EfLearning {
  /// ν is a fixed parameter: `ξ ← ξ + (t(x) − ξ) / (1 + ν)`.
  fixed('generalised-filtering', 0),

  /// Each sufficient statistic is tracked by a continuous node with one
  /// value parent; ν is implied by how much those parents learn (Mathys &
  /// Weber, 2020).
  hgf1('hgf-1', 1),

  /// As [hgf1], plus a volatility parent on each value parent.
  hgf2('hgf-2', 2),

  /// As [hgf2], plus a volatility parent on each volatility parent.
  hgf3('hgf-3', 3);

  const EfLearning(this.pyhgfName, this.levels);

  /// The pyhgf `learning` string.
  final String pyhgfName;

  /// Levels of implied continuous nodes above each sufficient statistic's
  /// own node (0 for [fixed]).
  final int levels;

  /// Parse a pyhgf `learning` string: as in pyhgf, any string containing
  /// `hgf` selects HGF learning, with `-2`/`-3` for the deeper variants.
  static EfLearning parse(String name) {
    if (!name.contains('hgf')) return fixed;
    if (name.contains('-3')) return hgf3;
    if (name.contains('-2')) return hgf2;
    return hgf1;
  }
}

/// The fields recorded by default in [Trajectories].
const List<NodeField> defaultRecordedFields = [
  NodeField.mean,
  NodeField.expectedMean,
  NodeField.precision,
  NodeField.expectedPrecision,
  NodeField.valuePredictionError,
  NodeField.volatilityPredictionError,
  NodeField.effectivePrecision,
];

/// One step of a belief-propagation sweep. [rshgfName] is the name rshgf
/// reports for the same step, which the parity tests compare against.
enum UpdateStep {
  predictionContinuous(
    'prediction_continuous_state_node',
    'continuous_node_prediction',
  ),
  predictionContinuousMeanField(
    'prediction_continuous_state_node_mean_field',
    'continuous_node_prediction_mean_field',
  ),
  predictionBinary(
    'prediction_binary_state_node',
    'binary_state_node_prediction',
  ),
  posteriorContinuous(
    'posterior_update_continuous_state_node',
    'continuous_node_posterior_update',
  ),
  posteriorContinuousMeanField(
    'posterior_update_continuous_state_node_mean_field',
    'continuous_node_posterior_update_mean_field',
  ),
  posteriorContinuousEhgf(
    'posterior_update_continuous_state_node_ehgf',
    'continuous_node_posterior_update_ehgf',
  ),
  posteriorContinuousEhgfMeanField(
    'posterior_update_continuous_state_node_ehgf_mean_field',
    'continuous_node_posterior_update_ehgf_mean_field',
  ),
  posteriorContinuousUnbounded(
    'posterior_update_continuous_state_node_unbounded',
    'continuous_node_posterior_update_unbounded',
  ),
  predictionErrorContinuous(
    'prediction_error_continuous_state_node',
    'continuous_node_prediction_error',
  ),
  predictionErrorBinary(
    'prediction_error_binary_state_node',
    'binary_state_node_prediction_error',
  ),

  /// An ef-state node without parents: ξ moves toward t(x) at rate
  /// `1 / (1 + ν)`.
  predictionErrorEfFixed(
    'prediction_error_exponential_state_node',
    'prediction_error_update_exponential_family_fixed',
  ),

  /// An ef-state node with HGF learning: t(x) goes to the value parents that
  /// track each sufficient statistic.
  predictionErrorEfDynamic(
    'prediction_error_update_exponential_family_dynamic',
    'prediction_error_update_exponential_family_dynamic',
  ),

  /// An ef-state node with HGF learning, at the end of the sweep: ν from
  /// the parents' learning, then the ξ update.
  posteriorEfDynamic(
    'posterior_update_exponential_family_dynamic',
    'posterior_update_exponential_family_dynamic',
  ),

  /// A categorical-state node passes each category's indicator (and its
  /// `observed` flag) to that category's binary node.
  predictionErrorCategorical(
    'categorical_state_prediction_error',
    'categorical_state_prediction_error',
  ),

  /// A categorical-state node, at the end of the sweep: the implied
  /// Dirichlet α from how much each binary HGF learned, its KL divergence
  /// and the surprise.
  posteriorCategorical('categorical_state_update', 'categorical_state_update'),

  /// A dp-state node reads each cluster's expected mean and standard
  /// deviation from its ef-state parents.
  predictionDp('dirichlet_node_prediction', 'dirichlet_node_prediction'),

  /// A dp-state node assigns the observation to the most likely existing
  /// cluster or opens a new one, and passes it (and `observed` flags) to the
  /// clusters' ef-state nodes, whose fixed-rate update runs next.
  predictionErrorDp(
    'dirichlet_node_prediction_error',
    'dirichlet_node_prediction_error',
  ),

  /// An input node with noise parents (continuous volatility parents of a
  /// leaf): the noise prediction error ε = π̂_u (Σ_b (t α_b g_b'(μ_b))² / π_b
  /// + (u − m_u)²) − 1 (Eq. 72 of Weber et al., 2026), from the posteriors
  /// of its value parents, so after their updates and before those of its
  /// noise parents. pyhgf #477.
  noisePredictionErrorContinuous(
    'noise_prediction_error_continuous_input_node',
    'continuous_node_noise_prediction_error',
  );

  const UpdateStep(this.rshgfName, this.pyhgfName);

  /// The name rshgf reports for this step (the JAX name where rshgf has no
  /// such step).
  final String rshgfName;

  /// The name of pyhgf's JAX update function for this step.
  final String pyhgfName;
}

/// The ordered steps of one belief-propagation sweep: predictions (roots to
/// leaves), then prediction errors and posteriors (leaves to roots).
final class UpdateSequence {
  const UpdateSequence(this.predictions, this.updates);

  final List<(int node, UpdateStep step)> predictions;
  final List<(int node, UpdateStep step)> updates;

  @override
  String toString() =>
      'UpdateSequence(predictions: $predictions, updates: $updates)';
}

/// Corrections of pyhgf bugs that hgf.dart applies, each listed in
/// `docs/upstream/README.md`. [Network.defaultFixes] ([recommended] unless
/// the program is started with `-DHGF_UPSTREAM=true`) is what a network gets
/// without `fixes:`; [none] reproduces pyhgf, as the parity tests do, until
/// the fixes reach pyhgf.
final class NetworkFixes {
  const NetworkFixes({this.constantNoiseVariance = false});

  /// Every correction off: pyhgf's behaviour.
  static const none = NetworkFixes();

  /// Every correction on.
  static const recommended = NetworkFixes(constantNoiseVariance: true);

  /// P21: the noise prediction error of an input with noise parents
  /// (pyhgf #477) adds the posterior variance of each value parent,
  /// `(t α g'(μ_b))² / π_b`. pyhgf reads it from a constant-state (bias)
  /// parent too, whose `precision` is a placeholder 1, so each bias parent
  /// adds `(t α)²` of variance a constant does not have. With this fix a
  /// bias parent adds none, as with the equivalent tonic drift.
  final bool constantNoiseVariance;

  /// As JSON (`Network.toJson`, [NetworkSpec]'s `"fixes"`).
  Map<String, Object?> toJson() => {
    'constant_noise_variance': constantNoiseVariance,
  };

  /// From [toJson]'s output; missing keys are off.
  factory NetworkFixes.fromJson(Map<String, Object?> json) => NetworkFixes(
    constantNoiseVariance: (json['constant_noise_variance'] as bool?) ?? false,
  );

  @override
  bool operator ==(Object other) =>
      other is NetworkFixes &&
      other.constantNoiseVariance == constantNoiseVariance;

  @override
  int get hashCode => constantNoiseVariance.hashCode;

  @override
  String toString() =>
      'NetworkFixes(constantNoiseVariance: $constantNoiseVariance)';
}
