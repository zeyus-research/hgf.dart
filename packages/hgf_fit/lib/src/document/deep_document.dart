import 'package:hgf/hgf.dart';
import 'package:hgf_deep/hgf_deep.dart';

/// One HGF neural network of a model document (pyhgf's `DeepNetwork`, from
/// `package:hgf_deep`): its structure, the columns it reads and how it is
/// trained. A document holds any number of them beside (or instead of) its
/// HGF network.
///
/// [layers] are in pyhgf's order: layer 0 is the **target** layer (the
/// observed outputs `y`, or a continuous network's observations), the last
/// is the **feature** layer (the predictors `x`, clamped on the top layer's
/// means). The builder draws them that way too, parents above children.
///
/// Only the structure is stored, not weights or beliefs: [build] gives a
/// fresh network (weights 1, or the identity for one-to-one layers, before
/// [DocDeepTraining.init]).
final class DocDeep {
  DocDeep({
    required this.id,
    this.name,
    this.settings = const DocDeepSettings(),
    required List<DocLayer> layers,
    List<String> features = const [],
    List<String> targets = const [],
    List<String>? categories,
    this.training = const DocDeepTraining(),
  }) : layers = List.unmodifiable(layers),
       features = List.unmodifiable(features),
       targets = List.unmodifiable(targets),
       categories = categories == null ? null : List.unmodifiable(categories);

  /// A small classifier to start from: two features, one hidden layer of 8
  /// tanh nodes, one binary target.
  factory DocDeep.starter(String id) => DocDeep(
    id: id,
    layers: const [
      DocLayer(size: 1, kind: LayerKind.binary, name: 'target'),
      DocLayer(size: 8, coupling: 'tanh', name: 'hidden'),
      DocLayer(size: 2, name: 'features'),
    ],
  );

  /// A document part for [net] (its configuration, not its weights).
  factory DocDeep.fromNetwork(
    DeepNetwork net, {
    required String id,
    String? name,
    List<String> features = const [],
    List<String> targets = const [],
    List<String>? categories,
    DocDeepTraining training = const DocDeepTraining(),
  }) {
    final s = net.settings;
    return DocDeep(
      id: id,
      name: name,
      settings: DocDeepSettings(
        coupling: couplingText(net.coupling),
        volatilityUpdates: s.volatilityUpdates,
        maxPosteriorPrecision: s.maxPosteriorPrecision,
        precisionClippingValue: s.precisionClippingValue,
        predictPrecision: s.predictPrecision,
        feedforwardUncertainty: s.feedforwardUncertainty,
        tonicVolatility: net.tonicVolatility,
        meanFieldUpdates: s.meanFieldUpdates,
        carryPrecision: s.carryPrecision,
        updateInputLayer: s.updateInputLayer,
        fixes: s.fixes,
      ),
      layers: [for (final c in net.configs) DocLayer.fromConfig(c)],
      features: features,
      targets: targets,
      categories: categories,
      training: training,
    );
  }

  factory DocDeep.fromJson(Map<String, Object?> json) => DocDeep(
    id: json['id'] as String,
    name: json['name'] as String?,
    settings: switch (json['settings']) {
      final Map<String, Object?> s => DocDeepSettings.fromJson(s),
      _ => const DocDeepSettings(),
    },
    layers: [
      for (final l in json['layers'] as List<Object?>)
        DocLayer.fromJson(l as Map<String, Object?>),
    ],
    features: _strings(json['features']),
    targets: _strings(json['targets']),
    categories: json['categories'] == null
        ? null
        : _strings(json['categories']),
    training: switch (json['training']) {
      final Map<String, Object?> t => DocDeepTraining.fromJson(t),
      _ => const DocDeepTraining(),
    },
  );

  /// Names the network in the document (a pipeline card, output column
  /// prefixes); unique among the document's deep networks.
  final String id;

  /// A label for the builder (defaults to [id]).
  final String? name;
  final DocDeepSettings settings;

  /// Target layer first, feature layer last (pyhgf's order).
  final List<DocLayer> layers;

  /// The columns read into the feature layer, one per node (layered
  /// networks; a continuous network has none).
  final List<String> features;

  /// The columns read into the target layer, one per node; or, with
  /// [categories], one label column read one-hot (a categorical target).
  final List<String> targets;

  /// The labels of a categorical target layer's classes, in node order,
  /// as written in the [targets] column.
  final List<String>? categories;
  final DocDeepTraining training;

  String get label => name ?? id;

  /// Whether the network is continuous (filtered by `inputData`, not
  /// trained by `fit`).
  bool get isContinuous =>
      layers.isNotEmpty && layers.first.kind == LayerKind.continuous;

  /// The target layer (index 0) and the feature layer (the last).
  DocLayer? get targetLayer => layers.isEmpty ? null : layers.first;
  DocLayer? get featureLayer => layers.isEmpty ? null : layers.last;

  /// Number of learned weights (biases included) of [net], as built.
  static int weightCount(DeepNetwork net) {
    var n = 0;
    for (final w in net.weights) {
      n += w?.data.length ?? 0;
    }
    return n;
  }

  /// A fresh [DeepNetwork]: [settings], then [layers] added target first.
  /// Throws [ArgumentError] (hgf_deep's checks, prefixed by the layer)
  /// when the structure is not allowed.
  DeepNetwork build({Map<String, Coupling> couplings = const {}}) {
    Coupling coupling(String text) => couplings[text] ?? parseCoupling(text);
    final s = settings;
    final net = DeepNetwork(
      coupling: coupling(s.coupling),
      volatilityUpdates: s.volatilityUpdates,
      maxPosteriorPrecision: s.maxPosteriorPrecision,
      precisionClippingValue: s.precisionClippingValue,
      predictPrecision: s.predictPrecision,
      feedforwardUncertainty: s.feedforwardUncertainty,
      tonicVolatility: s.tonicVolatility,
      meanFieldUpdates: s.meanFieldUpdates,
      carryPrecision: s.carryPrecision,
      updateInputLayer: s.updateInputLayer,
      fixes: s.fixes,
    );
    for (final (i, l) in layers.indexed) {
      try {
        net.addLayer(
          l.size,
          kind: l.kind,
          addConstantInput: l.addConstantInput,
          fullyConnected: l.fullyConnected,
          coupling: l.coupling == null ? null : coupling(l.coupling!),
          volatilityParent: l.volatilityParent,
          tonicVolatility: l.tonicVolatility,
          tonicVolatilityVol: l.tonicVolatilityVol,
          initial: l.initial,
          valueChildren: l.valueChildren,
          volatilityChildren: l.volatilityChildren,
          valueCoupling: l.valueCoupling,
          volatilityCoupling: l.volatilityCoupling,
          volatilityFullyConnected: l.volatilityFullyConnected,
          tonicDrift: l.tonicDrift,
          autoconnectionStrength: l.autoconnectionStrength,
        );
      } on ArgumentError catch (e) {
        throw ArgumentError('Layer $i (${layerName(i)}): ${e.message}');
      }
    }
    return net;
  }

  /// The name of layer [i]: its own, else `target`, `features` or
  /// `hidden k` (counted from the features down).
  String layerName(int i) {
    if (layers[i].name case final n?) return n;
    if (isContinuous) return i == 0 ? 'observations' : 'level $i';
    if (i == 0) return 'target';
    if (i == layers.length - 1) return 'features';
    return 'hidden ${layers.length - 1 - i}';
  }

  DocDeep copyWith({
    String? id,
    String? name,
    DocDeepSettings? settings,
    List<DocLayer>? layers,
    List<String>? features,
    List<String>? targets,
    List<String>? Function()? categories,
    DocDeepTraining? training,
  }) => DocDeep(
    id: id ?? this.id,
    name: name ?? this.name,
    settings: settings ?? this.settings,
    layers: layers ?? this.layers,
    features: features ?? this.features,
    targets: targets ?? this.targets,
    categories: categories != null ? categories() : this.categories,
    training: training ?? this.training,
  );

  Map<String, Object?> toJson() => {
    'id': id,
    'name': ?name,
    if (settings.toJson() case final s when s.isNotEmpty) 'settings': s,
    'layers': [for (final l in layers) l.toJson()],
    if (features.isNotEmpty) 'features': features,
    if (targets.isNotEmpty) 'targets': targets,
    'categories': ?categories,
    if (training.toJson() case final t when t.isNotEmpty) 'training': t,
  };
}

List<String> _strings(Object? v) => [
  for (final s in (v as List<Object?>?) ?? const []) s as String,
];

double? _num(Object? v) => switch (v) {
  null => null,
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Not a number: $v'),
};

Object _enc(double v) => v.isFinite
    ? v
    : v.isNaN
    ? 'NaN'
    : v > 0
    ? 'Infinity'
    : '-Infinity';

/// The network-wide settings of a [DocDeep] (`DeepNetwork`'s constructor).
final class DocDeepSettings {
  const DocDeepSettings({
    this.coupling = 'linear',
    this.volatilityUpdates = DeepVolatilityUpdate.unbounded,
    this.maxPosteriorPrecision = 1e10,
    this.precisionClippingValue = 1e-6,
    this.predictPrecision = true,
    this.feedforwardUncertainty = false,
    this.tonicVolatility = false,
    this.meanFieldUpdates = false,
    this.carryPrecision = true,
    this.updateInputLayer = false,
    this.fixes = DeepFixes.recommended,
  });

  factory DocDeepSettings.fromJson(Map<String, Object?> j) {
    const d = DocDeepSettings();
    final fixes = j['fixes'] as Map<String, Object?>?;
    return DocDeepSettings(
      coupling: j['coupling'] as String? ?? d.coupling,
      volatilityUpdates: switch (j['volatility_updates']) {
        final String v => DeepVolatilityUpdate.parse(v),
        _ => d.volatilityUpdates,
      },
      maxPosteriorPrecision:
          _num(j['max_posterior_precision']) ?? d.maxPosteriorPrecision,
      precisionClippingValue:
          _num(j['precision_clipping_value']) ?? d.precisionClippingValue,
      predictPrecision: j['predict_precision'] as bool? ?? d.predictPrecision,
      feedforwardUncertainty:
          j['feedforward_uncertainty'] as bool? ?? d.feedforwardUncertainty,
      tonicVolatility: j['tonic_volatility'] as bool? ?? d.tonicVolatility,
      meanFieldUpdates: j['mean_field_updates'] as bool? ?? d.meanFieldUpdates,
      carryPrecision: j['carry_precision'] as bool? ?? d.carryPrecision,
      updateInputLayer: j['update_input_layer'] as bool? ?? d.updateInputLayer,
      fixes: fixes == null
          ? d.fixes
          : DeepFixes(
              curvatureFloor:
                  fixes['curvature_floor'] as bool? ?? d.fixes.curvatureFloor,
              logSpaceVolatility:
                  fixes['log_space_volatility'] as bool? ??
                  d.fixes.logSpaceVolatility,
            ),
    );
  }

  /// The default coupling of layers that name none: a function name or an
  /// expression in `x` (see `parseCoupling`).
  final String coupling;
  final DeepVolatilityUpdate volatilityUpdates;
  final double maxPosteriorPrecision, precisionClippingValue;
  final bool predictPrecision, feedforwardUncertainty, tonicVolatility;
  final bool meanFieldUpdates, carryPrecision, updateInputLayer;

  /// hgf_deep's corrections to pyhgf (pyhgf itself is [DeepFixes.none]).
  final DeepFixes fixes;

  DocDeepSettings copyWith({
    String? coupling,
    DeepVolatilityUpdate? volatilityUpdates,
    double? maxPosteriorPrecision,
    double? precisionClippingValue,
    bool? predictPrecision,
    bool? feedforwardUncertainty,
    bool? tonicVolatility,
    bool? meanFieldUpdates,
    bool? carryPrecision,
    bool? updateInputLayer,
    DeepFixes? fixes,
  }) => DocDeepSettings(
    coupling: coupling ?? this.coupling,
    volatilityUpdates: volatilityUpdates ?? this.volatilityUpdates,
    maxPosteriorPrecision: maxPosteriorPrecision ?? this.maxPosteriorPrecision,
    precisionClippingValue:
        precisionClippingValue ?? this.precisionClippingValue,
    predictPrecision: predictPrecision ?? this.predictPrecision,
    feedforwardUncertainty:
        feedforwardUncertainty ?? this.feedforwardUncertainty,
    tonicVolatility: tonicVolatility ?? this.tonicVolatility,
    meanFieldUpdates: meanFieldUpdates ?? this.meanFieldUpdates,
    carryPrecision: carryPrecision ?? this.carryPrecision,
    updateInputLayer: updateInputLayer ?? this.updateInputLayer,
    fixes: fixes ?? this.fixes,
  );

  /// Only what differs from the defaults.
  Map<String, Object?> toJson() {
    const d = DocDeepSettings();
    return {
      if (coupling != d.coupling) 'coupling': coupling,
      if (volatilityUpdates != d.volatilityUpdates)
        'volatility_updates': volatilityUpdates.pyhgfName,
      if (maxPosteriorPrecision != d.maxPosteriorPrecision)
        'max_posterior_precision': _enc(maxPosteriorPrecision),
      if (precisionClippingValue != d.precisionClippingValue)
        'precision_clipping_value': _enc(precisionClippingValue),
      if (predictPrecision != d.predictPrecision)
        'predict_precision': predictPrecision,
      if (feedforwardUncertainty != d.feedforwardUncertainty)
        'feedforward_uncertainty': feedforwardUncertainty,
      if (tonicVolatility != d.tonicVolatility)
        'tonic_volatility': tonicVolatility,
      if (meanFieldUpdates != d.meanFieldUpdates)
        'mean_field_updates': meanFieldUpdates,
      if (carryPrecision != d.carryPrecision) 'carry_precision': carryPrecision,
      if (updateInputLayer != d.updateInputLayer)
        'update_input_layer': updateInputLayer,
      if (fixes.curvatureFloor != d.fixes.curvatureFloor ||
          fixes.logSpaceVolatility != d.fixes.logSpaceVolatility)
        'fixes': {
          'curvature_floor': fixes.curvatureFloor,
          'log_space_volatility': fixes.logSpaceVolatility,
        },
    };
  }
}

/// One layer of a [DocDeep]: `DeepNetwork.addLayer`'s arguments. `null`
/// leaves an argument to hgf_deep's (and pyhgf's) default.
final class DocLayer {
  const DocLayer({
    required this.size,
    this.kind = LayerKind.volatile,
    this.name,
    this.addConstantInput,
    this.fullyConnected = true,
    this.coupling,
    this.volatilityParent,
    this.tonicVolatility,
    this.tonicVolatilityVol,
    this.initial = const {},
    this.valueChildren,
    this.volatilityChildren,
    this.valueCoupling,
    this.volatilityCoupling,
    this.volatilityFullyConnected,
    this.tonicDrift,
    this.autoconnectionStrength,
  });

  /// The arguments that rebuild [c] exactly (every resolved value written
  /// out), as `DeepNetwork.fromJson` replays a saved network.
  factory DocLayer.fromConfig(LayerConfig c) {
    final continuous = c.kind == LayerKind.continuous;
    return DocLayer(
      size: c.size,
      kind: c.kind,
      addConstantInput: continuous ? null : c.addConstantInput,
      fullyConnected: c.fullyConnected,
      coupling: couplingText(c.coupling),
      volatilityParent: continuous ? null : c.volatilityParent,
      tonicVolatility: c.tonicVolatility,
      tonicVolatilityVol: continuous ? null : c.tonicVolatilityVol,
      initial: c.initial,
      valueChildren: continuous ? c.valueChild : null,
      volatilityChildren: continuous ? c.volatilityChild : null,
      valueCoupling: continuous ? c.valueCoupling : null,
      volatilityCoupling: continuous ? c.volatilityCoupling : null,
      volatilityFullyConnected: continuous ? c.volatilityFullyConnected : null,
      tonicDrift: continuous ? c.tonicDrift : null,
      autoconnectionStrength: continuous ? c.autoconnectionStrength : null,
    );
  }

  factory DocLayer.fromJson(Map<String, Object?> j) => DocLayer(
    size: j['size'] as int,
    kind: switch (j['kind']) {
      final String k => LayerKind.parse(k),
      _ => LayerKind.volatile,
    },
    name: j['name'] as String?,
    addConstantInput: j['add_constant_input'] as bool?,
    fullyConnected: j['fully_connected'] as bool? ?? true,
    coupling: j['coupling'] as String?,
    volatilityParent: j['volatility_parent'] as bool?,
    tonicVolatility: _num(j['tonic_volatility']),
    tonicVolatilityVol: _num(j['tonic_volatility_vol']),
    initial: {
      for (final MapEntry(:key, :value)
          in ((j['initial'] as Map<String, Object?>?) ?? const {}).entries)
        LayerField.parse(key): _num(value)!,
    },
    valueChildren: j['value_children'] as int?,
    volatilityChildren: j['volatility_children'] as int?,
    valueCoupling: _num(j['value_coupling']),
    volatilityCoupling: _num(j['volatility_coupling']),
    volatilityFullyConnected: j['volatility_fully_connected'] as bool?,
    tonicDrift: _num(j['tonic_drift']),
    autoconnectionStrength: _num(j['autoconnection_strength']),
  );

  final int size;
  final LayerKind kind;

  /// A label for the builder ("hidden", "eyes", ...).
  final String? name;
  final bool? addConstantInput;

  /// `false`: one-to-one (the same size as the layer below).
  final bool fullyConnected;

  /// The activation applied to this layer's means when they predict the
  /// layer below: a function name or an expression in `x`; `null` uses
  /// the network's.
  final String? coupling;
  final bool? volatilityParent;
  final double? tonicVolatility, tonicVolatilityVol;
  final Map<LayerField, double> initial;

  // Continuous layers only.
  final int? valueChildren, volatilityChildren;
  final double? valueCoupling, volatilityCoupling;
  final bool? volatilityFullyConnected;
  final double? tonicDrift, autoconnectionStrength;

  DocLayer copyWith({
    int? size,
    LayerKind? kind,
    String? Function()? name,
    bool? Function()? addConstantInput,
    bool? fullyConnected,
    String? Function()? coupling,
    bool? Function()? volatilityParent,
    double? Function()? tonicVolatility,
    double? Function()? tonicVolatilityVol,
    Map<LayerField, double>? initial,
  }) => DocLayer(
    size: size ?? this.size,
    kind: kind ?? this.kind,
    name: name != null ? name() : this.name,
    addConstantInput: addConstantInput != null
        ? addConstantInput()
        : this.addConstantInput,
    fullyConnected: fullyConnected ?? this.fullyConnected,
    coupling: coupling != null ? coupling() : this.coupling,
    volatilityParent: volatilityParent != null
        ? volatilityParent()
        : this.volatilityParent,
    tonicVolatility: tonicVolatility != null
        ? tonicVolatility()
        : this.tonicVolatility,
    tonicVolatilityVol: tonicVolatilityVol != null
        ? tonicVolatilityVol()
        : this.tonicVolatilityVol,
    initial: initial ?? this.initial,
    valueChildren: valueChildren,
    volatilityChildren: volatilityChildren,
    valueCoupling: valueCoupling,
    volatilityCoupling: volatilityCoupling,
    volatilityFullyConnected: volatilityFullyConnected,
    tonicDrift: tonicDrift,
    autoconnectionStrength: autoconnectionStrength,
  );

  Map<String, Object?> toJson() => {
    'size': size,
    if (kind != LayerKind.volatile) 'kind': kind.pyhgfName,
    'name': ?name,
    'add_constant_input': ?addConstantInput,
    if (!fullyConnected) 'fully_connected': false,
    'coupling': ?coupling,
    'volatility_parent': ?volatilityParent,
    if (tonicVolatility case final v?) 'tonic_volatility': _enc(v),
    if (tonicVolatilityVol case final v?) 'tonic_volatility_vol': _enc(v),
    if (initial.isNotEmpty)
      'initial': {
        for (final MapEntry(:key, :value) in initial.entries)
          key.pyhgfName: _enc(value),
      },
    'value_children': ?valueChildren,
    'volatility_children': ?volatilityChildren,
    if (valueCoupling case final v?) 'value_coupling': _enc(v),
    if (volatilityCoupling case final v?) 'volatility_coupling': _enc(v),
    'volatility_fully_connected': ?volatilityFullyConnected,
    if (tonicDrift case final v?) 'tonic_drift': _enc(v),
    if (autoconnectionStrength case final v?)
      'autoconnection_strength': _enc(v),
  };
}

/// The weight initialisers (`DeepNetwork.weightInitialisation`).
enum DeepInit {
  /// Weights stay as built (1, or the identity for one-to-one layers).
  none('none'),
  xavier('xavier'),
  he('he'),
  orthogonal('orthogonal'),
  sparse('sparse');

  const DeepInit(this.pyhgfName);
  final String pyhgfName;

  static DeepInit parse(String name) => values.firstWhere(
    (v) => v.pyhgfName == name,
    orElse: () => throw FormatException('Unknown weight initialiser "$name"'),
  );

  WeightInit? get strategy => switch (this) {
    none => null,
    xavier => const XavierInit(),
    he => const HeInit(),
    orthogonal => const OrthogonalInit(),
    sparse => const SparseInit(),
  };
}

/// How a [DocDeep] is trained: the optimiser, the learning rule, the
/// initial weights and the passes over the data.
final class DocDeepTraining {
  const DocDeepTraining({
    this.optimiser = 'adam',
    this.learningRate = 0.01,
    this.learningKind = LearningKind.precisionWeighted,
    this.synapticWindow = 20,
    this.init = DeepInit.xavier,
    this.seed = 0,
    this.epochs = 10,
    this.batchSize,
    this.shuffle = true,
    this.standardise = true,
    this.trainOn,
  });

  factory DocDeepTraining.fromJson(Map<String, Object?> j) {
    const d = DocDeepTraining();
    return DocDeepTraining(
      optimiser: j['optimiser'] as String? ?? d.optimiser,
      learningRate: _num(j['learning_rate']) ?? d.learningRate,
      learningKind: switch (j['learning_kind']) {
        final String k => LearningKind.parse(k),
        _ => d.learningKind,
      },
      synapticWindow: _num(j['synaptic_window']) ?? d.synapticWindow,
      init: switch (j['init']) {
        final String i => DeepInit.parse(i),
        _ => d.init,
      },
      seed: j['seed'] as int? ?? d.seed,
      epochs: j['epochs'] as int? ?? d.epochs,
      batchSize: j['batch_size'] as int?,
      shuffle: j['shuffle'] as bool? ?? d.shuffle,
      standardise: j['standardise'] as bool? ?? d.standardise,
      trainOn: switch (j['train_on']) {
        final String e => Expr.parse(e),
        _ => null,
      },
    );
  }

  /// `adam` or `sgd` (ignored by synaptic-uncertainty learning).
  final String optimiser;
  final double learningRate;
  final LearningKind learningKind;

  /// The window of `SynapticUncertainty` learning.
  final double synapticWindow;
  final DeepInit init;
  final int seed;

  /// Passes over the rows.
  final int epochs;

  /// Rows per minibatch (`batchUpdate`); `null`: one row at a time (`fit`).
  final int? batchSize;

  /// Whether each pass visits the rows in a new random order (seeded by
  /// the pass number). Off: in the data's order, which the Dart and Python
  /// exports then share exactly.
  final bool shuffle;

  /// Whether the features are standardised (mean 0, sd 1) first, with the
  /// training rows' means and sds.
  final bool standardise;

  /// The rows trained on: where this is nonzero (e.g. `split == 'train'`,
  /// after a split step); `null`: every row. The others are held out: the
  /// trained network predicts them too, and they are scored apart. A
  /// continuous network does not train and ignores it.
  final Expr? trainOn;

  Optimiser? get optimiserObject =>
      learningKind == LearningKind.synapticUncertainty
      ? null
      : switch (optimiser) {
          'sgd' => Sgd(learningRate),
          'adam' => Adam(learningRate),
          _ => throw FormatException('Unknown optimiser "$optimiser"'),
        };

  DocDeepTraining copyWith({
    String? optimiser,
    double? learningRate,
    LearningKind? learningKind,
    double? synapticWindow,
    DeepInit? init,
    int? seed,
    int? epochs,
    int? Function()? batchSize,
    bool? shuffle,
    bool? standardise,
    Expr? Function()? trainOn,
  }) => DocDeepTraining(
    optimiser: optimiser ?? this.optimiser,
    learningRate: learningRate ?? this.learningRate,
    learningKind: learningKind ?? this.learningKind,
    synapticWindow: synapticWindow ?? this.synapticWindow,
    init: init ?? this.init,
    seed: seed ?? this.seed,
    epochs: epochs ?? this.epochs,
    batchSize: batchSize != null ? batchSize() : this.batchSize,
    shuffle: shuffle ?? this.shuffle,
    standardise: standardise ?? this.standardise,
    trainOn: trainOn != null ? trainOn() : this.trainOn,
  );

  Map<String, Object?> toJson() {
    const d = DocDeepTraining();
    return {
      if (optimiser != d.optimiser) 'optimiser': optimiser,
      if (learningRate != d.learningRate) 'learning_rate': learningRate,
      if (learningKind != d.learningKind)
        'learning_kind': learningKind.pyhgfName,
      if (synapticWindow != d.synapticWindow) 'synaptic_window': synapticWindow,
      if (init != d.init) 'init': init.pyhgfName,
      if (seed != d.seed) 'seed': seed,
      if (epochs != d.epochs) 'epochs': epochs,
      'batch_size': ?batchSize,
      if (shuffle != d.shuffle) 'shuffle': shuffle,
      if (standardise != d.standardise) 'standardise': standardise,
      if (trainOn case final e?) 'train_on': '$e',
    };
  }
}

/// The issues of a [DocDeep]: its structure (hgf_deep's own checks), and
/// its columns against the layer sizes and, given [columns], the data.
/// Each message names the network; [error] and [warn] receive them.
void checkDeep(
  DocDeep d, {
  List<String>? columns,
  required void Function(String message) error,
  required void Function(String message) warn,
}) {
  final who = 'Deep network "${d.label}"';
  if (d.layers.length < 2) {
    error('$who needs at least two layers (targets and features)');
    return;
  }
  try {
    d.build();
  } on ArgumentError catch (e) {
    error('$who: ${e.message}');
    return;
  } on Object catch (e) {
    error('$who: $e');
    return;
  }
  final target = d.targetLayer!, feature = d.featureLayer!;
  if (d.isContinuous) {
    if (d.features.isNotEmpty) {
      warn('$who is continuous: it reads observations only, not features');
    }
  } else if (d.features.isEmpty) {
    warn('$who has no feature columns yet');
  } else if (d.features.length != feature.size) {
    error(
      '$who: ${d.features.length} feature columns for a feature layer of '
      '${feature.size}',
    );
  }
  if (d.categories case final cats?) {
    if (target.kind != LayerKind.categorical) {
      error('$who: target categories need a categorical target layer');
    } else if (cats.length != target.size) {
      error(
        '$who: ${cats.length} categories for a target layer of ${target.size}',
      );
    }
    if (d.targets.length != 1) {
      error('$who: a categorical target is read from one label column');
    }
  } else if (d.targets.isEmpty) {
    warn('$who has no target columns yet');
  } else if (d.targets.length != target.size) {
    error(
      '$who: ${d.targets.length} target columns for a target layer of '
      '${target.size}',
    );
  }
  if (columns != null) {
    final have = columns.toSet();
    for (final c in {...d.features, ...d.targets}) {
      if (!have.contains(c)) error('$who reads "$c", which is not a column');
    }
  }
}
