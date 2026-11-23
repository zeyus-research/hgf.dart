import 'dart:math' as math;

import 'package:hgf/hgf.dart';

import '../codec.dart';
import '../prior.dart';
import 'edit.dart';
import 'model_document.dart';
import 'model_edit.dart';
import 'network_document.dart';

/// A free parameter's target as document JSON (`FitCodec` target JSON with
/// node addresses): [nodeTarget], [couplingTarget] or [responseTarget].
typedef DocTarget = Map<String, Object?>;

/// The initial value of [field] at the node [address].
DocTarget nodeTarget(String address, NodeField field) => {
  'type': 'node',
  'node': address,
  'field': field.pyhgfName,
};

/// The strength of the [type] edge from [parent] to [child].
DocTarget couplingTarget(EdgeType type, String parent, String child) => {
  'type': type == EdgeType.value ? 'value_coupling' : 'volatility_coupling',
  'parent': parent,
  'child': child,
};

/// The response model's parameter [name] (a key of
/// `ResponseModel.parameters`, e.g. `inverseTemperature`, `bias_1`).
DocTarget responseTarget(String name) => {'type': 'response', 'name': name};

/// Whether [a] and [b] are the same target.
bool sameTarget(DocTarget a, DocTarget b) =>
    a['type'] == b['type'] &&
    a['node'] == b['node'] &&
    a['field'] == b['field'] &&
    a['parent'] == b['parent'] &&
    a['child'] == b['child'] &&
    a['name'] == b['name'];

/// [p]'s targets.
List<DocTarget> targetsOf(Map<String, Object?> p) => [
  for (final t in (p['targets'] as List<Object?>?) ?? const []) t! as DocTarget,
];

/// A short default name for a parameter of [target]: `omega_x2`, `mu_x1`,
/// `kappa_x2_x1` (a volatility coupling), `w_x1_x0` (a value coupling),
/// `beta` (a softmax's inverse temperature), …
String defaultParameterName(DocTarget target) {
  String id(Object? a) => '$a'.replaceAll('.', '_');
  return switch (target['type']) {
    'node' => '${_symbol(target['field']! as String)}_${id(target['node'])}',
    'volatility_coupling' =>
      'kappa_${id(target['parent'])}_${id(target['child'])}',
    'value_coupling' => 'w_${id(target['parent'])}_${id(target['child'])}',
    _ => switch (target['name']) {
      'inverseTemperature' => 'beta',
      'precision' => 'response_precision',
      final n => '$n',
    },
  };
}

String _symbol(String field) => switch (field) {
  'tonic_volatility' => 'omega',
  'mean' => 'mu',
  'precision' => 'pi',
  'expected_mean' => 'muhat',
  'expected_precision' => 'pihat',
  'tonic_drift' => 'rho',
  'autoconnection_strength' => 'lambda',
  'sensory_precision' => 'pi_s',
  'nus' => 'nu',
  final f => f,
};

/// A default prior for [target] around its [current] value:
/// - tonic volatility ω: `N(ω, 2²)` truncated to [−16, 4];
/// - means, drifts, value couplings: normal around the value;
/// - precisions, α, ν, volatility couplings, the inverse temperature and
///   a Gaussian readout's precision: log-normal with the value as median;
/// - the autoconnection strength, a lapse rate and a choice-kernel rate:
///   truncated to [0, 1];
/// - stickiness, a choice-kernel weight and option biases: `N(value, 1²)`,
///   `N(value, 2²)` and `N(value, 2²)`.
Prior suggestPrior(DocTarget target, double current) {
  final c = current.isFinite ? current : 0.0;
  double logOf(double v) => v > 0 ? _r(math.log(v)) : 0;
  switch (target['type']) {
    case 'volatility_coupling':
      return LogNormal(logOf(c), 0.5);
    case 'value_coupling':
      return Normal(_r(c), 1);
    case 'response':
      final name = target['name']! as String;
      return switch (name) {
        'inverseTemperature' => LogNormal(logOf(c), 1),
        'precision' => LogNormal(logOf(c), 2),
        'stickiness' => Normal(_r(c), 1),
        'choiceKernel' => Normal(_r(c), 2),
        'choiceKernelRate' => TruncatedNormal(
          c > 0 && c < 1 ? _r(c) : 0.3,
          0.3,
          lower: 0,
          upper: 1,
        ),
        'lapse' => TruncatedNormal(
          c > 0 && c < 1 ? _r(c) : 0.05,
          0.1,
          lower: 0,
          upper: 1,
        ),
        _ when name.startsWith('bias_') => Normal(_r(c), 2),
        _ => Normal(_r(c), math.max(1, _r(c.abs()))),
      };
  }
  return switch (target['field']) {
    'tonic_volatility' => TruncatedNormal(
      _r(c.clamp(-15.0, 3.0)),
      2,
      lower: -16,
      upper: 4,
    ),
    'precision' ||
    'expected_precision' ||
    'alpha' ||
    'sensory_precision' ||
    'nus' => LogNormal(logOf(c), 2),
    'autoconnection_strength' => TruncatedNormal(
      _r(c.clamp(0.0, 1.0)),
      0.25,
      lower: 0,
      upper: 1,
    ),
    _ => Normal(_r(c), 2),
  };
}

double _r(double v) => (v * 1000).roundToDouble() / 1000;

/// A parameter (document JSON) named [name] that sets [targets], with
/// [prior] and its default transform.
Map<String, Object?> docParameter(
  String name,
  List<DocTarget> targets,
  Prior prior,
) => {
  'name': name,
  'targets': targets,
  'prior': prior.toJson(),
  'transform': prior.defaultTransform.toJson(),
};

/// The free parameters of a [ModelDocument], as the model builder edits
/// them: by name, with their targets by address, and their roles in the
/// group level ([ModelDocument.group]).
extension ModelDocumentParameters on ModelDocument {
  List<String> get parameterNames => [
    for (final p in parameters) p['name']! as String,
  ];

  /// The parameter named [name], or null.
  Map<String, Object?>? parameter(String name) {
    for (final p in parameters) {
      if (p['name'] == name) return p;
    }
    return null;
  }

  /// The parameter that sets [target], or null.
  Map<String, Object?>? parameterFor(DocTarget target) {
    for (final p in parameters) {
      if (targetsOf(p).any((t) => sameTarget(t, target))) return p;
    }
    return null;
  }

  /// [base], or `base_1`, `base_2`, … : a name no parameter has.
  String freshParameterName(String base) {
    final used = parameterNames.toSet();
    if (!used.contains(base)) return base;
    for (var k = 2; ; k++) {
      if (!used.contains('${base}_$k')) return '${base}_$k';
    }
  }

  /// With a parameter for [target] (at its [current] value), named after
  /// it ([defaultParameterName]) with [suggestPrior]'s prior; this
  /// document when a parameter sets [target] already.
  ModelDocument freeTarget(DocTarget target, double current) {
    if (parameterFor(target) != null) return this;
    return withParameter(
      docParameter(freshParameterName(defaultParameterName(target)), [
        target,
      ], suggestPrior(target, current)),
    );
  }

  /// Without the parameter that sets [target] (or just [target], when the
  /// parameter sets others too).
  ModelDocument fixTarget(DocTarget target) {
    final p = parameterFor(target);
    if (p == null) return this;
    final rest = [
      for (final t in targetsOf(p))
        if (!sameTarget(t, target)) t,
    ];
    final name = p['name']! as String;
    if (rest.isEmpty) return withoutParameter(name);
    return replaceParameter(name, {...p, 'targets': rest});
  }

  /// With [p] added after the others.
  ModelDocument withParameter(Map<String, Object?> p) =>
      copyWith(parameters: [...parameters, p]);

  /// With the parameter [name] replaced by [p]; a new name is followed by
  /// the group level.
  ModelDocument replaceParameter(String name, Map<String, Object?> p) {
    final at = parameters.indexWhere((x) => x['name'] == name);
    if (at < 0) throw ArgumentError.value(name, 'name', 'No such parameter');
    final next = [...parameters]..[at] = p;
    final to = p['name']! as String;
    return copyWith(
      parameters: next,
      group: group.only({
        for (final q in next) q['name']! as String,
      }, to == name ? const {} : {name: to}),
    );
  }

  /// With the parameter [name]'s prior set to [prior] (and its transform
  /// to the prior's default).
  ModelDocument withPrior(String name, Prior prior) {
    final p = parameter(name);
    if (p == null) throw ArgumentError.value(name, 'name', 'No such parameter');
    final next = {
      ...p,
      'prior': prior.toJson(),
      'transform': prior.defaultTransform.toJson(),
    }..remove('initial');
    return replaceParameter(name, next);
  }

  ModelDocument withoutParameter(String name) => withParameters([
    for (final p in parameters)
      if (p['name'] != name) p,
  ]);

  /// With [next] as the parameters; the group level keeps only theirs.
  ModelDocument withParameters(List<Map<String, Object?>> next) => copyWith(
    parameters: next,
    group: group.only({for (final p in next) p['name']! as String}),
  );

  /// With [g] as the group level.
  ModelDocument withGroup(DocGroup g) => copyWith(group: g);

  /// With the parameter [name] fitted as [role]: a pooled parameter gets
  /// the population [DocPopulation.around] its prior (unless it has one).
  ModelDocument withRole(
    String name,
    ParameterRole role, {
    FitCodec codec = const FitCodec(),
  }) {
    final p = parameter(name);
    if (p == null) throw ArgumentError.value(name, 'name', 'No such parameter');
    final pooled = {...group.pooled}..remove(name);
    final shared = [
      for (final n in group.shared)
        if (n != name) n,
    ];
    switch (role) {
      case ParameterRole.individual:
        break;
      case ParameterRole.pooled:
        pooled[name] =
            group.pooled[name] ??
            DocPopulation.around(
              codec.decodePrior(p['prior']! as Map<String, Object?>),
              transform: codec.decodeTransform(
                p['transform']! as Map<String, Object?>,
              ),
            );
      case ParameterRole.shared:
        shared.add(name);
    }
    return withGroup(
      DocGroup(pooled: pooled, shared: shared, centered: group.centered),
    );
  }
}

/// The issues with [doc]'s free parameters and group level (part
/// [DocPart.fit]): names used twice or empty, priors and transforms that
/// don't decode, targets that address no node or edge, response
/// parameters the response model doesn't have ([responseParameters], when
/// the response decoded), and a group level that names no parameter,
/// pools and shares one parameter, shares one under MAP (empirical Bayes
/// can't), or has a population sd prior without positive support.
List<DocIssue> checkParameters(
  ModelDocument doc, {
  Set<String>? responseParameters,
  FitCodec codec = const FitCodec(),
}) {
  const part = DocPart.fit;
  final issues = <DocIssue>[];
  void error(String m) =>
      issues.add(DocIssue(IssueSeverity.error, m, parts: const [part]));
  final a = doc.network.addresses;
  bool known(Object? address) {
    if (address is! String) return false;
    try {
      a.indexOf(address);
      return true;
    } on Object {
      return false;
    }
  }

  final edges = {
    for (final e in doc.network.edges)
      (
        e.type == EdgeType.value ? 'value_coupling' : 'volatility_coupling',
        e.parent,
        e.child,
      ),
  };
  final names = <String>{};
  for (final p in doc.parameters) {
    final name = p['name'];
    if (name is! String || name.trim().isEmpty) {
      error('A free parameter has no name.');
      continue;
    }
    if (!names.add(name)) error('Two free parameters are named "$name".');
    try {
      codec.decodePrior(p['prior']! as Map<String, Object?>);
      codec.decodeTransform(p['transform']! as Map<String, Object?>);
    } on Object catch (e) {
      error('Parameter "$name": ${buildErrorMessage(e)}');
    }
    final targets = targetsOf(p);
    if (targets.isEmpty) error('Parameter "$name" sets nothing.');
    for (final t in targets) {
      switch (t['type']) {
        case 'node':
          if (!known(t['node'])) {
            error('Parameter "$name" sets "${t['node']}", not a node.');
          }
          try {
            NodeField.parse(t['field']! as String);
          } on Object {
            error('Parameter "$name": "${t['field']}" is not a node field.');
          }
        case final String type
            when type == 'value_coupling' || type == 'volatility_coupling':
          if (!edges.contains((type, t['parent'], t['child']))) {
            error(
              'Parameter "$name" sets the ${type.replaceAll('_', ' ')} '
              '${t['parent']} → ${t['child']}, which is not an edge.',
            );
          }
        case 'response':
          if (doc.response == null) {
            error('Parameter "$name" sets the response, but there is none.');
          } else if (responseParameters != null &&
              !responseParameters.contains(t['name'])) {
            error(
              'Parameter "$name" sets "${t['name']}", which the response '
              'does not have${responseParameters.isEmpty ? '' : ' (it has ${responseParameters.join(', ')})'}.',
            );
          }
        default:
          error('Parameter "$name": bad target ${t['type']}.');
      }
    }
  }

  final g = doc.group;
  for (final n in [...g.pooled.keys, ...g.shared]) {
    if (!names.contains(n)) {
      error('The group level names "$n", which is not a free parameter.');
    }
  }
  for (final n in g.shared) {
    if (g.pooled.containsKey(n)) error('"$n" is both pooled and shared.');
  }
  if (g.shared.isNotEmpty && doc.fit.method == 'map') {
    error(
      'Sharing ${g.shared.join(', ')} across participants needs NUTS: MAP '
      'fits the group by empirical Bayes, which pools but cannot share.',
    );
  }
  for (final MapEntry(:key, :value) in g.pooled.entries) {
    try {
      final pop = value.decode(codec);
      if (pop.sd.lower < 0) {
        error(
          'The population sd of "$key" needs a prior on positive values, '
          'not ${pop.sd}.',
        );
      }
    } on Object catch (e) {
      error('The population of "$key": ${buildErrorMessage(e)}');
    }
  }
  return issues;
}
