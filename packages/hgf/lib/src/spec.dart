import 'coupling.dart';
import 'network.dart';
import 'types.dart';

/// A declarative, JSON-serialisable network description, in the format the
/// pyhgf parity fixtures use:
///
/// ```json
/// {
///   "network": {"volatility_updates": "unbounded", "mean_field_updates": false},
///   "nodes": [
///     {"kind": "continuous-state", "params": {"precision": 1e4}},
///     {"kind": "continuous-state", "value_children": [0], "coupling_fn": "tanh"},
///     {"kind": "continuous-state",
///      "volatility_children": {"idxs": [1], "strengths": [2.0]}}
///   ]
/// }
/// ```
///
/// `"ehgf_mgf_correction": true` in `"network"` turns on
/// [Network.ehgfMgfCorrection], an hgf.dart extension that pyhgf ignores.
final class NetworkSpec {
  NetworkSpec.fromJson(Map<String, Object?> json)
    : json = Map.unmodifiable(json);

  final Map<String, Object?> json;

  /// Build the network. A `coupling_fn` name is looked up in [couplings]
  /// first (your [CustomCoupling]s, by name), then among the named
  /// [CouplingFn]s.
  Network build({Map<String, Coupling> couplings = const {}}) {
    final net = (json['network'] as Map<String, Object?>?) ?? const {};
    final network = Network(
      volatilityUpdates: VolatilityUpdates.parse(
        (net['volatility_updates'] as String?) ?? 'unbounded',
      ),
      meanFieldUpdates: (net['mean_field_updates'] as bool?) ?? false,
      ehgfMgfCorrection: (net['ehgf_mgf_correction'] as bool?) ?? false,
      maxPosteriorPrecision:
          (net['max_posterior_precision'] as num?)?.toDouble() ?? 1e10,
      precisionClippingValue:
          (net['precision_clipping_value'] as num?)?.toDouble() ?? 1e-6,
    );
    for (final raw in json['nodes'] as List<Object?>) {
      final node = raw as Map<String, Object?>;
      if (node['kind'] == NodeKind.efState.pyhgfName) {
        _addEfState(network, node);
        continue;
      }
      final params = <NodeField, double>{
        for (final e
            in ((node['params'] as Map<String, Object?>?) ?? const {}).entries)
          NodeField.parse(e.key): (e.value as num).toDouble(),
      };
      final vp = _edge(node['value_parents']);
      final vc = _edge(node['value_children']);
      final volp = _edge(node['volatility_parents']);
      final volc = _edge(node['volatility_children']);
      network.addNodes(
        kind: NodeKind.parse((node['kind'] as String?) ?? 'continuous-state'),
        nNodes: (node['n_nodes'] as int?) ?? 1,
        valueParents: vp?.$1,
        valueParentsStrengths: vp?.$2,
        valueChildren: vc?.$1,
        valueChildrenStrengths: vc?.$2,
        volatilityParents: volp?.$1,
        volatilityParentsStrengths: volp?.$2,
        volatilityChildren: volc?.$1,
        volatilityChildrenStrengths: volc?.$2,
        coupling: _coupling(node['coupling_fn'] as String?, couplings),
        parameters: params,
      );
    }
    return network;
  }

  /// An ef-state node takes pyhgf's `add_nodes(kind="ef-state", ...)`
  /// parameters: `dimension`, `distribution` (`"normal"` or
  /// `"multivariate-normal"`), `learning` and `nus`; the implied continuous
  /// nodes of HGF learning are added as in pyhgf.
  static void _addEfState(Network network, Map<String, Object?> node) {
    for (final key in const [
      'value_parents',
      'value_children',
      'volatility_parents',
      'volatility_children',
      'coupling_fn',
    ]) {
      if (node[key] != null) {
        throw FormatException('ef-state nodes take no "$key" in a spec');
      }
    }
    final params = (node['params'] as Map<String, Object?>?) ?? const {};
    for (final key in params.keys) {
      if (!const {
        'dimension',
        'distribution',
        'learning',
        'nus',
        'xis',
      }.contains(key)) {
        throw FormatException('Unknown ef-state parameter "$key"');
      }
    }
    final dimension = (params['dimension'] as int?) ?? 1;
    final distribution = (params['distribution'] as String?) ?? 'normal';
    switch (distribution) {
      case 'normal' when dimension != 1:
        throw FormatException(
          'The "normal" distribution is univariate; use "multivariate-normal" '
          'for dimension $dimension',
        );
      case 'normal' || 'multivariate-normal':
        break;
      default:
        throw FormatException('Unknown ef-state distribution "$distribution"');
    }
    network.addEfState(
      nNodes: (node['n_nodes'] as int?) ?? 1,
      dimension: dimension,
      learning: EfLearning.parse(
        (params['learning'] as String?) ?? 'generalised-filtering',
      ),
      nus: (params['nus'] as num?)?.toDouble() ?? 3.0,
      xis: (params['xis'] as List<Object?>?)
          ?.map((x) => (x as num).toDouble())
          .toList(),
    );
  }

  static Coupling _coupling(String? name, Map<String, Coupling> custom) {
    if (name == null) return CouplingFn.linear;
    return custom[name] ?? CouplingFn.parse(name);
  }

  static (List<int>, List<double>?)? _edge(Object? v) {
    if (v == null) return null;
    if (v is int) return ([v], null);
    if (v is List) return ([for (final x in v) x as int], null);
    if (v is Map) {
      return (
        [for (final x in v['idxs'] as List<Object?>) x as int],
        [
          for (final x in v['strengths'] as List<Object?>)
            (x as num).toDouble(),
        ],
      );
    }
    throw FormatException('Bad edge spec: $v');
  }
}
