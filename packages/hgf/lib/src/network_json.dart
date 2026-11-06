part of 'network.dart';

/// Format tag of [Network.toJson].
const String _kNetworkFormat = 'hgf.network/1';

/// A double as JSON: non-finite values as strings, which `jsonEncode`
/// would reject as numbers.
Object _encodeDouble(double v) => v.isFinite
    ? v
    : v.isNaN
    ? 'NaN'
    : v > 0
    ? 'Infinity'
    : '-Infinity';

double _decodeDouble(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Not a number: $v'),
};

List<Object> _encodeDoubles(List<double> v) => [
  for (final x in v) _encodeDouble(x),
];

List<double> _decodeDoubles(Object? v) => [
  for (final x in v as List<Object?>) _decodeDouble(x),
];

Map<String, Object?> _networkToJson(Network network) {
  network.build();
  final n = network._n;
  Map<String, Object?> edges(List<int> idxs, List<double> strengths) => {
    'idxs': List<int>.of(idxs),
    'strengths': _encodeDoubles(strengths),
  };
  return {
    'format': _kNetworkFormat,
    'volatility_updates': network.volatilityUpdates.pyhgfName,
    'mean_field_updates': network.meanFieldUpdates,
    'ehgf_mgf_correction': network.ehgfMgfCorrection,
    'max_posterior_precision': _encodeDouble(network.maxPosteriorPrecision),
    'precision_clipping_value': _encodeDouble(network.precisionClippingValue),
    'nodes': [
      for (final d in network._defs)
        {
          'kind': d.kind.pyhgfName,
          'coupling_fn': switch (d.coupling) {
            final CouplingFn c => c.pyhgfName,
            final CustomCoupling c => c.name,
            final c => throw UnsupportedError(
              'Cannot serialise the coupling ${c.runtimeType}',
            ),
          },
          'custom_coupling': d.coupling is! CouplingFn,
          // As declared (addNodes recompiles from these).
          'params': {
            for (final MapEntry(:key, :value) in d.params.entries)
              key.pyhgfName: _encodeDouble(value),
          },
          if (d.dimension != 1) 'dimension': d.dimension,
          if (d.batchSize != 0) 'batch_size': d.batchSize,
          if (d.xis case final xis?) 'xis': _encodeDoubles(xis),
          'value_parents': edges(d.valueParents, d.valueParentStrengths),
          'value_children': edges(d.valueChildren, d.valueChildStrengths),
          'volatility_parents': edges(
            d.volatilityParents,
            d.volatilityParentStrengths,
          ),
          'volatility_children': edges(
            d.volatilityChildren,
            d.volatilityChildStrengths,
          ),
        },
    ],
    // Every field's initial value, which covers the parameters given to
    // addNodes and everything set with setInitial since.
    'initial': {
      for (final f in NodeField.values)
        f.pyhgfName: _encodeDoubles(
          Float64List.sublistView(
            network._initial,
            f.index * n,
            (f.index + 1) * n,
          ),
        ),
    },
    'vector_initial': {
      for (final f in VectorField.values)
        if (network._vectorInitial(f).isNotEmpty)
          f.pyhgfName: _encodeDoubles(network._vectorInitial(f)),
    },
  };
}

Network _networkFromJson(
  Map<String, Object?> json,
  Map<String, Coupling> couplings,
) {
  if (json['format'] != _kNetworkFormat) {
    throw FormatException(
      'Not a Network.toJson() snapshot (format "${json['format']}"); '
      'NetworkSpec reads the pyhgf-style format',
    );
  }
  final network = Network(
    volatilityUpdates: VolatilityUpdates.parse(
      json['volatility_updates'] as String,
    ),
    meanFieldUpdates: json['mean_field_updates'] as bool,
    ehgfMgfCorrection: json['ehgf_mgf_correction'] as bool,
    maxPosteriorPrecision: _decodeDouble(json['max_posterior_precision']),
    precisionClippingValue: _decodeDouble(json['precision_clipping_value']),
  );
  for (final raw in json['nodes'] as List<Object?>) {
    final node = raw as Map<String, Object?>;
    final name = node['coupling_fn'] as String;
    final Coupling coupling;
    if (node['custom_coupling'] == true) {
      coupling =
          couplings[name] ??
          (throw ArgumentError(
            'The network uses the custom coupling "$name"; pass it in '
            '`couplings`',
          ));
    } else {
      coupling = CouplingFn.parse(name);
    }
    final d = _NodeDef(NodeKind.parse(node['kind'] as String), coupling, {
      for (final MapEntry(:key, :value)
          in ((node['params'] as Map<String, Object?>?) ?? const {}).entries)
        NodeField.parse(key): _decodeDouble(value),
    });
    d.dimension = (node['dimension'] as num?)?.toInt() ?? 1;
    d.batchSize = (node['batch_size'] as num?)?.toInt() ?? 0;
    if (node['xis'] case final xis?) d.xis = _decodeDoubles(xis);
    void edges(String key, List<int> idxs, List<double> strengths) {
      final e = node[key] as Map<String, Object?>;
      idxs.addAll([
        for (final i in e['idxs'] as List<Object?>) (i as num).toInt(),
      ]);
      strengths.addAll(_decodeDoubles(e['strengths']));
    }

    edges('value_parents', d.valueParents, d.valueParentStrengths);
    edges('value_children', d.valueChildren, d.valueChildStrengths);
    edges(
      'volatility_parents',
      d.volatilityParents,
      d.volatilityParentStrengths,
    );
    edges(
      'volatility_children',
      d.volatilityChildren,
      d.volatilityChildStrengths,
    );
    network._defs.add(d);
  }
  network.build();
  final n = network._n;
  final initial = json['initial'] as Map<String, Object?>;
  for (final MapEntry(:key, :value) in initial.entries) {
    final f = NodeField.parse(key);
    final v = _decodeDoubles(value);
    if (v.length != n) {
      throw FormatException('initial "$key" has ${v.length} values, not $n');
    }
    network._initial.setAll(f.index * n, v);
  }
  final vectors = (json['vector_initial'] as Map<String, Object?>?) ?? const {};
  for (final MapEntry(:key, :value) in vectors.entries) {
    final f = VectorField.values.firstWhere(
      (f) => f.pyhgfName == key,
      orElse: () => throw FormatException('Unknown vector field "$key"'),
    );
    final dst = network._vectorInitial(f), v = _decodeDoubles(value);
    if (v.length != dst.length) {
      throw FormatException(
        'vector_initial "$key" has ${v.length} values, not ${dst.length}',
      );
    }
    dst.setAll(0, v);
  }
  network.reset();
  return network;
}
