part of 'deep_network.dart';

/// Format tag of [DeepNetwork.toJson].
const String _kDeepFormat = 'hgf_deep.network/1';

Object _enc(double v) => v.isFinite
    ? v
    : v.isNaN
    ? 'NaN'
    : v > 0
    ? 'Infinity'
    : '-Infinity';

double _dec(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Not a number: $v'),
};

List<Object> _encList(Float64List v) => [for (final x in v) _enc(x)];

Float64List _decList(Object? v, int length, String what) {
  final list = v as List<Object?>;
  if (list.length != length) {
    throw FormatException('$what has ${list.length} values, not $length');
  }
  return Float64List.fromList([for (final x in list) _dec(x)]);
}

String _couplingName(Coupling c) => switch (c) {
  CouplingFn() => c.pyhgfName,
  CustomCoupling() => c.name,
  _ => throw UnsupportedError(
    'Cannot name the coupling ${c.runtimeType}: use a CouplingFn or a '
    'CustomCoupling',
  ),
};

Coupling _coupling(String name, Map<String, Coupling> custom) {
  final c = custom[name];
  if (c != null) return c;
  try {
    return CouplingFn.parse(name);
  } on ArgumentError {
    throw ArgumentError(
      'Unknown coupling "$name": pass it in DeepNetwork.fromJson(couplings:)',
    );
  }
}

Map<String, Object?> _deepNetworkToJson(DeepNetwork net) {
  final s = net.settings;
  return {
    'format': _kDeepFormat,
    'coupling': _couplingName(net.coupling),
    'tonic_volatility': net.tonicVolatility,
    'settings': {
      'volatility_updates': s.volatilityUpdates.pyhgfName,
      'max_posterior_precision': _enc(s.maxPosteriorPrecision),
      'precision_clipping_value': _enc(s.precisionClippingValue),
      'predict_precision': s.predictPrecision,
      'feedforward_uncertainty': s.feedforwardUncertainty,
      'carry_precision': s.carryPrecision,
      'mean_field_updates': s.meanFieldUpdates,
      'update_input_layer': s.updateInputLayer,
      'fixes': {
        'curvature_floor': s.fixes.curvatureFloor,
        'log_space_volatility': s.fixes.logSpaceVolatility,
      },
    },
    'layers': [
      for (var l = 0; l < net._configs.length; l++)
        _layerToJson(net._configs[l], net._layers[l], net._state[l]),
    ],
  };
}

Map<String, Object?> _layerToJson(LayerConfig c, Layer layer, LayerState st) =>
    {
      'size': c.size,
      'kind': c.kind.pyhgfName,
      'add_constant_input': c.addConstantInput,
      'fully_connected': c.fullyConnected,
      'coupling': _couplingName(c.coupling),
      'volatility_parent': c.volatilityParent,
      'tonic_volatility': c.tonicVolatility == null
          ? null
          : _enc(c.tonicVolatility!),
      'tonic_volatility_vol': _enc(c.tonicVolatilityVol),
      'initial': {
        for (final MapEntry(:key, :value) in c.initial.entries)
          key.pyhgfName: _enc(value),
      },
      if (c.kind == LayerKind.continuous) ...{
        'value_children': c.valueChild,
        'volatility_children': c.volatilityChild,
        'value_coupling': _enc(c.valueCoupling),
        'volatility_coupling': _enc(c.volatilityCoupling),
        'volatility_fully_connected': c.volatilityFullyConnected,
        'tonic_drift': _enc(c.tonicDrift),
        'autoconnection_strength': _enc(c.autoconnectionStrength),
        'volatility_weights': layer.volatilityWeights == null
            ? null
            : _encList(layer.volatilityWeights!.data),
        'node_tonic_drift': _encList(layer.tonicDrift!),
        'node_autoconnection_strength': _encList(layer.autoconnectionStrength!),
      },
      'weights': layer.weights == null ? null : _encList(layer.weights!.data),
      if (layer.weightsPrecisionDelta case final d?)
        'weights_precision_delta': _encList(d),
      'node_tonic_volatility': layer.tonicVolatility == null
          ? null
          : _encList(layer.tonicVolatility!),
      'node_tonic_volatility_vol': _encList(layer.tonicVolatilityVol),
      'node_prior_precision': _encList(layer.priorPrecision),
      'state': {
        for (final f in LayerField.values)
          if (st.field(f) case final v?) f.pyhgfName: _encList(v),
        if (st.knownVariance case final v?) 'known_variance': _encList(v),
      },
    };

DeepNetwork _deepNetworkFromJson(
  Map<String, Object?> json,
  Map<String, Coupling> couplings,
) {
  if (json['format'] != _kDeepFormat) {
    throw FormatException(
      'Not a DeepNetwork.toJson() snapshot (format "${json['format']}"); '
      'expected "$_kDeepFormat"',
    );
  }
  final s = json['settings'] as Map<String, Object?>;
  final fixes = s['fixes'] as Map<String, Object?>;
  final net = DeepNetwork(
    coupling: _coupling(json['coupling'] as String, couplings),
    volatilityUpdates: DeepVolatilityUpdate.parse(
      s['volatility_updates'] as String,
    ),
    maxPosteriorPrecision: _dec(s['max_posterior_precision']),
    precisionClippingValue: _dec(s['precision_clipping_value']),
    predictPrecision: s['predict_precision'] as bool,
    feedforwardUncertainty: s['feedforward_uncertainty'] as bool,
    tonicVolatility: json['tonic_volatility'] as bool,
    meanFieldUpdates: s['mean_field_updates'] as bool,
    carryPrecision: s['carry_precision'] as bool,
    updateInputLayer: s['update_input_layer'] as bool? ?? false,
    fixes: DeepFixes(
      curvatureFloor: fixes['curvature_floor'] as bool,
      logSpaceVolatility: fixes['log_space_volatility'] as bool,
    ),
  );
  final layers = (json['layers'] as List<Object?>).cast<Map<String, Object?>>();
  for (final l in layers) {
    final tv = l['tonic_volatility'];
    if (l['kind'] == LayerKind.continuous.pyhgfName) {
      net.addLayer(
        l['size'] as int,
        kind: LayerKind.continuous,
        fullyConnected: l['fully_connected'] as bool,
        coupling: _coupling(l['coupling'] as String, couplings),
        tonicVolatility: _dec(tv),
        initial: {
          for (final MapEntry(:key, :value)
              in (l['initial'] as Map<String, Object?>).entries)
            LayerField.parse(key): _dec(value),
        },
        // Explicit children (null, null) would mean the chain default.
        valueChildren: l['value_children'] as int?,
        volatilityChildren: l['volatility_children'] as int?,
        valueCoupling: _dec(l['value_coupling']),
        volatilityCoupling: _dec(l['volatility_coupling']),
        volatilityFullyConnected: l['volatility_fully_connected'] as bool,
        tonicDrift: _dec(l['tonic_drift']),
        autoconnectionStrength: _dec(l['autoconnection_strength']),
      );
      continue;
    }
    net.addLayer(
      l['size'] as int,
      kind: LayerKind.parse(l['kind'] as String),
      addConstantInput: l['add_constant_input'] as bool,
      fullyConnected: l['fully_connected'] as bool,
      coupling: _coupling(l['coupling'] as String, couplings),
      volatilityParent: l['volatility_parent'] as bool,
      tonicVolatility: tv == null ? null : _dec(tv),
      tonicVolatilityVol: _dec(l['tonic_volatility_vol']),
      initial: {
        for (final MapEntry(:key, :value)
            in (l['initial'] as Map<String, Object?>).entries)
          LayerField.parse(key): _dec(value),
      },
    );
  }
  for (var i = 0; i < layers.length; i++) {
    final l = layers[i];
    final layer = net._layers[i];
    final where = 'layer $i';
    final w = layer.weights;
    if ((w == null) != (l['weights'] == null)) {
      throw FormatException('$where: weights present on one side only');
    }
    if (w != null) {
      w.data.setAll(0, _decList(l['weights'], w.data.length, '$where weights'));
      if (l['weights_precision_delta'] case final d?) {
        layer.weightsPrecisionDelta = _decList(
          d,
          w.data.length,
          '$where weights_precision_delta',
        );
      }
    }
    final n = layer.size;
    final kw = layer.volatilityWeights;
    if ((kw == null) != (l['volatility_weights'] == null)) {
      throw FormatException('$where: volatility_weights on one side only');
    }
    if (kw != null) {
      kw.data.setAll(
        0,
        _decList(l['volatility_weights'], kw.data.length, '$where κ'),
      );
    }
    if (layer.tonicDrift case final d? when l['node_tonic_drift'] != null) {
      d.setAll(0, _decList(l['node_tonic_drift'], n, where));
    }
    if (layer.autoconnectionStrength case final a?
        when l['node_autoconnection_strength'] != null) {
      a.setAll(0, _decList(l['node_autoconnection_strength'], n, where));
    }
    final tv = layer.tonicVolatility;
    if (tv != null && l['node_tonic_volatility'] != null) {
      tv.setAll(0, _decList(l['node_tonic_volatility'], n, where));
    }
    layer.tonicVolatilityVol.setAll(
      0,
      _decList(l['node_tonic_volatility_vol'], n, where),
    );
    layer.priorPrecision.setAll(
      0,
      _decList(l['node_prior_precision'], n, where),
    );
    final st = net._state[i];
    final state = l['state'] as Map<String, Object?>;
    for (final f in LayerField.values) {
      final dst = st.field(f);
      final src = state[f.pyhgfName];
      if (dst == null || src == null) continue;
      dst.setAll(0, _decList(src, n, '$where ${f.pyhgfName}'));
    }
    if (st.knownVariance case final kv? when state['known_variance'] != null) {
      kv.setAll(0, _decList(state['known_variance'], n, where));
    }
  }
  return net;
}
