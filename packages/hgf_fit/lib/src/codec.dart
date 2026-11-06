import 'package:hgf/hgf.dart';

import 'empirical_bayes.dart' show populationPrior;
import 'model.dart';
import 'parameter.dart';
import 'prior.dart';
import 'response_model.dart';
import 'transform.dart';

/// Format tag of [FitCodec.encodeModel].
const String _kModelFormat = 'hgf_fit.model/1';

/// Reads a custom type's `toJson()` output back.
typedef JsonDecoder<T> = T Function(Map<String, Object?> json, FitCodec codec);

/// Turns an [HgfModel] into JSON-encodable maps and back: to send it to a
/// Web Worker (see `package:hgf_fit/fit_worker.dart`), or to save it.
///
/// The built-in priors, transforms and response models write themselves
/// (`toJson()`), and the network is `Network.toJson`'s exact copy, so a
/// decoded model evaluates bit for bit like the original. Custom types
/// override `toJson()` (with a `"type"` key) and are read back by the
/// decoders given here under that type. Custom couplings and the
/// `ResponseFunction` of an [ObservationSurprise] are code, not data: they
/// are written by name and looked up in [couplings] and
/// [responseFunctions]. A `ResponseModel.fn` can't be serialised; write a
/// `ResponseModel` subclass with a `toJson()` instead.
///
/// Non-finite numbers are written as the strings `"NaN"`, `"Infinity"`,
/// `"-Infinity"`, so `jsonEncode` accepts the result.
final class FitCodec {
  const FitCodec({
    this.priors = const {},
    this.transforms = const {},
    this.responseModels = const {},
    this.responseFunctions = const {},
    this.couplings = const {},
  });

  /// Decoders of custom [Prior]s, by their `"type"`.
  final Map<String, JsonDecoder<Prior>> priors;

  /// Decoders of custom [Transform]s, by their `"type"`.
  final Map<String, JsonDecoder<Transform>> transforms;

  /// Decoders of custom [ResponseModel]s, by their `"type"`.
  final Map<String, JsonDecoder<ResponseModel>> responseModels;

  /// `ResponseFunction`s other than the built-in parameter-free ones, for
  /// [ObservationSurprise], by a name of your choice.
  final Map<String, ResponseFunction> responseFunctions;

  /// `CustomCoupling`s used by networks, by their name.
  final Map<String, Coupling> couplings;

  /// [model] as JSON.
  Map<String, Object?> encodeModel(HgfModel model) => {
    'format': _kModelFormat,
    'network': model.network.toJson(),
    'response': encodeResponse(model.response),
    'parameters': encodeParameters(model.parameters),
    'block_size': model.blockSize,
  };

  /// A model from [encodeModel]'s output.
  HgfModel decodeModel(Map<String, Object?> json) {
    if (json['format'] != _kModelFormat) {
      throw FormatException(
        'Not an encoded HgfModel (format "${json['format']}")',
      );
    }
    return HgfModel(
      network: Network.fromJson(
        json['network'] as Map<String, Object?>,
        couplings: couplings,
      ),
      response: decodeResponse(json['response'] as Map<String, Object?>),
      parameters: decodeParameters(json['parameters'] as List<Object?>),
      blockSize: (json['block_size'] as num).toInt(),
    );
  }

  /// [parameters] as JSON.
  List<Object?> encodeParameters(List<Parameter> parameters) => [
    for (final p in parameters)
      {
        'name': p.name,
        'targets': [for (final t in p.targets) _encodeTarget(t)],
        'prior': encodePrior(p.prior),
        'transform': encodeTransform(p.transform),
        if (p.initial case final x?) 'initial': jsonDouble(x),
      },
  ];

  /// Parameters from [encodeParameters]'s output.
  List<Parameter> decodeParameters(List<Object?> json) => [
    for (final raw in json)
      switch (raw as Map<String, Object?>) {
        final p => Parameter(
          p['name'] as String,
          targets: [
            for (final t in p['targets'] as List<Object?>)
              _decodeTarget(t as Map<String, Object?>),
          ],
          prior: decodePrior(p['prior'] as Map<String, Object?>),
          transform: decodeTransform(p['transform'] as Map<String, Object?>),
          initial: p['initial'] == null ? null : readJsonDouble(p['initial']),
        ),
      },
  ];

  Map<String, Object?> encodePrior(Prior prior) => prior.toJson();

  Prior decodePrior(Map<String, Object?> json) {
    double d(String key) => readJsonDouble(json[key]);
    return switch (json['type']) {
      'Normal' => Normal(d('mean'), d('sd')),
      'TruncatedNormal' => TruncatedNormal(
        d('mean'),
        d('sd'),
        lower: d('lower'),
        upper: d('upper'),
      ),
      'Uniform' => Uniform(d('lower'), d('upper')),
      'HalfNormal' => HalfNormal(d('sd')),
      'LogNormal' => LogNormal(d('mu'), d('sigma')),
      'Gamma' => GammaPrior(d('shape'), d('rate')),
      'Flat' => Flat(
        lower: d('lower'),
        upper: d('upper'),
        typical: json['typical'] == null ? null : d('typical'),
      ),
      'Population' => populationPrior(
        decodeTransform(json['transform'] as Map<String, Object?>),
        d('mu'),
        d('sd'),
      ),
      final String type => _custom(priors, type, json, 'prior'),
      final type => throw FormatException('Bad prior type: $type'),
    };
  }

  Map<String, Object?> encodeTransform(Transform transform) =>
      transform.toJson();

  Transform decodeTransform(Map<String, Object?> json) {
    double d(String key) => readJsonDouble(json[key]);
    return switch (json['type']) {
      'Identity' => const IdentityTransform(),
      'LowerBound' => LowerBoundTransform(d('lower')),
      'UpperBound' => UpperBoundTransform(d('upper')),
      'Interval' => IntervalTransform(d('lower'), d('upper')),
      final String type => _custom(transforms, type, json, 'transform'),
      final type => throw FormatException('Bad transform type: $type'),
    };
  }

  Map<String, Object?> encodeResponse(ResponseModel response) {
    if (response case ObservationSurprise(:final function)) {
      for (final MapEntry(:key, :value) in responseFunctions.entries) {
        if (identical(value, function) || value == function) {
          return {'type': 'ObservationSurprise', 'function': key};
        }
      }
    }
    return response.toJson();
  }

  ResponseModel decodeResponse(Map<String, Object?> json) {
    return switch (json['type']) {
      'GaussianReadout' => GaussianReadout(
        node: (json['node'] as num).toInt(),
        field: NodeField.parse(json['field'] as String),
        precision: readJsonDouble(json['precision']),
      ),
      'BinarySoftmaxReadout' => BinarySoftmaxReadout(
        node: (json['node'] as num).toInt(),
        inverseTemperature: readJsonDouble(json['inverseTemperature']),
      ),
      'ObservationSurprise' => ObservationSurprise(switch (json['function']) {
        'FirstLevelGaussianSurprise' => const FirstLevelGaussianSurprise(),
        'TotalGaussianSurprise' => const TotalGaussianSurprise(),
        'FirstLevelBinarySurprise' => const FirstLevelBinarySurprise(),
        final name =>
          responseFunctions[name] ??
              (throw ArgumentError(
                'Unknown response function "$name": add it to '
                'FitCodec.responseFunctions',
              )),
      }),
      final String type => _custom(
        responseModels,
        type,
        json,
        'response model',
      ),
      final type => throw FormatException('Bad response model type: $type'),
    };
  }

  T _custom<T>(
    Map<String, JsonDecoder<T>> decoders,
    String type,
    Map<String, Object?> json,
    String what,
  ) {
    final decode = decoders[type];
    if (decode == null) {
      throw ArgumentError(
        'Unknown $what type "$type": give the FitCodec a decoder for it',
      );
    }
    return decode(json, this);
  }

  static Map<String, Object?> _encodeTarget(ParameterTarget t) => switch (t) {
    NodeTarget(:final node, :final field) => {
      'type': 'node',
      'node': node,
      'field': field.pyhgfName,
    },
    ValueCouplingTarget(:final parent, :final child) => {
      'type': 'value_coupling',
      'parent': parent,
      'child': child,
    },
    VolatilityCouplingTarget(:final parent, :final child) => {
      'type': 'volatility_coupling',
      'parent': parent,
      'child': child,
    },
    ResponseTarget(:final name) => {'type': 'response', 'name': name},
  };

  static ParameterTarget _decodeTarget(Map<String, Object?> t) {
    int i(String key) => (t[key] as num).toInt();
    return switch (t['type']) {
      'node' => NodeTarget(i('node'), NodeField.parse(t['field'] as String)),
      'value_coupling' => ValueCouplingTarget(i('parent'), i('child')),
      'volatility_coupling' => VolatilityCouplingTarget(
        i('parent'),
        i('child'),
      ),
      'response' => ResponseTarget(t['name'] as String),
      final type => throw FormatException('Bad parameter target: $type'),
    };
  }
}

/// [v] as a JSON value: non-finite values as strings.
Object jsonDouble(double v) => v.isFinite
    ? v
    : v.isNaN
    ? 'NaN'
    : v > 0
    ? 'Infinity'
    : '-Infinity';

/// Reads [jsonDouble]'s output (any JSON number, or one of its strings).
double readJsonDouble(Object? v) => switch (v) {
  num() => v.toDouble(),
  'NaN' => double.nan,
  'Infinity' => double.infinity,
  '-Infinity' => double.negativeInfinity,
  _ => throw FormatException('Not a number: $v'),
};
