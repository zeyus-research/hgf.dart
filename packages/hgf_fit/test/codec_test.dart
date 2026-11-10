import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_fit/src/empirical_bayes.dart' show populationPrior;
import 'package:test/test.dart';

/// A custom prior, serialisable through a decoder.
final class _Laplace extends Prior {
  const _Laplace(this.location, this.scale);

  final double location, scale;

  @override
  double logDensity(double x) =>
      -(x - location).abs() / scale - math.log(2 * scale);

  @override
  double sample(math.Random rng) => throw UnimplementedError();

  @override
  double get typical => location;

  @override
  Map<String, Object?> toJson() => {
    'type': 'Laplace',
    'location': location,
    'scale': scale,
  };

  static _Laplace fromJson(Map<String, Object?> json, FitCodec codec) =>
      _Laplace(
        (json['location'] as num).toDouble(),
        (json['scale'] as num).toDouble(),
      );
}

const _codec = FitCodec(priors: {'Laplace': _Laplace.fromJson});

HgfModel _roundTrip(HgfModel model, [FitCodec codec = _codec]) =>
    codec.decodeModel(
      jsonDecode(jsonEncode(codec.encodeModel(model))) as Map<String, Object?>,
    );

/// Evaluates [a] and [b] (with gradients where possible) at random points
/// and expects identical results.
void _sameObjective(HgfModel a, HgfModel b, List<Subject> subjects) {
  final rng = math.Random(7);
  final k = a.dimension, n = subjects.length * 3;
  final subjectOf = Int32List.fromList([
    for (var e = 0; e < n; e++) e % subjects.length,
  ]);
  final y = Float64List(n * k);
  final y0 = a.toUnconstrained(a.start());
  for (var e = 0; e < n; e++) {
    for (var i = 0; i < k; i++) {
      y[e * k + i] = y0[i] + 0.3 * (rng.nextDouble() - 0.5);
    }
  }
  for (final jacobian in [false, true]) {
    final outA = Float64List(n), outB = Float64List(n);
    final gA = a.hasExactGradients ? Float64List(n * k) : null;
    final gB = a.hasExactGradients ? Float64List(n * k) : null;
    a.negLogPosterior(
      subjects,
      subjectOf,
      y,
      outA,
      jacobian: jacobian,
      gradient: gA,
    );
    b.negLogPosterior(
      subjects,
      subjectOf,
      y,
      outB,
      jacobian: jacobian,
      gradient: gB,
    );
    expect(outB, outA);
    expect(outA.any((v) => v.isFinite), isTrue);
    if (gA != null) expect(gB, gA);
  }
}

Float64List _binaryInputs(math.Random rng, int n) {
  final out = Float64List(n);
  var p = 0.8;
  for (var t = 0; t < n; t++) {
    if (t % 40 == 0) p = 1 - p;
    out[t] = rng.nextDouble() < p ? 1.0 : 0.0;
  }
  return out;
}

void main() {
  final rng = math.Random(1);

  test('binary model with every prior and transform', () {
    final model = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0])
        ..addNodes(volatilityChildren: [1], tonicVolatility: -6)
        ..setInitial(1, NodeField.mean, 0.25),
      response: const BinarySoftmaxReadout(inverseTemperature: 2),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: TruncatedNormal(-4, 2, upper: -1),
        ),
        Parameter.node(
          'omega3',
          2,
          NodeField.tonicVolatility,
          prior: const _Laplace(-6, 1),
        ),
        Parameter.volatilityCoupling(
          'kappa',
          2,
          1,
          prior: const GammaPrior(4, 4),
        ),
        Parameter.node(
          'mu2',
          1,
          NodeField.mean,
          prior: const Flat(lower: -3, upper: 3),
          transform: const IdentityTransform(),
          initial: 0.1,
        ),
        Parameter.response(
          'beta',
          responseName: 'inverseTemperature',
          prior: const Uniform(0.5, 10),
        ),
      ],
      blockSize: 64,
    );
    final copy = _roundTrip(model);
    expect(copy.names, model.names);
    expect(copy.blockSize, 64);
    expect(copy.parameters[3].initial, 0.1);
    expect(copy.parameters[3].transform, isA<IdentityTransform>());
    expect(copy.parameters[1].prior, isA<_Laplace>());
    final subjects = [
      for (var s = 0; s < 4; s++)
        Subject(
          inputs: _binaryInputs(rng, 120),
          responses: _binaryInputs(rng, 120),
        ),
    ];
    _sameObjective(model, copy, subjects);
    expect(
      () => _roundTrip(model, const FitCodec()),
      throwsA(isA<ArgumentError>()),
    );
  });

  test('categorical models: choices and observation surprise', () {
    final picks = [
      for (var t = 0; t < 90; t++) (t ~/ 30 + (t % 7 == 0 ? 1 : 0)) % 3,
    ];
    final u = Float64List.fromList([
      for (final k in picks)
        for (var c = 0; c < 3; c++) c == k ? 1.0 : 0.0,
    ]);
    final parameters = [
      Parameter.node(
        'omega_vol',
        7,
        NodeField.tonicVolatility,
        prior: const Normal(-6, 2),
      ),
    ];
    for (final response in <ResponseModel>[
      const CategoricalSoftmaxReadout(inverseTemperature: 2),
      const ObservationSurprise(CategoricalSurprise()),
    ]) {
      final model = HgfModel(
        network: Network()..addCategoricalState(nCategories: 3),
        response: response,
        parameters: [
          ...parameters,
          if (response is CategoricalSoftmaxReadout)
            Parameter.response(
              'beta',
              responseName: 'inverseTemperature',
              prior: const LogNormal(0.7, 1),
            ),
        ],
      );
      final copy = _roundTrip(model);
      expect(copy.response.toJson(), response.toJson());
      _sameObjective(model, copy, [
        Subject(
          inputs: u,
          responses: Float64List.fromList([
            for (final k in picks) (k + 1) % 3 == 0 ? double.nan : k.toDouble(),
          ]),
        ),
      ]);
    }
  });

  test('softmax choices over belief combinations', () {
    const response = SoftmaxReadout(
      options: [
        {0: 1.0, 1: -1.0},
        {2: 0.5},
        {3: 1.0},
      ],
      field: NodeField.expectedMean,
      inverseTemperature: 2,
    );
    final model = HgfModel(
      network: IowaGamblingTask.network(),
      response: response,
      parameters: [
        Parameter(
          'omega',
          targets: [
            for (final n in IowaGamblingTask.valueNodes)
              NodeTarget(n, NodeField.tonicVolatility),
          ],
          prior: const Normal(-2, 1.5),
        ),
        Parameter.response('inverseTemperature', prior: const LogNormal(0, 1)),
      ],
    );
    final copy = _roundTrip(model);
    expect(copy.response.toJson(), response.toJson());
    final rng = math.Random(3);
    final subjects = model.simulateChoices(
      [Subject(inputs: IowaGamblingTask.outcomes(60, rng))],
      [
        [-2.0, 2.0],
      ],
      rng,
      reveals: const [
        [0, 1],
        [2],
        [3],
      ],
    );
    _sameObjective(model, copy, subjects);
  });

  test('continuous model with a Gaussian readout and couplings', () {
    final model = HgfModel(
      network: Network(volatilityUpdates: VolatilityUpdates.eHGF)
        ..addNodes(precision: 1e2)
        ..addNodes(valueChildren: [0], coupling: CouplingFn.tanh)
        ..addNodes(volatilityChildren: [1]),
      response: const GaussianReadout(precision: 4),
      parameters: [
        Parameter.node(
          'omega',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 1),
        ),
        Parameter.valueCoupling('alpha', 1, 0, prior: const LogNormal(0, 0.3)),
        Parameter.node(
          'pi0',
          0,
          NodeField.precision,
          prior: const HalfNormal(200),
        ),
        Parameter.response('precision', prior: const LogNormal(1, 0.5)),
      ],
    );
    final copy = _roundTrip(model);
    final subjects = [
      for (var s = 0; s < 3; s++)
        Subject(
          inputs: Float64List.fromList([
            for (var t = 0; t < 80; t++) math.sin(t / 9) + rng.nextDouble(),
          ]),
          responses: Float64List.fromList([
            for (var t = 0; t < 80; t++) t == 5 ? double.nan : math.sin(t / 9),
          ]),
        ),
    ];
    _sameObjective(model, copy, subjects);
  });

  test('observation surprise, built-in and named functions', () {
    Network net() => Network()
      ..addNodes(precision: 1e2)
      ..addNodes(valueChildren: [0]);
    final params = [
      Parameter.node(
        'omega',
        1,
        NodeField.tonicVolatility,
        prior: const Normal(-3, 1),
      ),
    ];
    final subjects = [
      Subject(
        inputs: Float64List.fromList([
          for (var t = 0; t < 60; t++) math.cos(t / 5),
        ]),
      ),
    ];
    final builtIn = HgfModel(
      network: net(),
      response: const ObservationSurprise(TotalGaussianSurprise()),
      parameters: params,
    );
    _sameObjective(builtIn, _roundTrip(builtIn), subjects);

    final custom = ResponseFunction.fn(
      (n, t) => 0.0,
      batch: (n, t, e) => n.get(e, 0, NodeField.expectedMean).abs(),
    );
    final named = HgfModel(
      network: net(),
      response: ObservationSurprise(custom),
      parameters: params,
    );
    expect(() => const FitCodec().encodeModel(named), throwsUnsupportedError);
    final codec = FitCodec(responseFunctions: {'abs': custom});
    _sameObjective(named, _roundTrip(named, codec), subjects);
  });

  test('empirical Bayes population priors', () {
    final base = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: const BinarySoftmaxReadout(),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
        Parameter.response(
          'inverseTemperature',
          prior: const LogNormal(0.5, 1),
        ),
      ],
    );
    // What fitEmpiricalBayes refits with (package-internal priors, which
    // reach Web Workers through the codec).
    final model = base.withPriors({
      for (final p in base.parameters)
        p.name: populationPrior(
          p.transform,
          p.name == 'omega2' ? -4.2 : 0.3,
          0.6,
        ),
    });
    final copy = _roundTrip(model);
    expect(
      copy.parameters[0].prior.toString(),
      model.parameters[0].prior.toString(),
    );
    final subjects = [
      for (var s = 0; s < 4; s++)
        Subject(
          inputs: _binaryInputs(rng, 150),
          responses: _binaryInputs(rng, 150),
        ),
    ];
    _sameObjective(model, copy, subjects);
  });

  test('ResponseModel.fn is not serialisable', () {
    final model = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: ResponseModel.fn(surprise: (n, e, t, r, p, o) => 0.0),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
      ],
    );
    expect(() => const FitCodec().encodeModel(model), throwsUnsupportedError);
  });

  test('non-finite numbers survive jsonEncode', () {
    final params = const FitCodec().decodeParameters(
      jsonDecode(
            jsonEncode(
              const FitCodec().encodeParameters([
                Parameter.node(
                  'x',
                  0,
                  NodeField.mean,
                  prior: TruncatedNormal(0, 1, lower: 0),
                ),
              ]),
            ),
          )
          as List<Object?>,
    );
    final p = params.single.prior as TruncatedNormal;
    expect(p.lower, 0);
    expect(p.upper, double.infinity);
    expect(params.single.transform, isA<LowerBoundTransform>());
  });
}
