@TestOn('vm')
library;

import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

/// Draws from three normal clusters (centres −6, 0.5, 7), in random order.
Float64List _mixture(int nTime, int seed) {
  final rng = math.Random(seed);
  const centres = [-6.0, 0.5, 7.0];
  return Float64List.fromList([
    for (var t = 0; t < nTime; t++)
      centres[rng.nextInt(3)] +
          math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
              math.cos(2 * math.pi * rng.nextDouble()),
  ]);
}

HgfModel _model({bool withPrecision = false}) => HgfModel(
  network: Network()
    ..addDpState(batchSize: 6, alpha: 1.0)
    ..addEfState(nNodes: 6, valueChildren: [0], nus: 15.0),
  response: const ObservationSurprise(DpSurprise()),
  parameters: [
    Parameter.node('alpha', 0, NodeField.alpha, prior: const LogNormal(-2, 2)),
    if (withPrecision)
      Parameter.node(
        'sensoryPrecision',
        0,
        NodeField.sensoryPrecision,
        prior: const LogNormal(0, 1),
      ),
  ],
);

void main() {
  test('a dp model has no exact gradients', () {
    expect(_model().hasExactGradients, isFalse);
  });

  test('the codec round-trips a dp model', () {
    const codec = FitCodec();
    final model = _model(withPrecision: true);
    final json = jsonDecode(jsonEncode(codec.encodeModel(model)));
    final back = codec.decodeModel(json as Map<String, Object?>);
    final subjects = [Subject(inputs: _mixture(80, 1))];
    final y = Float64List.fromList([math.log(0.2), 0.1]);
    final a = Float64List(1), b = Float64List(1);
    model.negLogPosterior(subjects, Int32List(1), y, a);
    back.negLogPosterior(subjects, Int32List(1), y, b);
    expect(b[0], a[0]);
  });

  test('Nelder-Mead MAP improves on its start and on a large alpha', () {
    final model = _model();
    final subjects = [Subject(inputs: _mixture(100, 0))];
    final fit = fitMap(model, subjects, optimizer: const NelderMead());
    final r = fit.results.single;
    expect(r.negLogPosterior.isFinite, isTrue);
    expect(r.x[0], inInclusiveRange(1e-4, 2.0));
    double nlp(double alpha) {
      final out = Float64List(1);
      model.negLogPosterior(
        subjects,
        Int32List(1),
        Float64List.fromList([math.log(alpha)]),
        out,
      );
      return out[0];
    }

    expect(r.negLogPosterior, lessThanOrEqualTo(nlp(model.start()[0])));
    expect(r.negLogPosterior, lessThan(nlp(10)));
  });
}
