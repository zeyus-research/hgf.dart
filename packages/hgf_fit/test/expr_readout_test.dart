import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

void main() {
  final rng = math.Random(3);
  final outcomes = Float64List.fromList([
    for (var t = 0; t < 120; t++)
      for (final p in [(t ~/ 30).isEven ? 0.8 : 0.2, 0.5, 0.3, 0.5])
        rng.nextDouble() < p ? 1.0 : 0.0,
  ]);
  Network bandit() {
    final net = Network()..addNodes(kind: NodeKind.binaryState, nNodes: 4);
    for (var i = 0; i < 4; i++) {
      net.addNodes(valueChildren: [i], tonicVolatility: -2.5);
    }
    return net;
  }

  final omega = Parameter(
    'omega',
    targets: [
      for (final n in [4, 5, 6, 7]) NodeTarget(n, NodeField.tonicVolatility),
    ],
    prior: const Normal(-2.5, 2),
  );

  test('ExprReadout.choice scores as the SoftmaxReadout it spells out', () {
    final builtIn = HgfModel(
      network: bandit(),
      response: const SoftmaxReadout(
        options: [
          {0: 1.0, 1: -1.0},
          {2: 1.0, 3: -1.0},
        ],
      ),
      parameters: [
        omega,
        Parameter.response('inverseTemperature', prior: const LogNormal(1, 1)),
      ],
    );
    final expr = HgfModel(
      network: bandit(),
      response: ExprReadout.choice(
        ['beta * (w1 - l1)', 'beta * (w2 - l2)'],
        beliefs: {
          for (final (name, node) in [
            ('w1', 0),
            ('l1', 1),
            ('w2', 2),
            ('l2', 3),
          ])
            name: (node: node, field: NodeField.expectedMean),
        },
        parameters: {'beta': 1.0},
      ),
      parameters: [
        omega,
        Parameter.response('beta', prior: const LogNormal(1, 1)),
      ],
    );
    final subjects = builtIn.simulateChoices(
      [Subject(inputs: outcomes)],
      [
        [-2.5, 3.0],
      ],
      math.Random(1),
      reveals: const [
        [0, 1],
        [2, 3],
      ],
    );
    for (final x in [
      [-2.5, 3.0],
      [-1.0, 0.5],
    ]) {
      expect(
        expr.negLogPosteriorAt(subjects.single, x),
        closeTo(builtIn.negLogPosteriorAt(subjects.single, x), 1e-10),
      );
    }
  });

  test('ExprReadout.gaussian scores as GaussianReadout', () {
    final u = Float64List.fromList([
      for (var t = 0; t < 100; t++) math.sin(t / 9) + 0.2 * math.cos(t),
    ]);
    Network net() => Network()
      ..addNodes(precision: 4)
      ..addNodes(valueChildren: [0], tonicVolatility: -3);
    final builtIn = HgfModel(
      network: net(),
      response: const GaussianReadout(),
      parameters: [
        Parameter.response('precision', prior: const LogNormal(0, 1)),
      ],
    );
    final expr = HgfModel(
      network: net(),
      response: ExprReadout.gaussian(
        mean: 'mu',
        precision: 'p',
        beliefs: {'mu': (node: 0, field: NodeField.expectedMean)},
        parameters: {'p': 1.0},
      ),
      parameters: [Parameter.response('p', prior: const LogNormal(0, 1))],
    );
    final responses = Float64List.fromList([for (final x in u) x + 0.1]);
    final s = Subject(inputs: u, responses: responses);
    for (final p in [0.5, 2.0]) {
      expect(
        expr.negLogPosteriorAt(s, [p]),
        closeTo(builtIn.negLogPosteriorAt(s, [p]), 1e-10),
      );
    }
  });

  test('rejects unbound variables, data functions and misplaced response', () {
    expect(
      () => ExprReadout.choice(['a', 'b']),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => ExprReadout.choice(['lag(a)', 'a'], parameters: {'a': 1}),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => ExprReadout.gaussian(mean: 'response', precision: '1'),
      throwsA(isA<FormatException>()),
    );
    expect(
      () => ExprReadout.surprise(
        'x',
        beliefs: {'x': (node: 0, field: NodeField.mean)},
        parameters: {'x': 1},
      ),
      throwsArgumentError,
    );
  });

  test('round-trips through FitCodec', () {
    final r = ExprReadout.choice(
      ['beta * m', '0'],
      beliefs: {'m': (node: 2, field: NodeField.expectedMean)},
      parameters: {'beta': 2.0},
    );
    final json = const FitCodec().encodeResponse(r);
    final back = const FitCodec().decodeResponse(json) as ExprReadout;
    expect(back.toJson(), r.toJson());
    expect(back.expressions.map((e) => '$e'), ['beta * m', '0']);
    expect(back.beliefs['m'], (node: 2, field: NodeField.expectedMean));
  });
}
