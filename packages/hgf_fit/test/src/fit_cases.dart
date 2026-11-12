// Models with subjects and evaluation points, shared by hgf_fit's
// reverse-mode tests and hgf_gpu's GPU fit tests.
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';

double _normal(math.Random rng) =>
    math.sqrt(-2 * math.log(1 - rng.nextDouble())) *
    math.cos(2 * math.pi * rng.nextDouble());

/// A noisy random walk with calm and volatile phases.
Float64List walk(int n, math.Random rng) {
  final out = Float64List(n);
  var x = 0.0;
  for (var t = 0; t < n; t++) {
    x += ((t ~/ 50).isEven ? 0.05 : 0.4) * _normal(rng);
    out[t] = x + 0.5 * _normal(rng);
  }
  return out;
}

/// Binary outcomes whose probability switches between 0.8 and 0.2 every
/// 40 trials.
Float64List binaryOutcomes(int n, math.Random rng) => Float64List.fromList([
  for (var t = 0; t < n; t++)
    rng.nextDouble() < ((t ~/ 40).isEven ? 0.8 : 0.2) ? 1.0 : 0.0,
]);

/// A model, its subjects, and points (natural scale) to evaluate at.
typedef FitCase = ({
  String name,
  HgfModel model,
  List<Subject> subjects,
  List<List<double>> points,
});

List<FitCase> fitCases() {
  final cases = <FitCase>[];
  final u = binaryOutcomes(320, math.Random(7));
  {
    final model = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0]),
      response: const BinarySoftmaxReadout(),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: TruncatedNormal(-3, 5, lower: -8, upper: 2),
        ),
        Parameter.response('inverseTemperature', prior: const Uniform(0.2, 20)),
      ],
    );
    final r = model.simulate(
      [Subject(inputs: u), Subject(inputs: u)],
      [
        [-3.5, 3.0],
        [-1.5, 1.0],
      ],
      math.Random(1),
    );
    final r1 = Float64List.fromList(r[1])..[10] = double.nan;
    cases.add((
      name: 'binary 2-level, softmax (ω₂, β), missed response, short series',
      model: model,
      subjects: [
        Subject(inputs: u, responses: r[0]),
        Subject(inputs: u, responses: r1),
        Subject(
          inputs: Float64List.sublistView(u, 0, 150),
          responses: Float64List.sublistView(r[0], 0, 150),
        ),
      ],
      points: [
        [-4.0, 2.0],
        [-2.5, 6.0],
      ],
    ));
  }
  final data = walk(300, math.Random(3));
  for (final mode in VolatilityUpdates.values) {
    final model = HgfModel(
      network: Network(volatilityUpdates: mode)
        ..addNodes(precision: 4)
        ..addNodes(valueChildren: [0], tonicVolatility: -3)
        ..addNodes(volatilityChildren: [1], tonicVolatility: -4),
      response: const GaussianReadout(),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
        Parameter.node(
          'omega3',
          2,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
        Parameter.volatilityCoupling('kappa', 2, 1, prior: const HalfNormal(2)),
        Parameter.node(
          'inputPrecision',
          0,
          NodeField.precision,
          prior: const LogNormal(1, 1),
        ),
        Parameter.response('precision', prior: const GammaPrior(2, 1)),
      ],
    );
    final sim = model.simulate(
      [Subject(inputs: data)],
      [
        [-3.0, -4.0, 1.0, 4.0, 2.0],
      ],
      math.Random(4),
    )[0];
    final observed = Float64List(data.length)..fillRange(0, data.length, 1);
    for (var t = 40; t < 50; t++) {
      observed[t] = 0;
    }
    final steps = Float64List.fromList([
      for (var t = 0; t < data.length; t++) t.isEven ? 1.0 : 0.5,
    ]);
    cases.add((
      name:
          'continuous 3-level ${mode.name}, Gaussian readout, 5 parameters, '
          'missing observations, time steps',
      model: model,
      subjects: [
        Subject(inputs: data, responses: sim),
        Subject(
          inputs: data,
          responses: sim,
          observed: observed,
          timeSteps: steps,
        ),
      ],
      points: [
        [-3.0, -4.0, 1.0, 4.0, 2.0],
        [-2.0, -5.0, 0.5, 2.0, 0.5],
      ],
    ));
  }
  {
    final d = walk(200, math.Random(5));
    cases.add((
      name: 'eHGF perceptual: total surprise, value coupling, prior mean',
      model: HgfModel(
        network: Network(volatilityUpdates: VolatilityUpdates.eHGF)
          ..addNodes(precision: 4)
          ..addNodes(valueChildren: [0], tonicVolatility: -3)
          ..addNodes(volatilityChildren: [1], tonicVolatility: -4),
        response: const ObservationSurprise(TotalGaussianSurprise()),
        parameters: [
          Parameter.node(
            'omega2',
            1,
            NodeField.tonicVolatility,
            prior: const Normal(-3, 2),
          ),
          Parameter.valueCoupling('alpha', 1, 0, prior: const Normal(1, 0.3)),
          Parameter.node('mu2', 1, NodeField.mean, prior: const Normal(0, 1)),
        ],
      ),
      subjects: [Subject(inputs: d)],
      points: [
        [-3.0, 1.0, 0.0],
        [-2.0, 0.8, 0.5],
      ],
    ));
  }
  {
    final model = HgfModel(
      network: Network()
        ..addNodes(kind: NodeKind.binaryState)
        ..addNodes(valueChildren: [0])
        ..addNodes(volatilityChildren: [1]),
      response: const ObservationSurprise(FirstLevelBinarySurprise()),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
        Parameter.node(
          'omega3',
          2,
          NodeField.tonicVolatility,
          prior: const Normal(-6, 2),
        ),
      ],
    );
    cases.add((
      name: 'binary 3-level perceptual (first-level binary surprise)',
      model: model,
      subjects: [Subject(inputs: u)],
      points: [
        [-4.0, -6.0],
        [-3.0, -5.0],
      ],
    ));
  }
  {
    // The Iowa Gambling Task: four continuous branches, a softmax over the
    // predicted payoffs; only the chosen deck is observed.
    final model = HgfModel(
      network: IowaGamblingTask.network(),
      response: IowaGamblingTask.readout(),
      parameters: [
        Parameter(
          'omega',
          targets: [
            for (final n in IowaGamblingTask.valueNodes)
              NodeTarget(n, NodeField.tonicVolatility),
          ],
          prior: const Normal(-2, 1.5),
        ),
        Parameter.response(
          'inverseTemperature',
          prior: const LogNormal(0.5, 1),
        ),
      ],
    );
    final rng = math.Random(6);
    final sims = model.simulateChoices(
      [
        Subject(inputs: IowaGamblingTask.outcomes(100, rng)),
        Subject(inputs: IowaGamblingTask.outcomes(100, rng)),
      ],
      [
        [-2.0, 1.5],
        [-3.0, 4.0],
      ],
      rng,
      reveals: IowaGamblingTask.reveals,
    );
    sims[1].responses![20] = double.nan;
    cases.add((
      name: 'Iowa Gambling Task: softmax over 4 decks (ω, β), missed choice',
      model: model,
      subjects: sims,
      points: [
        [-2.0, 1.5],
        [-1.0, 3.0],
      ],
    ));
  }
  {
    // pyhgf's two-armed bandit: arm values W − L from four binary branches.
    final net = Network()..addNodes(kind: NodeKind.binaryState, nNodes: 4);
    for (var i = 0; i < 4; i++) {
      net.addNodes(valueChildren: [i], tonicVolatility: -2.5);
    }
    final model = HgfModel(
      network: net,
      response: const SoftmaxReadout(
        options: [
          {0: 1.0, 1: -1.0},
          {2: 1.0, 3: -1.0},
        ],
      ),
      parameters: [
        Parameter(
          'omega',
          targets: [
            for (final n in [4, 5, 6, 7])
              NodeTarget(n, NodeField.tonicVolatility),
          ],
          prior: const Normal(-2.5, 2),
        ),
        Parameter.response('inverseTemperature', prior: const LogNormal(1, 1)),
      ],
    );
    final rng = math.Random(8);
    final outcomes = Float64List.fromList([
      for (var t = 0; t < 160; t++)
        for (final p in [(t ~/ 40).isEven ? 0.8 : 0.2, 0.5, 0.3, 0.5])
          rng.nextDouble() < p ? 1.0 : 0.0,
    ]);
    cases.add((
      name: 'two-armed bandit: softmax over W − L (ω, β)',
      model: model,
      subjects: model.simulateChoices(
        [Subject(inputs: outcomes)],
        [
          [-2.5, 4.0],
        ],
        rng,
        reveals: const [
          [0, 1],
          [2, 3],
        ],
      ),
      points: [
        [-2.5, 4.0],
        [-1.5, 1.0],
      ],
    ));
  }
  return cases;
}

/// Evaluates every case's points for every subject on the CPU and on
/// [fitter], with and without gradients and the Jacobian; returns the
/// mismatches (and logs the largest errors with [log]).
