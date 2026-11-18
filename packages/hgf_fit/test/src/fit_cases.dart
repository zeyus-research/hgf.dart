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
  // Noise parents of an input (pyhgf #477): node 3 is a noise parent of the
  // input and a volatility parent of its tanh value parent, which shares
  // the input with a bias node.
  final noisy = Float64List.fromList([
    for (final x in walk(240, math.Random(9))) 0.4 * _tanh(x / 2),
  ]);
  for (final mode in VolatilityUpdates.values) {
    final model = HgfModel(
      network: Network(volatilityUpdates: mode)
        ..addNodes()
        ..addNodes(
          valueChildren: [0],
          coupling: CouplingFn.tanh,
          tonicVolatility: -3,
        )
        ..addNodes(kind: NodeKind.constantState, valueChildren: [0])
        ..addNodes(volatilityChildren: [0, 1], tonicVolatility: -4),
      response: const ObservationSurprise(TotalGaussianSurprise()),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
        Parameter.node(
          'omegaNoise',
          3,
          NodeField.tonicVolatility,
          prior: const Normal(-4, 2),
        ),
        Parameter.node(
          'epsilonU',
          0,
          NodeField.tonicVolatility,
          prior: const Normal(0, 1),
        ),
        Parameter.volatilityCoupling(
          'kappaU',
          3,
          0,
          prior: const HalfNormal(2),
        ),
        Parameter.valueCoupling('alpha', 1, 0, prior: const Normal(1, 0.3)),
        Parameter.valueCoupling('bias', 2, 0, prior: const Normal(0, 0.3)),
      ],
    );
    final observed = Float64List(noisy.length)..fillRange(0, noisy.length, 1);
    for (var t = 60; t < 70; t++) {
      observed[t] = 0;
    }
    final steps = Float64List.fromList([
      for (var t = 0; t < noisy.length; t++) t % 3 == 0 ? 0.5 : 1.25,
    ]);
    cases.add((
      name:
          'noise parent ${mode.name}: tanh and bias value parents, missing '
          'observations, time steps',
      model: model,
      subjects: [
        Subject(inputs: noisy),
        Subject(inputs: noisy, observed: observed, timeSteps: steps),
      ],
      points: [
        [-3.0, -4.0, 0.0, 1.0, 1.0, 0.1],
        [-2.0, -3.0, 0.5, 0.6, 0.8, -0.2],
      ],
    ));
  }
  {
    // An expression coupling (ExprCoupling): symbolic g', g'', g''' on the
    // CPU, lowered to the IR for the GPU and reverse mode.
    final model = HgfModel(
      network: Network()
        ..addNodes()
        ..addNodes(
          valueChildren: [0],
          coupling: ExprCoupling('2 * tanh(x / 2) + 0.1 * sqrt(1 + x^2)'),
          tonicVolatility: -3,
        )
        ..addNodes(volatilityChildren: [1], tonicVolatility: -4),
      response: const ObservationSurprise(TotalGaussianSurprise()),
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
        Parameter.valueCoupling('alpha', 1, 0, prior: const Normal(1, 0.3)),
      ],
    );
    cases.add((
      name: 'expression coupling: total surprise, 3 levels',
      model: model,
      subjects: [Subject(inputs: noisy)],
      points: [
        [-3.0, -4.0, 1.0],
        [-2.0, -5.0, 0.7],
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
    // The Iowa Gambling Task with a fifth option, "pass" (worth its bias
    // alone, reveals nothing), stickiness and a lapse rate.
    final model = HgfModel(
      network: IowaGamblingTask.network(),
      response: const SoftmaxReadout(
        options: [
          {0: 1.0},
          {1: 1.0},
          {2: 1.0},
          {3: 1.0},
          {},
        ],
        stickiness: 0.0,
        lapse: 0.05,
        bias: [0, 0, 0, 0, 0],
      ),
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
        Parameter.response('stickiness', prior: const Normal(0, 1)),
        Parameter.response('lapse', prior: const Uniform(0, 0.5)),
        Parameter.response('bias_4', prior: const Normal(0, 1)),
      ],
    );
    final rng = math.Random(9);
    final sims = model.simulateChoices(
      [
        Subject(inputs: IowaGamblingTask.outcomes(120, rng)),
        Subject(inputs: IowaGamblingTask.outcomes(120, rng)),
      ],
      [
        [-2.0, 1.5, 1.0, 0.1, -0.5],
        [-3.0, 3.0, -0.5, 0.02, 0.5],
      ],
      rng,
      reveals: const [...IowaGamblingTask.reveals, <int>[]],
    );
    sims[1].responses![30] = double.nan;
    cases.add((
      name:
          'choice extras: IGT + pass, stickiness, lapse, bias (missed choice)',
      model: model,
      subjects: sims,
      points: [
        [-2.0, 1.5, 1.0, 0.1, -0.5],
        [-1.0, 3.0, -0.8, 0.3, 1.0],
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
  {
    // The bandit with an ExprReadout: a bias towards arm 2, β, and a
    // nonlinear value (tanh of the difference).
    final net = Network()..addNodes(kind: NodeKind.binaryState, nNodes: 4);
    for (var i = 0; i < 4; i++) {
      net.addNodes(valueChildren: [i], tonicVolatility: -2.5);
    }
    final model = HgfModel(
      network: net,
      response: ExprReadout.choice(
        ['beta * tanh(w1 - l1)', 'beta * (w2 - l2) + bias'],
        beliefs: {
          'w1': (node: 0, field: NodeField.expectedMean),
          'l1': (node: 1, field: NodeField.expectedMean),
          'w2': (node: 2, field: NodeField.expectedMean),
          'l2': (node: 3, field: NodeField.expectedMean),
        },
        parameters: {'beta': 1.0, 'bias': 0.0},
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
        Parameter.response('beta', prior: const LogNormal(1, 1)),
        Parameter.response('bias', prior: const Normal(0, 1)),
      ],
    );
    final rng = math.Random(18);
    final outcomes = Float64List.fromList([
      for (var t = 0; t < 160; t++)
        for (final p in [(t ~/ 40).isEven ? 0.8 : 0.2, 0.5, 0.3, 0.5])
          rng.nextDouble() < p ? 1.0 : 0.0,
    ]);
    final sims = model.simulateChoices(
      [Subject(inputs: outcomes)],
      [
        [-2.5, 4.0, 0.3],
      ],
      rng,
      reveals: const [
        [0, 1],
        [2, 3],
      ],
    );
    sims[0].responses![30] = double.nan;
    cases.add((
      name: 'ExprReadout choice: bandit with bias and tanh value',
      model: model,
      subjects: sims,
      points: [
        [-2.5, 4.0, 0.3],
        [-1.5, 1.0, -0.5],
      ],
    ));
  }
  {
    // The bandit with an ExprReadout that reads the previous choice:
    // stickiness written as an expression.
    final net = Network()..addNodes(kind: NodeKind.binaryState, nNodes: 4);
    for (var i = 0; i < 4; i++) {
      net.addNodes(valueChildren: [i], tonicVolatility: -2.5);
    }
    final model = HgfModel(
      network: net,
      response: ExprReadout.choice(
        [
          'beta * (w1 - l1) + where(previous == 0, kappa, 0)',
          'beta * (w2 - l2) + where(previous == 1, kappa, 0)',
        ],
        beliefs: {
          'w1': (node: 0, field: NodeField.expectedMean),
          'l1': (node: 1, field: NodeField.expectedMean),
          'w2': (node: 2, field: NodeField.expectedMean),
          'l2': (node: 3, field: NodeField.expectedMean),
        },
        parameters: {'beta': 1.0, 'kappa': 0.0},
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
        Parameter.response('beta', prior: const LogNormal(1, 1)),
        Parameter.response('kappa', prior: const Normal(0, 1)),
      ],
    );
    final rng = math.Random(19);
    final outcomes = Float64List.fromList([
      for (var t = 0; t < 160; t++)
        for (final p in [(t ~/ 40).isEven ? 0.8 : 0.2, 0.5, 0.3, 0.5])
          rng.nextDouble() < p ? 1.0 : 0.0,
    ]);
    final sims = model.simulateChoices(
      [Subject(inputs: outcomes)],
      [
        [-2.5, 3.0, 1.0],
      ],
      rng,
      reveals: const [
        [0, 1],
        [2, 3],
      ],
    );
    sims[0].responses![40] = double.nan;
    cases.add((
      name: 'ExprReadout choice: stickiness through previous',
      model: model,
      subjects: sims,
      points: [
        [-2.5, 3.0, 1.0],
        [-1.5, 1.0, -0.7],
      ],
    ));
  }
  {
    // ExprReadout.gaussian: the response reads a mix of two levels, with
    // a log-precision parameter.
    final model = HgfModel(
      network: Network()
        ..addNodes(precision: 4)
        ..addNodes(valueChildren: [0], tonicVolatility: -3)
        ..addNodes(volatilityChildren: [1], tonicVolatility: -4),
      response: ExprReadout.gaussian(
        mean: 'mu1 + w * sigmoid(mu2)',
        precision: 'exp(logPrecision)',
        beliefs: {
          'mu1': (node: 1, field: NodeField.expectedMean),
          'mu2': (node: 2, field: NodeField.mean),
        },
        parameters: {'w': 0.5, 'logPrecision': 1.0},
      ),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
        Parameter.response('w', prior: const Normal(0, 1)),
        Parameter.response('logPrecision', prior: const Normal(0, 2)),
      ],
    );
    final sim = model.simulate(
      [Subject(inputs: data)],
      [
        [-3.0, 0.5, 1.0],
      ],
      math.Random(14),
    )[0];
    sim[17] = double.nan;
    cases.add((
      name: 'ExprReadout gaussian: mean and precision expressions',
      model: model,
      subjects: [Subject(inputs: data, responses: sim)],
      points: [
        [-3.0, 0.5, 1.0],
        [-2.0, -0.3, 0.2],
      ],
    ));
  }
  {
    // ExprReadout.surprise: a Laplace (double-exponential) response.
    final model = HgfModel(
      network: Network()
        ..addNodes(precision: 4)
        ..addNodes(valueChildren: [0], tonicVolatility: -3),
      response: ExprReadout.surprise(
        'abs(response - mu) / b + log(2 * b)',
        beliefs: {'mu': (node: 1, field: NodeField.expectedMean)},
        parameters: {'b': 0.5},
      ),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
        Parameter.response('b', prior: const LogNormal(0, 1)),
      ],
    );
    final rng = math.Random(15);
    final responses = Float64List.fromList([
      for (final x in data) x + 0.3 * (rng.nextDouble() - 0.5),
    ]);
    cases.add((
      name: 'ExprReadout surprise: Laplace responses',
      model: model,
      subjects: [Subject(inputs: data, responses: responses)],
      points: [
        [-3.0, 0.5],
        [-2.0, 0.2],
      ],
    ));
  }
  {
    // ef-state, bivariate, HGF-1 learning: five statistics, each tracked by
    // a node (1 + 2s) with a value parent (2 + 2s). Two columns per trial,
    // one observed flag. (HGF-2's 16 nodes make kernels that SwiftShader,
    // the browser suite's WebGPU on CI, takes minutes to compile.)
    final rng = math.Random(11);
    const n = 180;
    final x = walk(n, rng), y = walk(n, rng);
    final inputs = Float64List(2 * n);
    for (var t = 0; t < n; t++) {
      inputs[2 * t] = 0.5 * x[t];
      inputs[2 * t + 1] = 0.3 * (x[t] + y[t]);
    }
    final observed = Float64List(n)..fillRange(0, n, 1.0);
    for (final t in [0, 17, 18, 90, 151]) {
      observed[t] = 0.0;
    }
    final steps = Float64List.fromList([
      for (var t = 0; t < n; t++) t % 3 == 0 ? 0.5 : 1.0,
    ]);
    final model = HgfModel(
      network: Network()..addEfState(dimension: 2, learning: EfLearning.hgf1),
      response: const ObservationSurprise(TotalGaussianSurprise()),
      parameters: [
        Parameter(
          'omega1',
          targets: [
            for (var s = 0; s < 5; s++)
              NodeTarget(1 + 2 * s, NodeField.tonicVolatility),
          ],
          prior: const Normal(-4, 2),
        ),
        Parameter(
          'omega2',
          targets: [
            for (var s = 0; s < 5; s++)
              NodeTarget(2 + 2 * s, NodeField.tonicVolatility),
          ],
          prior: const Normal(-6, 2),
        ),
        Parameter.node(
          'pi1',
          1,
          NodeField.precision,
          prior: const LogNormal(0, 1),
        ),
      ],
    );
    cases.add((
      name:
          'ef-state 2-d HGF-1: total surprise, missing observations, time '
          'steps',
      model: model,
      subjects: [Subject(inputs: inputs, observed: observed, timeSteps: steps)],
      points: [
        [-4.0, -6.0, 1.0],
        [-2.5, -5.0, 2.0],
      ],
    ));
  }
  {
    // A continuous input (node 0, with nodes 1 and 2 above) beside a
    // univariate ef-state input with HGF-1 learning (node 3; statistics
    // tracked by 4 and 6 under 5 and 7): columns [u, x], observed flags
    // per input node, each missing on its own trials. eHGF updates (the
    // bivariate case above runs the default, unbounded ones).
    final rng = math.Random(12);
    const n = 160;
    final u = walk(n, rng), v = walk(n, rng);
    final inputs = Float64List(2 * n);
    final observed = Float64List(2 * n)..fillRange(0, 2 * n, 1.0);
    for (var t = 0; t < n; t++) {
      inputs[2 * t] = u[t];
      inputs[2 * t + 1] = 0.4 * v[t];
    }
    for (final t in [3, 40, 41]) {
      observed[2 * t] = 0.0;
    }
    for (final t in [0, 41, 99]) {
      observed[2 * t + 1] = 0.0;
    }
    final model = HgfModel(
      network: Network(volatilityUpdates: VolatilityUpdates.eHGF)
        ..addNodes(precision: 2)
        ..addNodes(valueChildren: [0], tonicVolatility: -3)
        ..addNodes(volatilityChildren: [1], tonicVolatility: -4)
        ..addEfState(learning: EfLearning.hgf1),
      response: const ObservationSurprise(TotalGaussianSurprise()),
      parameters: [
        Parameter.node(
          'omega2',
          1,
          NodeField.tonicVolatility,
          prior: const Normal(-3, 2),
        ),
        Parameter(
          'omegaEf',
          targets: [
            NodeTarget(5, NodeField.tonicVolatility),
            NodeTarget(7, NodeField.tonicVolatility),
          ],
          prior: const Normal(-4, 2),
        ),
      ],
    );
    cases.add((
      name: 'mixed inputs: continuous beside ef-state HGF-1, missing per input',
      model: model,
      subjects: [Subject(inputs: inputs, observed: observed)],
      points: [
        [-3.0, -4.0],
        [-2.0, -2.5],
      ],
    ));
  }
  return cases;
}

/// Evaluates every case's points for every subject on the CPU and on
/// [fitter], with and without gradients and the Jacobian; returns the
/// mismatches (and logs the largest errors with [log]).

double _tanh(double x) {
  final e = math.exp(2 * x);
  return (e - 1) / (e + 1);
}
