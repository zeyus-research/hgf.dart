import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:test/test.dart';

HgfModel _igt() => HgfModel(
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
    Parameter.response('inverseTemperature', prior: const LogNormal(0.5, 1)),
  ],
);

double _corr(List<double> a, List<double> b) {
  final n = a.length;
  final ma = a.reduce((x, y) => x + y) / n;
  final mb = b.reduce((x, y) => x + y) / n;
  var sab = 0.0, saa = 0.0, sbb = 0.0;
  for (var i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) * (a[i] - ma);
    sbb += (b[i] - mb) * (b[i] - mb);
  }
  return sab / math.sqrt(saa * sbb);
}

void main() {
  group('IowaGamblingTask', () {
    test('outcomes have the decks\' long-run payoffs', () {
      final u = IowaGamblingTask.outcomes(20000, math.Random(1));
      for (final (d, net) in [(0, -0.25), (1, -0.25), (2, 0.25), (3, 0.25)]) {
        var sum = 0.0;
        for (var t = 0; t < 20000; t++) {
          sum += u[t * 4 + d];
        }
        expect(sum / 20000, closeTo(net, 0.08), reason: 'deck $d');
      }
      expect(
        IowaGamblingTask.outcomes(30, math.Random(7)),
        IowaGamblingTask.outcomes(30, math.Random(7)),
      );
    });

    test('data() puts the chosen deck\'s payoff in its column', () {
      final (:inputs, :observed) = IowaGamblingTask.data(
        [2, 0, double.nan, 3],
        [50, -150, 100, -200],
      );
      expect(inputs, [0, 0, 0.5, 0, -1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2]);
      expect(observed, [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
      expect(() => IowaGamblingTask.data([4], [1]), throwsArgumentError);
      expect(() => IowaGamblingTask.data([1, 2], [1]), throwsArgumentError);
    });
  });

  group('simulateChoices', () {
    test(
      'reveals only the chosen option, and refits to the same objective',
      () {
        final model = _igt();
        final rng = math.Random(2);
        final env = [
          for (var s = 0; s < 5; s++)
            Subject(inputs: IowaGamblingTask.outcomes(120, rng), id: s),
        ];
        // A subject missing a few records of its own.
        final own = Float64List(120 * 4)..fillRange(0, 120 * 4, 1.0);
        own.fillRange(40, 48, 0.0); // trials 10-11
        env[3] = Subject(inputs: env[3].inputs, observed: own, id: 3);
        final x = [
          for (var s = 0; s < 5; s++) [-2.0 - s * 0.3, 0.5 + s],
        ];
        final sims = model.simulateChoices(
          env,
          x,
          rng,
          reveals: IowaGamblingTask.reveals,
        );
        for (var s = 0; s < 5; s++) {
          final sim = sims[s];
          expect(sim.id, s);
          expect(identical(sim.inputs, env[s].inputs), isTrue);
          final r = sim.responses!, o = sim.observed!;
          for (var t = 0; t < 120; t++) {
            final c = r[t].toInt();
            expect(c, inInclusiveRange(0, 3));
            for (var d = 0; d < 4; d++) {
              final expected = d == c && !(s == 3 && (t == 10 || t == 11));
              expect(o[t * 4 + d], expected ? 1.0 : 0.0);
            }
          }
        }
        // A sharp agent learns to prefer the good decks (C, D).
        final sharp = model
            .simulateChoices(
              [Subject(inputs: IowaGamblingTask.outcomes(200, rng))],
              [
                [-2.0, 6.0],
              ],
              rng,
              reveals: IowaGamblingTask.reveals,
            )
            .single
            .responses!;
        final late = sharp.sublist(100).where((c) => c >= 2).length;
        expect(late, greaterThan(80));

        // The simulated subjects are ordinary fitting data: re-filtering them
        // with their own choices gives the same per-trial predictions, so the
        // objective is finite and lowest near the truth for β.
        final out = Float64List(1);
        double nll(Subject s, List<double> at) {
          model.negLogPosterior(
            [s],
            Int32List(1),
            model.toUnconstrained(at),
            out,
            includePrior: const [false, false],
          );
          return out[0];
        }

        final s4 = sims[4];
        expect(nll(s4, [-3.2, 4.5]), lessThan(nll(s4, [-3.2, 0.2])));
        expect(nll(s4, [-3.2, 4.5]).isFinite, isTrue);
      },
    );

    test('several inputs per option (pyhgf\'s two-armed bandit)', () {
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
          Parameter.response(
            'beta',
            responseName: 'inverseTemperature',
            prior: const LogNormal(0, 1),
          ),
        ],
      );
      final rng = math.Random(8);
      final outcomes = Float64List.fromList([
        for (var t = 0; t < 80; t++)
          for (var i = 0; i < 4; i++) rng.nextBool() ? 1.0 : 0.0,
      ]);
      final sim = model
          .simulateChoices(
            [Subject(inputs: outcomes)],
            [
              [2.0],
            ],
            rng,
            reveals: const [
              [0, 1],
              [2, 3],
            ],
          )
          .single;
      for (var t = 0; t < 80; t++) {
        final a = sim.responses![t] == 0 ? 1.0 : 0.0;
        expect(sim.observed!.sublist(t * 4, t * 4 + 4), [a, a, 1 - a, 1 - a]);
      }
      expect(
        () => model.simulateChoices(
          [Subject(inputs: outcomes)],
          [
            [2.0],
          ],
          rng,
          reveals: const [
            [0, 4],
            [2],
          ],
        ),
        throwsRangeError,
      );
      expect(
        () => model.simulateChoices(
          [Subject(inputs: outcomes)],
          [
            [2.0],
          ],
          rng,
          reveals: const [
            [0],
          ],
        ),
        throwsStateError,
      );
    });

    test('SoftmaxReadout checks its options and responses', () {
      expect(
        () => HgfModel(
          network: IowaGamblingTask.network(),
          response: SoftmaxReadout.nodes(const [0]),
          parameters: const [],
        ),
        throwsArgumentError,
      );
      expect(
        () => HgfModel(
          network: IowaGamblingTask.network(),
          response: SoftmaxReadout.nodes(const [0, 9]),
          parameters: const [],
        ),
        throwsA(anything),
      );
      final model = _igt();
      final u = IowaGamblingTask.outcomes(5, math.Random(1));
      expect(
        () => model.negLogPosteriorAt(
          Subject(inputs: u, responses: Float64List.fromList([0, 1, 2, 3, 4])),
          [-2, 1],
        ),
        throwsArgumentError,
      );
    });

    test('stickiness, lapse and bias shape simulated choices', () {
      HgfModel model() => HgfModel(
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
          lapse: 0.0,
          bias: [0, 0, 0, 0, 0],
        ),
        parameters: [
          Parameter.response('stickiness', prior: const Normal(0, 1)),
          Parameter.response('lapse', prior: const Uniform(0, 1)),
          Parameter.response('bias_4', prior: const Normal(0, 1)),
        ],
      );
      List<double> run(List<double> x) {
        final rng = math.Random(3);
        return model()
            .simulateChoices(
              [Subject(inputs: IowaGamblingTask.outcomes(2000, rng))],
              [x],
              rng,
              reveals: const [...IowaGamblingTask.reveals, <int>[]],
            )
            .single
            .responses!;
      }

      double repeats(List<double> c) {
        var n = 0;
        for (var t = 1; t < c.length; t++) {
          if (c[t] == c[t - 1]) n++;
        }
        return n / (c.length - 1);
      }

      expect(repeats(run([3, 0, 0])), greaterThan(0.6));
      expect(repeats(run([-3, 0, 0])), lessThan(0.1));
      final passes = run([0, 0, 4]).where((c) => c == 4).length / 2000;
      expect(passes, greaterThan(0.8));
      // A lapse of 1 chooses uniformly whatever the bias.
      final uniform = run([0, 1, 6]).where((c) => c == 4).length / 2000;
      expect(uniform, closeTo(0.2, 0.04));
    });

    test('choice kernel: α = 1 is stickiness; φ > 0 repeats', () {
      HgfModel model(SoftmaxReadout readout) => HgfModel(
        network: IowaGamblingTask.network(),
        response: readout,
        parameters: [
          Parameter.response(
            'inverseTemperature',
            prior: const LogNormal(0.5, 1),
          ),
        ],
      );
      const decks = [
        {0: 1.0},
        {1: 1.0},
        {2: 1.0},
        {3: 1.0},
      ];
      final kernel = model(
        const SoftmaxReadout(
          options: decks,
          choiceKernel: 1.5,
          choiceKernelRate: 1.0,
        ),
      );
      final sticky = model(
        const SoftmaxReadout(options: decks, stickiness: 1.5),
      );
      final rng = math.Random(5);
      final sims = kernel.simulateChoices(
        [Subject(inputs: IowaGamblingTask.outcomes(200, rng))],
        [
          [2.0],
        ],
        rng,
        reveals: IowaGamblingTask.reveals,
      );
      double nll(HgfModel m) {
        final out = Float64List(1), ll = Float64List(1);
        m.negLogPosterior(
          sims,
          Int32List(1),
          Float64List.fromList([0.7]),
          out,
          negLogLikelihood: ll,
        );
        return ll[0];
      }

      expect(nll(kernel), closeTo(nll(sticky), 1e-9));

      double repeats(double phi, double rate) {
        final m = HgfModel(
          network: IowaGamblingTask.network(),
          response: SoftmaxReadout(
            options: decks,
            inverseTemperature: 0.0,
            choiceKernel: phi,
            choiceKernelRate: rate,
          ),
          parameters: [
            Parameter.response('choiceKernel', prior: const Normal(0, 1)),
          ],
        );
        final r = math.Random(8);
        final c = m
            .simulateChoices(
              [Subject(inputs: IowaGamblingTask.outcomes(2000, r))],
              [
                [phi],
              ],
              r,
              reveals: IowaGamblingTask.reveals,
            )
            .single
            .responses!;
        var n = 0;
        for (var t = 1; t < c.length; t++) {
          if (c[t] == c[t - 1]) n++;
        }
        return n / (c.length - 1);
      }

      // β = 0: chance repeats 1/4 without a kernel.
      expect(repeats(0, 0.3), closeTo(0.25, 0.04));
      expect(repeats(4, 0.3), greaterThan(0.6));
      expect(repeats(-4, 1), lessThan(0.1));
    });

    test('SoftmaxReadout needs both choice-kernel terms', () {
      final net = IowaGamblingTask.network();
      expect(
        () => const SoftmaxReadout(
          options: [
            {0: 1.0},
            {1: 1.0},
          ],
          choiceKernel: 1,
        ).prepare(net),
        throwsArgumentError,
      );
    });

    test('parameter recovery on simulated IGT agents (MAP)', () {
      final model = _igt();
      final rng = math.Random(11);
      const n = 80;
      final truth = [
        for (var a = 0; a < n; a++)
          [for (final p in model.parameters) p.prior.sample(rng)],
      ];
      final sims = model.simulateChoices(
        [
          for (var a = 0; a < n; a++)
            Subject(inputs: IowaGamblingTask.outcomes(150, rng)),
        ],
        truth,
        rng,
        reveals: IowaGamblingTask.reveals,
      );
      final fit = fitMap(model, sims).results;
      final rOmega = _corr(
        [for (final t in truth) t[0]],
        [for (final f in fit) f.x[0]],
      );
      final rBeta = _corr(
        [for (final t in truth) math.log(t[1])],
        [for (final f in fit) math.log(f.x[1])],
      );
      expect(rBeta, greaterThan(0.9));
      expect(rOmega, greaterThan(0.5));
    });
  });
}
