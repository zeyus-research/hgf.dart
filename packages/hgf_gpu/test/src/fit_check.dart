import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';
import 'package:hgf_fit/hgf_fit.dart';
import 'package:hgf_gpu/hgf_gpu.dart';

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
  return cases;
}

/// Evaluates every case's points for every subject on the CPU and on
/// [fitter], with and without gradients and the Jacobian; returns the
/// mismatches (and logs the largest errors with [log]).
Future<List<String>> checkFitCases(
  GpuFitter fitter, {
  double valueTol = 2e-5,
  double gradientTol = 2e-3,
  void Function(String)? log,
}) async {
  final failures = <String>[];
  for (final c in fitCases()) {
    final model = c.model, k = model.dimension;
    final n = c.subjects.length * c.points.length;
    final subjectOf = Int32List(n);
    final y = Float64List(n * k);
    var e = 0;
    for (final x in c.points) {
      final yy = model.toUnconstrained(x);
      for (var s = 0; s < c.subjects.length; s++) {
        subjectOf[e] = s;
        y.setRange(e * k, (e + 1) * k, yy);
        e++;
      }
    }
    var maxV = 0.0, maxG = 0.0;
    for (final gradient in [false, true]) {
      for (final jacobian in [false, true]) {
        Pass make() => Pass(
          model,
          c.subjects,
          subjectOf,
          y,
          jacobian: jacobian,
          negLogLikelihood: true,
          gradient: gradient,
        );
        final cpu = make()..run();
        final gpu = make();
        await fitter.evaluatePass(gpu);
        for (var j = 0; j < n; j++) {
          final a = gpu.out[j], b = cpu.out[j];
          final errV = (a - b).abs() / (1 + b.abs());
          if (errV > maxV) maxV = errV;
          if (!(errV <= valueTol)) {
            failures.add('${c.name}: element $j value $a vs $b');
          }
          if (!gradient) continue;
          // Gradient errors relative to the gradient's scale.
          var scale = 1.0;
          for (var i = 0; i < k; i++) {
            scale = math.max(scale, cpu.gradient![j * k + i].abs());
          }
          for (var i = 0; i < k; i++) {
            final ga = gpu.gradient![j * k + i], gb = cpu.gradient![j * k + i];
            final errG = (ga - gb).abs() / scale;
            if (errG > maxG) maxG = errG;
            if (!(errG <= gradientTol)) {
              failures.add(
                '${c.name}: element $j ${model.names[i]} gradient $ga vs $gb',
              );
            }
          }
        }
      }
    }
    log?.call(
      '${c.name}: max value err ${maxV.toStringAsExponential(2)}, '
      'max gradient err ${maxG.toStringAsExponential(2)}',
    );
  }
  return failures;
}
