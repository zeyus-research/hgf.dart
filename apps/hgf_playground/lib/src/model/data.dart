import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'presets.dart';

/// Settings of the simulated observations.
class DataSettings {
  DataSettings({
    this.nTrials = 400,
    this.seed = 1,
    this.pHigh = 0.8,
    this.block = 50,
    this.sdLow = 0.05,
    this.sdHigh = 0.4,
    this.noise = 0.1,
    this.nSources = 3,
    this.sourceSd = 0.4,
  });

  int nTrials;
  int seed;

  /// Binary: the probability of the likely outcome.
  double pHigh;

  /// Binary: trials between reversals. Continuous: trials between switches
  /// of volatility.
  int block;

  /// Continuous: the hidden value's step size in calm and volatile phases.
  double sdLow, sdHigh;

  /// Continuous: observation noise (standard deviation).
  double noise;

  /// Mixture: the number of Gaussian sources (means spread over −2.5..2.5).
  int nSources;

  /// Mixture: each source's standard deviation.
  double sourceSd;
}

/// Simulated observations, and the hidden state that generated them.
class SimulatedData {
  SimulatedData(this.observations, this.hidden);

  /// One value per trial (a category index for categorical data).
  final Float64List observations;

  /// Binary: P(u = 1) per trial. Continuous: the hidden value. Categorical:
  /// P(category k), one series per category. Mixture: the mean of the
  /// source drawn from.
  final List<Float64List> hidden;

  int get length => observations.length;
}

double _gauss(math.Random r) =>
    math.sqrt(-2 * math.log(1 - r.nextDouble())) *
    math.cos(2 * math.pi * r.nextDouble());

/// A block's pick out of [n], different from the previous block's.
int _nextPick(math.Random rng, int n, int previous) {
  if (n < 2) return 0;
  final k = rng.nextInt(n - 1);
  return k >= previous ? k + 1 : k;
}

/// Simulate [model]; [nCategories] for [DataModel.categoricalReversal].
SimulatedData simulate(DataModel model, DataSettings s, {int nCategories = 3}) {
  final rng = math.Random(s.seed);
  final n = s.nTrials;
  final u = Float64List(n), hidden = Float64List(n);
  switch (model) {
    case DataModel.categoricalReversal:
      // One category has probability pHigh, the rest share the remainder;
      // which one changes every block.
      final m = nCategories;
      final p = [for (var k = 0; k < m; k++) Float64List(n)];
      final pHigh = math.max(s.pHigh, 1 / m);
      var likely = 0;
      for (var t = 0; t < n; t++) {
        if (t > 0 && t % s.block == 0) likely = _nextPick(rng, m, likely);
        final r = rng.nextDouble();
        var acc = 0.0, pick = m - 1;
        for (var k = m - 1; k >= 0; k--) {
          final pk = k == likely ? pHigh : (1 - pHigh) / (m - 1);
          p[k][t] = pk;
        }
        for (var k = 0; k < m; k++) {
          acc += p[k][t];
          if (r < acc) {
            pick = k;
            break;
          }
        }
        u[t] = pick.toDouble();
      }
      return SimulatedData(u, p);
    case DataModel.mixture:
      final m = math.max(1, s.nSources);
      var source = 0;
      for (var t = 0; t < n; t++) {
        // Every source once, then at random.
        if (t > 0 && t % s.block == 0) {
          final b = t ~/ s.block;
          source = b < m ? b : _nextPick(rng, m, source);
        }
        final mean = m == 1 ? 0.0 : -2.5 + 5.0 * source / (m - 1);
        hidden[t] = mean;
        u[t] = mean + s.sourceSd * _gauss(rng);
      }
    case DataModel.reversal:
      // Reversal learning: P(u = 1) flips between pHigh and 1 − pHigh.
      for (var t = 0; t < n; t++) {
        final p = (t ~/ s.block).isEven ? s.pHigh : 1 - s.pHigh;
        hidden[t] = p;
        u[t] = rng.nextDouble() < p ? 1 : 0;
      }
    case DataModel.randomWalk:
      // A random walk whose step size switches between calm and volatile.
      var x = 0.0;
      for (var t = 0; t < n; t++) {
        x += ((t ~/ s.block).isEven ? s.sdLow : s.sdHigh) * _gauss(rng);
        hidden[t] = x;
        u[t] = x + s.noise * _gauss(rng);
      }
  }
  return SimulatedData(u, [hidden]);
}

/// One-hot rows (`nTime × n`) for category indices; a missing (non-finite)
/// observation gets a row of zeros.
Float64List oneHot(Float64List categories, int n) {
  final out = Float64List(categories.length * n);
  for (var t = 0; t < categories.length; t++) {
    final c = categories[t];
    if (c.isFinite) out[t * n + c.toInt()] = 1;
  }
  return out;
}

/// Parse one column of numbers from CSV or plain text: the first numeric
/// column of each line; a header line and blank lines are skipped, `NaN`
/// or an empty field marks a missing observation.
Float64List parseColumn(String text, {int column = 0}) {
  final out = <double>[];
  for (final raw in const LineSplitter().convert(text)) {
    final line = raw.trim();
    if (line.isEmpty || line.startsWith('#')) continue;
    final cells = line.split(RegExp(r'[,;\t ]+'));
    if (column >= cells.length) continue;
    final cell = cells[column].trim();
    final v = cell.isEmpty ? double.nan : double.tryParse(cell);
    if (v == null) {
      if (out.isEmpty) continue; // a header
      throw FormatException('Not a number on line ${out.length + 1}', cell);
    }
    out.add(v);
  }
  return Float64List.fromList(out);
}
