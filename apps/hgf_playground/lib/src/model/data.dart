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
}

/// Simulated observations, and the hidden state that generated them.
class SimulatedData {
  SimulatedData(this.observations, this.hidden);

  final Float64List observations;

  /// Binary: P(u = 1) per trial. Continuous: the hidden value.
  final Float64List hidden;

  int get length => observations.length;
}

double _gauss(math.Random r) =>
    math.sqrt(-2 * math.log(1 - r.nextDouble())) *
    math.cos(2 * math.pi * r.nextDouble());

SimulatedData simulate(InputKind kind, DataSettings s) {
  final rng = math.Random(s.seed);
  final n = s.nTrials;
  final u = Float64List(n), hidden = Float64List(n);
  switch (kind) {
    case InputKind.binary:
      // Reversal learning: P(u = 1) flips between pHigh and 1 − pHigh.
      for (var t = 0; t < n; t++) {
        final p = (t ~/ s.block).isEven ? s.pHigh : 1 - s.pHigh;
        hidden[t] = p;
        u[t] = rng.nextDouble() < p ? 1 : 0;
      }
    case InputKind.continuous:
      // A random walk whose step size switches between calm and volatile.
      var x = 0.0;
      for (var t = 0; t < n; t++) {
        x += ((t ~/ s.block).isEven ? s.sdLow : s.sdHigh) * _gauss(rng);
        hidden[t] = x;
        u[t] = x + s.noise * _gauss(rng);
      }
  }
  return SimulatedData(u, hidden);
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
