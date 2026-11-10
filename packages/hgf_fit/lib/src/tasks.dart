import 'dart:math' as math;
import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'response_model.dart';

/// The Iowa Gambling Task (Bechara et al., 1994): four decks, each card
/// pays a fixed win and sometimes a loss. A and B win more per card but
/// lose more in the long run (−$25 per card on average), C and D win less
/// and gain in the long run (+$25 per card). A and C lose often (on half the
/// cards), B and D rarely (one card in ten) but big.
///
/// | deck | win | loss (probability) | net per card |
/// |---|--:|---|--:|
/// | A | $100 | $150–350 (0.5) | −$25 |
/// | B | $100 | $1250 (0.1) | −$25 |
/// | C | $50 | $25–75 (0.5) | +$25 |
/// | D | $50 | $250 (0.1) | +$25 |
///
/// The original task uses fixed card sequences; [outcomes] draws the cards
/// at random with these probabilities. Outcomes are the net payoff of a
/// card (win minus loss) in units of [scale] dollars, $100 by default, so
/// they sit on the scale of a continuous HGF's default parameters.
///
/// The model ([network], [readout], [reveals]) learns each deck's expected
/// payoff with its own two-level continuous HGF, observes only the chosen
/// deck's payoff on each trial, and chooses by a softmax over the four
/// predicted payoffs. Fit it to choices with `HgfModel` (inputs: the
/// chosen deck's payoff in its column, the others marked unobserved;
/// responses: the deck index), or simulate agents with
/// `HgfModel.simulateChoices(…, reveals: IowaGamblingTask.reveals)` on
/// the full [outcomes].
abstract final class IowaGamblingTask {
  /// Deck names, in input order.
  static const decks = ['A', 'B', 'C', 'D'];

  /// The net payoff of a card from each deck, `nTrials × 4` row-major (the
  /// layout of `Subject.inputs` for [network]), in units of [scale]
  /// dollars.
  static Float64List outcomes(
    int nTrials,
    math.Random rng, {
    double scale = 100,
  }) {
    final out = Float64List(nTrials * 4);
    for (var t = 0; t < nTrials; t++) {
      // Draw the four cards in a fixed order, so a seed gives one table.
      final a = rng.nextDouble() < 0.5 ? 150.0 + 50 * rng.nextInt(5) : 0.0;
      final b = rng.nextDouble() < 0.1 ? 1250.0 : 0.0;
      final c = rng.nextDouble() < 0.5 ? 25.0 + 25 * rng.nextInt(3) : 0.0;
      final d = rng.nextDouble() < 0.1 ? 250.0 : 0.0;
      out[t * 4] = 100 - a;
      out[t * 4 + 1] = 100 - b;
      out[t * 4 + 2] = 50 - c;
      out[t * 4 + 3] = 50 - d;
    }
    for (var i = 0; i < out.length; i++) {
      out[i] /= scale;
    }
    return out;
  }

  /// Four two-level continuous HGFs, one per deck: input nodes 0–3 (the
  /// decks' payoffs, observed with [precision]) and their value parents
  /// 4–7 (each deck's expected payoff, drifting with [tonicVolatility]).
  static Network network({
    double tonicVolatility = -2,
    double precision = 1,
    double initialPrecision = 1,
  }) {
    final net = Network()..addNodes(nNodes: 4, precision: precision);
    for (var d = 0; d < 4; d++) {
      net.addNodes(
        valueChildren: [d],
        tonicVolatility: tonicVolatility,
        precision: initialPrecision,
      );
    }
    return net;
  }

  /// The nodes holding each deck's expected payoff, in [network].
  static const valueNodes = [4, 5, 6, 7];

  /// Choosing deck `k` reveals input `k` only.
  static const reveals = [
    [0],
    [1],
    [2],
    [3],
  ];

  /// A softmax over the four predicted payoffs (the input nodes'
  /// `expectedMean`), inverse temperature [inverseTemperature].
  static SoftmaxReadout readout({double inverseTemperature = 1.0}) =>
      SoftmaxReadout.nodes(const [
        0,
        1,
        2,
        3,
      ], inverseTemperature: inverseTemperature);

  /// The model's inputs and observed flags for recorded data: the chosen
  /// deck's payoff ([payoffs], one per trial, in [scale] dollars) in its
  /// column, the other decks unobserved. [choices] are deck indices (NaN:
  /// no choice, nothing observed).
  static ({Float64List inputs, Float64List observed}) data(
    List<double> choices,
    List<double> payoffs, {
    double scale = 100,
  }) {
    if (choices.length != payoffs.length) {
      throw ArgumentError('One payoff per choice');
    }
    final n = choices.length;
    final inputs = Float64List(n * 4), observed = Float64List(n * 4);
    for (var t = 0; t < n; t++) {
      final c = choices[t];
      if (c.isNaN || payoffs[t].isNaN) continue;
      final k = c.toInt();
      if (k < 0 || k > 3 || k != c) {
        throw ArgumentError.value(c, 'choices[$t]', 'Not a deck index 0-3');
      }
      inputs[t * 4 + k] = payoffs[t] / scale;
      observed[t * 4 + k] = 1;
    }
    return (inputs: inputs, observed: observed);
  }
}
