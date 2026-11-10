# hgf_deep

Deep HGF networks in pure Dart: layers of volatile, binary and categorical
nodes connected by learned weights, trained by belief propagation. A port of
pyhgf's vectorised `DeepNetwork`
([pyhgf](https://github.com/ComputationalPsychiatry/pyhgf)), checked against
its JAX backend in float64.

```dart
import 'package:hgf_deep/hgf_deep.dart';

final net = DeepNetwork()
  ..addLayer(4)                            // output (observations y)
  ..addLayer(16, coupling: CouplingFn.tanh)
  ..addLayer(2)                            // input (predictors x)
  ..weightInitialisation(const XavierInit(), seed: 1);

final predictions = net.fit(x, y, optimiser: Adam(1e-3)); // one sample at a time
final errors = net.batchUpdate(xBatch, yBatch, optimiser: Adam(1e-3));
final yHat = net.predict(xNew);
```

Layers are added bottom (output) first. Each step predicts every layer from
the one above (`W @ g(μ̂_parent)`), clamps the observations at the bottom,
propagates prediction errors up through each layer's value and volatility
posteriors, and learns the weights from the settled beliefs.

## Getting started with your own data

`example/workflow.dart` is a template for the usual steps, on the UCI wine
data (`example/data/wine.csv`; test accuracy 0.97). In short:

```dart
import 'dart:convert';
import 'dart:io';

import 'package:hgf_deep/datasets.dart';
import 'package:hgf_deep/hgf_deep.dart';

// Load: a CSV with a header row. Every column but the target is a feature
// (or pass `features:`); text labels become classes 0, 1, … (`classNames`);
// rows with missing values are dropped. A real-valued target, or several,
// make a regression set.
final data = Dataset.fromCsv(
  File('wine.csv').readAsStringSync(),
  target: ['cultivar'],
);

// Split into training, validation and test sets; scale the inputs with the
// training set's statistics.
final (train: rest, test: testRaw) = split(data, testFraction: 0.2, seed: 1);
final (train: trainRaw, test: validRaw) = split(rest, testFraction: 0.25);
final scale = Standardiser.fit(trainRaw.x);
final train = trainRaw.withX(scale.apply(trainRaw.x));
final valid = validRaw.withX(scale.apply(validRaw.x));

final net = DeepNetwork()
  ..addLayer(data.classes, kind: LayerKind.categorical) // output first
  ..addLayer(16, coupling: CouplingFn.tanh, volatilityParent: false)
  ..addLayer(data.x.cols)                                // inputs last
  ..weightInitialisation(const XavierInit(), seed: 1);

// Train: one sample at a time (`fit`), or in minibatches (`batchUpdate`).
// Keep the best network on the validation set.
final adam = Adam(1e-3);
var best = double.infinity;
DeepSnapshot? bestNet;
for (var epoch = 1; epoch <= 60; epoch++) {
  final e = shuffled(train, seed: epoch);
  net.fit(e.x, e.y, optimiser: adam);
  // or: for (final b in minibatches(train, 16, seed: epoch)) {
  //       net.batchUpdate(b.x, b.y, optimiser: adam);
  //     }
  final loss = crossEntropy(net.predict(valid.x), valid.y);
  if (loss < best) (best, bestNet) = (loss, net.snapshot());
}
net.restore(bestNet!);

// Save the network with its input scaling, and load it back.
File('model.json').writeAsStringSync(
  jsonEncode({'network': net.toJson(), 'scale': scale.toJson()}),
);
final saved = jsonDecode(File('model.json').readAsStringSync()) as Map;
final model = DeepNetwork.fromJson(saved['network'] as Map<String, Object?>);
final s = Standardiser.fromJson(saved['scale'] as Map<String, Object?>);
final p = model.predict(s.apply(newRows)); // class probabilities per row
```

| task | tools |
|---|---|
| load data | `Dataset.fromCsv` (pure Dart: read the text with `dart:io`, `fetch` or a file picker), `CsvData`, or build `Matrix`es yourself (`Matrix.fromRows`) |
| split, shuffle, batch | `split`, `shuffled`, `minibatches`, `Dataset.subset` |
| scale inputs | `Standardiser` (`fit` on the training set, `apply` to every set, `toJson` to save) |
| train | `fit` (sequential, one sample at a time, as pyhgf's `fit`), `batchUpdate` (one step per minibatch), any `Optimiser` (`Sgd`, `Adam`) |
| early stopping, checkpoints | `snapshot` / `restore` (in memory), `toJson` / `DeepNetwork.fromJson` (to disk) |
| evaluate | `accuracy`, `confusionMatrix` + `formatConfusion`, `crossEntropy`, `mse`, `r2`, `predictedClasses` |
| output layer | classification: `LayerKind.categorical` (3+ classes) or `LayerKind.binary` (two classes, one column); regression: the default volatile layer |
| on the GPU | `GpuDeepNetwork` (minibatches, `trainEpoch`) and `GpuDeepEnsemble` (many sequential fits at once), in `hgf_gpu` |

`toJson` saves the layers, settings, weights, per-node parameters and the
current beliefs, so a loaded network predicts and keeps learning bit for bit
like the original; the optimiser's moments are not saved (training goes on
with a fresh one).

## Examples

Run from `packages/hgf_deep` (`melos run examples:deep` runs them all
briefly):

| example | task | test score |
|---|---|--:|
| `toy_classification.dart [xor\|moons\|circles\|spirals\|all]` | 2-D classification, binary or 3-way categorical output; prints a confusion matrix and an ASCII map of the decision regions | XOR 1.000, moons 0.987, circles 1.000, spirals 1.000 |
| `regression.dart` | noisy `sin(x)`, 1-16-16-1 tanh | R² 0.963 (noise floor MSE 0.010, reached 0.018) |
| `digits.dart` | UCI handwritten digits (8×8, ten classes), 64-32-10 | accuracy 0.969 |
| `workflow.dart [--minibatch] [--epochs n]` | your own CSV, end to end: load, split, standardise, train with early stopping, evaluate, save and reload (UCI wine, 13 features, three cultivars) | accuracy 0.972 |
| `volatility_effect.dart` | the same networks with and without hidden volatility levels, under each update scheme, with and without the fixes below | see output |
| `../hgf_gpu/example/deep_digits_gpu.dart` | digits on the GPU: minibatches against the CPU, and 64 seeds fitted sequentially at once | 0.976; 0.964 mean over seeds |

```
$ dart run example/toy_classification.dart moons 10
== moons: 450 train / 150 test, 2 → 16 → 16 → 1 binary
epoch  train loss  train acc  test loss  test acc      ms
    1      0.3355      0.869     0.2465     0.900      17
    2      0.1673      0.940     0.0656     0.993       4
    3      0.0781      0.976     0.0285     0.993       3
```

`package:hgf_deep/datasets.dart` holds the toy generators (`xor`, `moons`,
`circles`, `spirals`, `sine`), `split`, `shuffled`, `Standardiser`, and the
scores (`accuracy`, `confusionMatrix`, `mse`, `r2`, `crossEntropy`).

## Inspecting one step

`trace` runs one sample exactly as a row of `fit` and records the network
after every phase: the input clamp, each layer's top-down prediction, the
observation, each layer's posterior and errors bottom-up, the weight step
(with the gradients) and the carry. `snapshot` and `restore` undo it:

```dart
final snap = net.snapshot();
final trace = net.trace(x, y: y, optimiser: adam.copy());
for (final step in trace.steps) {
  print('${step.phase.name} ${step.layer}: ${step.states[1].mean}');
}
net.restore(snap);
```

## Fixes to pyhgf

Two corrections are on by default (`DeepFixes.recommended`); both are
proposed upstream ([P18, P19](../../docs/upstream/README.md#p18)):

- `curvatureFloor`: a hidden node's posterior precision is floored at its
  prior plus the evidence, not the prior alone, so negative curvature cannot
  turn a weak prior into an unbounded step. This is what ran pyhgf's
  volatility levels away (P18).
- `logSpaceVolatility`: the volatility terms are evaluated where they
  overflow (same values elsewhere), so a runaway node stays finite instead of
  turning NaN.

`DeepNetwork(fixes: DeepFixes.none)`, `DeepNetwork.defaultFixes` or
`-DHGF_DEEP_UPSTREAM=true` reproduce pyhgf bit for bit; the parity tests use
that. Even with the fixes, hidden volatility levels destabilise learning on
some tasks (they read large posterior steps as volatility); the examples use
hidden layers without one (`volatilityParent: false`) and
`volatility_effect.dart` shows the difference.

## Parity

`test/parity_test.dart` runs 30 scenarios (pyhgf `1e6e98d7`) generated by
`tool/parity/gen_deep_fixtures.py` (`melos run fixtures:deep`): every
volatility update, linear and curved couplings, binary and categorical
leaves, a binary interior layer, feedforward uncertainty, mean-field
updates, frozen or uncarried precisions, tonic volatility, one-to-one
layers, initial overrides, SGD and Adam (continued across calls), and
batched steps, and four patched references for `curvatureFloor`.
Predictions, recorded trajectories, input errors, weights
and every belief field after each operation match pyhgf's JAX backend
(float64) to 1e-9 relative, NaN included.

Some scenarios go NaN in pyhgf itself (the volatility runaway,
[P18](../../docs/upstream/README.md#p18)); `tool/nan_probe.dart` and
`tool/vol_trace.dart` show where. rshgf's `DeepNetwork` differs from JAX in
its batched step ([R4](../../docs/upstream/README.md#r4)), panics with
`predict_precision=False` ([R5](../../docs/upstream/README.md#r5)) and
approximates GELU ([R3](../../docs/upstream/README.md#r3)); hgf_deep follows
JAX.

## Performance

`melos run bench:deep` (one core, i7-7700; tanh networks, Adam):

| | hgf_deep (AOT) | pyhgf JAX f64 | JAX f32 | rshgf f32 |
|---|--:|--:|--:|--:|
| fit, 2-8-8-4, 2000 samples | 18.8 ms | 29.6 | 29.4 | 22.4 |
| fit, 10-64-64-16, 2000 samples | 266 ms | 130 | 83 | 325 |
| fit, 16-256-256-32, 500 samples | 640 ms | 293 | 186 | 143 |
| batchUpdate, 10-64-64-16, 256 | 21.7 ms | 5.9 | 3.3 | 15.4 |
| predict, 10-64-64-16, 2000 | 14.4 ms | 2.7 | 1.3 | 5.5 |

Sequential fitting is within ~2× of JAX and ahead on small networks;
batched and dense work trails BLAS-backed JAX by 4–8× (scalar code, no
SIMD). The GPU kernels (next, in hgf_gpu) are the route for large batches.
