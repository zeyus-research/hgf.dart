# hgf.dart

[![CI](https://github.com/zeyus-research/hgf.dart/actions/workflows/ci.yml/badge.svg)](https://github.com/zeyus-research/hgf.dart/actions/workflows/ci.yml)

**[Try the playground](https://zeyus-research.github.io/hgf.dart/)** (Wasm; needs a recent Chrome, Firefox or Safari, or falls back to JS).
Every node kind has a preset; link to one with `?preset=binary3`,
`continuous2`, `continuous3`, `categorical3`, `dp` or `ef` (and `&trials=N`).

Hierarchical Gaussian Filter (HGF) networks in Dart and Flutter. This is a
pure-Dart port of the nodalised engine of
[pyhgf](https://github.com/ComputationalPsychiatry/pyhgf), checked numerically
against pyhgf, and it runs on the Dart VM, natively (AOT), and on the web
(JS/Wasm).

> Status: feasibility spike. See [docs/FEASIBILITY.md](docs/FEASIBILITY.md)
> for the parity and performance results, and what is and isn't ported yet,
> and [docs/ROADMAP.md](docs/ROADMAP.md) for what's next.

```dart
import 'package:hgf/hgf.dart';

final hgf = Network()                              // unbounded volatility updates
  ..addNodes(precision: 1e4)                       // 0: continuous input
  ..addNodes(valueChildren: [0], mean: 1.04)       // 1: value parent
  ..addNodes(volatilityChildren: [1]);             // 2: volatility parent

final traj = hgf.inputSeries(prices);              // whole series
final mu1 = traj.series(1, NodeField.mean);        // Float64List view

hgf.reset();
hgf.step(Float64List.fromList([1.02]));            // or one trial at a time
```

## Repository layout

A [melos](https://melos.invertase.dev) / pub workspace monorepo, laid out like
[liblsl.dart](https://github.com/NexusDynamic/liblsl.dart): each package and
app builds, tests and publishes on its own.

| path | what |
|---|---|
| `packages/hgf` | the core engine (pure Dart, web-safe) |
| `packages/hgf_fit` | parameter estimation: priors, response models, batched MAP fitting (L-BFGS, Nelder–Mead) |
| `packages/hgf_deep` | deep HGF networks (pyhgf's vectorised `DeepNetwork`): layers of volatile, binary and categorical nodes with learned weights; f64, matches JAX to 1e-9 |
| `packages/hgf_gpu` | GPU kernels (phase 7): a `Network` compiled to a WGSL compute shader, run on WebGPU (browser) or wgpu-native (desktop); f32 |
| `packages/hgf_widgets` | Flutter widgets: trajectory plots (10k trials per frame) and a network graph |
| `apps/hgf_playground` | the interactive playground (web/Wasm first; desktop and mobile) |
| `tool/parity` | pyhgf fixture generator and Python benchmarks (a `uv` project using `../pyhgf`) |
| `tool/rs_native_bench` | rshgf timed natively, without Python |
| `tool/bench_compare.dart` | runs every runtime and writes `docs/benchmark_results.md` |
| `docs/` | feasibility report, roadmap, benchmark results |

## Development

```sh
dart pub get                     # or: melos bootstrap
melos run lint:all               # analyze + format check
melos run test                   # unit + parity tests (Dart VM)
CHROME_EXECUTABLE=/usr/bin/chromium melos run test:web   # parity on dart2js/dart2wasm
melos run fixtures               # regenerate pyhgf fixtures (needs ../pyhgf, uv, cargo)
melos run bench                  # benchmarks: Dart AOT/JIT/JS/Wasm vs JAX vs rshgf
melos run build:playground       # playground web build (Wasm, -O4)
melos run bench:playground       # its frame times in headless Chromium (uv + playwright)
melos run bench:playground-fit   # its fitting view: times, frames, recovery
```

Fixtures and benchmarks expect a pyhgf checkout next to this repository
(`../pyhgf`). See [tool/parity/README.md](tool/parity/README.md).

## License

MIT. See [LICENSE](LICENSE).
