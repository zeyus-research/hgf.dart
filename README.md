# hgf.dart

[![CI](https://github.com/zeyus-research/hgf.dart/actions/workflows/ci.yml/badge.svg)](https://github.com/zeyus-research/hgf.dart/actions/workflows/ci.yml)

**[Try the playground](https://zeyus-research.github.io/hgf.dart/)** (Wasm; needs a recent Chrome, Firefox or Safari, or falls back to JS).

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
| `packages/hgf_widgets` | Flutter widgets: trajectory plots (10k trials per frame) and a network graph |
| `apps/hgf_playground` | the interactive playground (web/Wasm first; desktop and mobile) |
| `tool/parity` | pyhgf fixture generator and Python benchmarks (a `uv` project using `../pyhgf`) |
| `tool/rs_native_bench` | rshgf timed natively, without Python |
| `tool/bench_compare.dart` | runs every runtime and writes `docs/benchmark_results.md` |
| `docs/` | feasibility report, roadmap, benchmark results |

Planned, and created only when its phase starts: `packages/hgf_fit`
(gradients, optimisers, samplers).

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
```

Fixtures and benchmarks expect a pyhgf checkout next to this repository
(`../pyhgf`). See [tool/parity/README.md](tool/parity/README.md).

## License

MIT. See [LICENSE](LICENSE).
