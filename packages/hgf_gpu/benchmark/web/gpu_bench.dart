/// Phase 7 GPU probe in the browser (see `../src/probe.dart`): WebGPU,
/// against `BatchNetwork` on the page's main thread. Results go to the
/// console as `HGF_GPU {json}` lines and to `globalThis.hgfResults` (see
/// `tool/gpu_bench.py`). URL parameters: `sizes=1000,4000,...`,
/// `trials=1000`, `scenarios=c2_unbounded,...`, `chunk=0` (trials per
/// dispatch), `repeat=5`, `cpuRepeat=3`, `poke=0` (no queue polling while
/// waiting, see `WebGpuBackend.pokeQueue`).
library;

import 'dart:convert';
import 'dart:js_interop';
import 'package:hgf_gpu/web.dart';

import '../src/probe.dart';

@JS('fetch')
external JSPromise<_Response> _fetch(String url);

extension type _Response._(JSObject _) implements JSObject {
  external JSPromise<JSString> text();
}

@JS('location.search')
external String get _search;

@JS('hgfResults')
external JSArray<JSString> get _results;

/// Logs [m] and appends it to `globalThis.hgfResults`, which the driver
/// reads at the end (Playwright can drop console events).
void emit(Map<String, Object?> m) {
  final line = jsonEncode(m, toEncodable: (o) => o is double ? '$o' : o);
  _results.add(line.toJS);
  print('HGF_GPU $line');
}

Future<void> main() async {
  try {
    final backend = await WebGpuBackend.create();
    if (backend == null) {
      emit({'error': 'no WebGPU adapter'});
    } else {
      final inputs =
          jsonDecode(
                (await (await _fetch(
                  'inputs.json',
                ).toDart).text().toDart).toDart,
              )
              as Map<String, Object?>;
      final params = Uri.splitQueryString(
        _search.isEmpty ? '' : _search.substring(1),
      );
      backend.pokeQueue = params['poke'] != '0';
      emit({'adapter': backend.description, 'poke': backend.pokeQueue});
      await runProbe(backend, inputs, params, emit);
    }
  } catch (e, st) {
    emit({'error': '$e', 'stack': '$st'});
  }
  emit({'done': true});
}
