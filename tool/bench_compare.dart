/// Run every benchmark runtime on the same scenarios and print a comparison.
///
///     dart run tool/bench_compare.dart [--repeat 7] [--rust] [--gpu] [--no-python] [--no-web]
///         [--out docs/benchmark_results.md]
///     dart run tool/bench_compare.dart --report-only   # rewrite the markdown
///                                                      # from the last JSON
///
/// * Python: pyhgf JAX (float64, CPU) and rshgf, via `uv` in tool/parity.
/// * Rust (`--rust`): rshgf linked directly (tool/rs_native_bench), the
///   native floor without Python conversion overhead. Needs cargo and the
///   ../pyhgf checkout.
/// * Dart: AOT (`dart compile exe`), JIT (`dart run`), and JS / Wasm under
///   Node (needs `node` on PATH).
/// * GPU (`--gpu`): hgf_gpu's f32 kernel on the batch scenario, natively on
///   wgpu-native (needs libwgpu_native, or `WGPU_LIB`) and, unless
///   `--no-web`, on WebGPU in headless Chromium (dart2wasm page; needs
///   `CHROMIUM` or /usr/bin/chromium with Vulkan). End to end: upload of
///   new ω, dispatch, readback.
///
/// * JAX on the GPU (`--jax-gpu PYTHON`): bench_python.py's JAX runs again
///   with PYTHON, an interpreter whose JAX has a GPU plugin (ROCm or CUDA)
///   and pyhgf installed; see tool/parity/README.md.
///
/// Run `python tool/parity/bench_plot.py` (in the parity uv project)
/// afterwards to redraw `docs/benchmark.svg` from the JSON.
///
/// Writes `.dart_tool/hgf_bench/bench_all.json` (customSmallerIsBetter, for
/// github-action-benchmark) and `docs/benchmark_results.md`.
library;

import 'dart:convert';
import 'dart:io';

final root = File.fromUri(Platform.script).parent.parent.path;
final pkg = '$root/packages/hgf';
final out = Directory('$root/.dart_tool/hgf_bench');

Future<void> main(List<String> args) async {
  var repeat = 7;
  var python = true, web = true, rust = false, gpu = false;
  var reportOnly = false;
  String? jaxGpu;
  var report = '$root/docs/benchmark_results.md';
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--repeat':
        repeat = int.parse(args[++i]);
      case '--no-python':
        python = false;
      case '--no-web':
        web = false;
      case '--rust':
        rust = true;
      case '--gpu':
        gpu = true;
      case '--jax-gpu':
        jaxGpu = File(args[++i]).absolute.path;
      case '--report-only':
        reportOnly = true;
      case '--out':
        report = File(args[++i]).absolute.path;
      default:
        stderr.writeln('Unknown argument ${args[i]}');
        exit(64);
    }
  }
  out.createSync(recursive: true);
  final results = <Map<String, Object?>>[];
  if (reportOnly) {
    results.addAll(_readJson('${out.path}/bench_all.json'));
    for (final r in results) {
      if ((r['name'] as String).contains('/gpu_')) {
        gpu = true;
        _gpuName ??= r['extra'] as String?;
      }
    }
    await _writeReport(results, repeat: repeat, gpu: gpu, path: report);
    return;
  }

  final inputs = File('$pkg/benchmark/data/inputs.json');
  if (!inputs.existsSync() || python) {
    // The Python side owns the inputs (it writes them when missing).
    if (python) {
      final pyOut = '${out.path}/bench_python.json';
      await _run('uv', [
        'run',
        'python',
        'bench_python.py',
        '--repeat',
        '$repeat',
        '--out',
        pyOut,
      ], cwd: '$root/tool/parity');
      results.addAll(_readJson(pyOut));
      // pyhgf's default precision: JAX in float32 (rshgf is always float64).
      final pyOut32 = '${out.path}/bench_python_f32.json';
      await _run(
        'uv',
        [
          'run',
          'python',
          'bench_python.py',
          '--repeat',
          '$repeat',
          '--jax-only',
          '--out',
          pyOut32,
        ],
        cwd: '$root/tool/parity',
        environment: {'JAX_ENABLE_X64': '0'},
      );
      results.addAll(_readJson(pyOut32));
    } else {
      stderr.writeln('Missing ${inputs.path}; run without --no-python once.');
      exit(1);
    }
  }

  if (jaxGpu != null) {
    for (final x64 in ['1', '0']) {
      final o = '${out.path}/bench_jax_gpu_$x64.json';
      await _run(
        jaxGpu,
        ['bench_python.py', '--repeat', '$repeat', '--jax-only', '--out', o],
        cwd: '$root/tool/parity',
        environment: {'JAX_ENABLE_X64': x64, 'JAX_PLATFORMS': ''},
      );
      final rs = _readJson(o);
      if (!rs.any((r) => (r['name'] as String).contains('_gpu'))) {
        stderr.writeln('$jaxGpu: JAX found no GPU');
        exit(1);
      }
      results.addAll(rs);
    }
  }

  if (rust) results.addAll(await _rustNative(inputs, repeat));

  // Dart AOT.
  final exe = '${out.path}/hgf_bench';
  await _run('dart', [
    'compile',
    'exe',
    'benchmark/bin/hgf_benchmark.dart',
    '-o',
    exe,
  ], cwd: pkg);
  await _run(exe, [
    '--repeat',
    '$repeat',
    '--out',
    '${out.path}/bench_aot.json',
  ], cwd: pkg);
  results.addAll(_readJson('${out.path}/bench_aot.json'));

  // Dart JIT.
  await _run('dart', [
    'run',
    'benchmark/bin/hgf_benchmark.dart',
    '--runtime',
    'dart_jit',
    '--repeat',
    '$repeat',
    '--out',
    '${out.path}/bench_jit.json',
  ], cwd: pkg);
  results.addAll(_readJson('${out.path}/bench_jit.json'));

  if (web) {
    final js = '${out.path}/web/bench.js';
    await _run('dart', [
      'compile',
      'js',
      '-O4',
      'benchmark/bin/web_benchmark.dart',
      '-o',
      js,
    ], cwd: pkg);
    results.addAll(
      _parseBench(
        await _run('node', ['$root/tool/web/run_js.cjs', js], cwd: pkg),
      ),
    );
    final wasm = '${out.path}/web/bench.wasm';
    await _run('dart', [
      'compile',
      'wasm',
      '-O4',
      'benchmark/bin/web_benchmark.dart',
      '-o',
      wasm,
    ], cwd: pkg);
    results.addAll(
      _parseBench(
        await _run('node', ['$root/tool/web/run_wasm.mjs', wasm], cwd: pkg),
      ),
    );
  }

  if (gpu) results.addAll(await _gpu(repeat, web: web));

  File(
    '${out.path}/bench_all.json',
  ).writeAsStringSync(const JsonEncoder.withIndent(' ').convert(results));
  await _writeReport(results, repeat: repeat, gpu: gpu, path: report);
}

Future<void> _writeReport(
  List<Map<String, Object?>> results, {
  required int repeat,
  required bool gpu,
  required String path,
}) async {
  final table = _table(results);
  final md = File(path);
  md.parent.createSync(recursive: true);
  md.writeAsStringSync(
    '# Benchmark results\n\n'
    'Generated by `dart run tool/bench_compare.dart` on '
    '${DateTime.now().toIso8601String().substring(0, 10)}.\n\n'
    '- Machine: ${await _cpu()}, ${Platform.numberOfProcessors} threads, '
    '${Platform.operatingSystem}\n'
    '- Dart ${Platform.version.split(' ').first}; '
    'Node ${(await _capture('node', ['--version'])).trim()}\n'
    '- Median of $repeat warm runs, milliseconds (lower is better). JAX on '
    'CPU in float64 (`jax_*`, like-for-like with the float64 Dart port) and '
    'in float32 (`jax_*_f32`, pyhgf\'s default); `jax_first_call*` is trace '
    '+ compile + first run. `*_no_traj` skips trajectory recording.\n'
    '${gpu ? '- `gpu_*_f32`: hgf_gpu\'s WGSL kernel (f32, one thread per ω) '
              'on ${_gpuName ?? 'the GPU'}, end to end (ω upload, dispatch, '
              'readback); `gpu_webgpu_f32` is WebGPU in headless Chromium '
              '(dart2wasm).\n' : ''}'
    '${results.any((r) => (r['name'] as String).contains('_gpu')) ? '- `jax_*_gpu*`: '
              'pyhgf JAX on the same GPU (${_jaxGpu(results)}; '
              '`tool/parity/jax_rocm.sh`), timed to results in host memory. '
              'Its `scan` launches kernels every trial, which is why the '
              'series rows are slow there.\n' : ''}'
    '\n<picture>\n'
    '  <source media="(prefers-color-scheme: dark)" '
    'srcset="benchmark-dark.svg">\n'
    '  <img alt="Benchmark plot: milliseconds per backend for each scenario" '
    'src="benchmark.svg">\n'
    '</picture>\n\n'
    'Redraw with `melos run bench:plot`.\n\n'
    '$table\n'
    '${_nutsSection()}${_deepSection()}',
  );
  stdout
    ..writeln()
    ..writeln(table);
}

String _picture(String name, String alt) =>
    '<picture>\n'
    '  <source media="(prefers-color-scheme: dark)" srcset="$name-dark.svg">\n'
    '  <img alt="$alt" src="$name.svg">\n'
    '</picture>\n\n';

List<Map<String, Object?>> _jsonl(String path) {
  final f = File(path);
  if (!f.existsSync()) return [];
  return [
    for (final l in f.readAsLinesSync())
      if (l.trim().isNotEmpty) jsonDecode(l) as Map<String, Object?>,
  ];
}

/// `melos run bench:nuts-sweep`'s results, if any.
String _nutsSection() {
  final dir = '$root/.dart_tool/nuts_sweep';
  final rows = [..._jsonl('$dir/dart.jsonl'), ..._jsonl('$dir/python.jsonl')];
  if (rows.isEmpty) return '';
  const order = [
    'hgf_cpu1',
    'hgf_pool8',
    'hgf_gpu',
    'numpyro_cpu_f64',
    'numpyro_gpu_f32',
    'pymc_cpu_f64',
  ];
  int rank(Object? b) {
    final i = order.indexOf(b! as String);
    return i < 0 ? order.length : i;
  }

  rows.sort((a, b) {
    for (final k in ['model', 'agents', 'chains']) {
      final c = Comparable.compare(a[k]! as Comparable, b[k]! as Comparable);
      if (c != 0) return c;
    }
    return rank(a['backend']) - rank(b['backend']);
  });
  String n(Object? v, [int digits = 1]) =>
      v is num ? v.toStringAsFixed(digits) : '';
  final b = StringBuffer()
    ..writeln('## NUTS (`melos run bench:nuts-sweep`)\n')
    ..writeln(
      'Binary (b) and continuous (c) 2- and 3-level HGFs, 320 trials per '
      'agent, ${rows.first['warmup']} warmup + ${rows.first['draws']} draws '
      'per chain. hgf_fit samples each agent (`hgf_cpu1`: one isolate; '
      '`hgf_pool8`: `FitWorkers`, 8 isolates; `hgf_gpu`: `GpuFitter`, f32). '
      'NumPyro and PyMC sample all agents jointly, the same posterior, on '
      'the pyhgf model that hgf_fit\'s `PythonExport` writes for each Dart '
      'model; CPU runs are pinned to one core. Seconds are sampling time '
      '(NumPyro\'s compile, measured on a short run, is subtracted); ESS is '
      'the minimum bulk ESS over all agents\' parameters. JAX on this GPU '
      'needs ~40 ms per gradient of 20 agents (0.6 ms on the CPU: its scan '
      'launches kernels every trial), so NumPyro runs on the GPU for one '
      'cell only.\n',
    )
    ..writeln(
      _picture('benchmark_nuts', 'NUTS: min bulk ESS per second by backend'),
    )
    ..writeln(
      '| model | agents | chains | backend | s | min ESS | min ESS/s | '
      'max R̂ | divergences |',
    )
    ..writeln('|---|--:|--:|---|--:|--:|--:|--:|--:|');
  for (final r in rows) {
    final timeout = r['timeout_s'];
    b.writeln(
      '| ${r['model']} | ${r['agents']} | ${r['chains']} | ${r['backend']} | '
      '${timeout != null ? '> $timeout' : n((r['ms']! as num) / 1e3)} | '
      '${n(r['min_ess'], 0)} | ${n(r['min_ess_per_s'])} | '
      '${n(r['max_rhat'], 3)} | ${r['divergences'] ?? ''} |',
    );
  }
  return '\n$b';
}

/// `melos run bench:deep-sweep`'s results, if any.
String _deepSection() {
  final f = File('$root/.dart_tool/deep_sweep.json');
  if (!f.existsSync()) return '';
  final problems = [
    for (final p in jsonDecode(f.readAsStringSync()) as List<Object?>)
      p! as Map<String, Object?>,
  ];
  final rows = _jsonl('$root/.dart_tool/deep_sweep_rows.jsonl');
  // backend -> op -> width -> ms
  final v = <String, Map<String, Map<int, double>>>{};
  for (final p in problems) {
    for (final MapEntry(:key, :value) in (p['dart_ms']! as Map).entries) {
      final k = key as String, i = k.indexOf('_');
      final backend = k.substring(0, i) == 'cpu' ? 'hgf_deep' : 'hgf_gpu_f32';
      ((v[backend] ??= {})[k.substring(i + 1)] ??= {})[p['width']! as int] =
          (value as num).toDouble();
    }
  }
  for (final r in rows) {
    for (final op in ['fit', 'batch_update', 'predict']) {
      ((v[r['backend']! as String] ??= {})[op] ??= {})[r['width']! as int] =
          (r[op]! as num).toDouble();
    }
  }
  final widths = [for (final p in problems) p['width']! as int];
  final first = problems.first;
  final b = StringBuffer()
    ..writeln('## Deep HGF networks (`melos run bench:deep-sweep`)\n')
    ..writeln(
      'tanh networks [4, H, H, 16] (output layer first), Adam, best of '
      'several runs, ms: `fit` is sequential over '
      '${first['fit_samples']} samples, `batch_update` takes '
      '${first['batch']} samples, `predict` '
      '${(first['x']! as List).length}. hgf_deep runs on one core in f64; '
      'GPU and JAX-on-GPU rows include the readback to host memory. The GPU '
      'has no single sequential fit (`GpuDeepEnsemble` fits many at once).\n',
    )
    ..writeln(
      _picture('benchmark_deep', 'Deep HGF: time against network width'),
    )
    ..writeln('| op | backend | ${widths.map((w) => 'H=$w').join(' | ')} |')
    ..writeln('|---|---|${widths.map((_) => '--:').join('|')}|');
  for (final op in ['fit', 'batch_update', 'predict']) {
    for (final MapEntry(key: backend, value: ops) in v.entries) {
      final m = ops[op];
      if (m == null) continue;
      b.writeln(
        '| $op | $backend | '
        '${widths.map((w) => m[w] == null ? '' : _fmt(m[w]!)).join(' | ')} |',
      );
    }
  }
  return '\n$b';
}

/// What JAX ran on for the `_gpu` rows (bench_python.py's `extra`).
String _jaxGpu(List<Map<String, Object?>> results) =>
    results
        .where((r) => (r['name'] as String).contains('_gpu'))
        .map((r) => r['extra'])
        .whereType<String>()
        .firstOrNull ??
    'GPU';

/// The GPU adapter's name, from the native probe.
String? _gpuName;

/// hgf_gpu's probe on the batch scenario (`c2_unbounded` is
/// `continuous_2level` as specified in inputs.json, over the same ω grid).
Future<List<Map<String, Object?>>> _gpu(int repeat, {required bool web}) async {
  final pkgGpu = '$root/packages/hgf_gpu';
  const name = 'batch_1000x1000_continuous_2level';
  final query = {
    'sizes': '1000',
    'scenarios': 'c2_unbounded',
    'repeat': '$repeat',
    'cpuRepeat': '1',
  };
  double pick(Iterable<Map<String, Object?>> lines) {
    for (final m in lines) {
      if (m['scenario'] == 'c2_unbounded' && m['batch'] == 1000) {
        return double.parse('${m['totalMs']}');
      }
    }
    throw StateError('No GPU batch result');
  }

  Iterable<Map<String, Object?>> jsonLines(String text) sync* {
    for (final line in const LineSplitter().convert(text)) {
      if (!line.startsWith('{')) continue;
      final m = jsonDecode(line) as Map<String, Object?>;
      // wgpu: "driver / name (chip) / version".
      if (m['backend'] case final String b) {
        final parts = b.split(' / ');
        _gpuName ??= (parts.length > 1 ? parts[1] : b).replaceAll(
          RegExp(r' \(.*\)'),
          '',
        );
      }
      yield m;
    }
  }

  final exe = '${out.path}/gpu_native';
  await _run('dart', [
    'compile',
    'exe',
    'benchmark/bin/gpu_native.dart',
    '-o',
    exe,
  ], cwd: pkgGpu);
  final native = await _run(exe, [
    for (final e in query.entries) '${e.key}=${e.value}',
  ], cwd: pkgGpu);
  final results = [
    {
      'name': '$name/gpu_native_f32',
      'unit': 'ms',
      'value': pick(jsonLines(native)),
      'extra': ?_gpuName,
    },
  ];
  if (web) {
    final dir = Directory('${out.path}/gpu_web')..createSync(recursive: true);
    for (final f in [
      '$pkgGpu/benchmark/web/index.html',
      '$pkg/benchmark/data/inputs.json',
    ]) {
      File(f).copySync('${dir.path}/${f.split('/').last}');
    }
    await _run('dart', [
      'compile',
      'wasm',
      '-O4',
      'benchmark/web/gpu_bench.dart',
      '-o',
      '${dir.path}/bench.wasm',
    ], cwd: pkgGpu);
    final page = await _run('uv', [
      'run',
      '--with',
      'playwright',
      'python',
      '-I',
      'tool/gpu_bench.py',
      dir.path,
      Uri(queryParameters: query).query,
    ], cwd: pkgGpu);
    results.add({
      'name': '$name/gpu_webgpu_f32',
      'unit': 'ms',
      'value': pick(jsonLines(page)),
      'extra': ?_gpuName,
    });
  }
  return results;
}

Future<List<Map<String, Object?>>> _rustNative(File inputs, int repeat) async {
  final crate = '$root/tool/rs_native_bench';
  File('$root/../pyhgf/Cargo.lock').copySync('$crate/Cargo.lock');
  final target = '${out.path}/cargo';
  await _run('cargo', [
    'build',
    '--release',
    '--target-dir',
    target,
  ], cwd: crate);
  final data = Directory('${out.path}/rsdata')..createSync(recursive: true);
  final json = jsonDecode(inputs.readAsStringSync()) as Map<String, Object?>;
  for (final key in ['continuous', 'binary', 'usdchf', 'omegas']) {
    File('${data.path}/$key.txt').writeAsStringSync(
      (json[key] as List<Object?>).map((v) => (v as num).toDouble()).join('\n'),
    );
  }
  final text = await _run('$target/release/rs_native_bench', [
    data.path,
    '$repeat',
  ], cwd: crate);
  return [
    for (final e in jsonDecode(text.trim()) as List<Object?>)
      e as Map<String, Object?>,
  ];
}

List<Map<String, Object?>> _readJson(String path) => [
  for (final e in jsonDecode(File(path).readAsStringSync()) as List<Object?>)
    e as Map<String, Object?>,
];

List<Map<String, Object?>> _parseBench(String stdoutText) {
  for (final line in const LineSplitter().convert(stdoutText)) {
    if (line.startsWith('BENCH_JSON:')) {
      return [
        for (final e
            in jsonDecode(line.substring('BENCH_JSON:'.length))
                as List<Object?>)
          e as Map<String, Object?>,
      ];
    }
  }
  throw StateError('No BENCH_JSON line in output');
}

String _table(List<Map<String, Object?>> results) {
  final scenarios = <String>[];
  final backends = <String>[];
  final values = <String, double>{};
  for (final r in results) {
    final name = r['name'] as String;
    final slash = name.indexOf('/');
    final s = name.substring(0, slash), b = name.substring(slash + 1);
    if (!scenarios.contains(s)) scenarios.add(s);
    if (!backends.contains(b)) backends.add(b);
    values[name] = (r['value'] as num).toDouble();
  }
  final b = StringBuffer()
    ..writeln('| backend | ${scenarios.join(' | ')} |')
    ..writeln('|---|${scenarios.map((_) => '--:').join('|')}|');
  for (final backend in backends) {
    b.write('| $backend |');
    for (final s in scenarios) {
      final v = values['$s/$backend'];
      b.write(v == null ? ' |' : ' ${_fmt(v)} |');
    }
    b.writeln();
  }
  return b.toString();
}

String _fmt(double v) => v >= 100
    ? v.toStringAsFixed(0)
    : v >= 10
    ? v.toStringAsFixed(1)
    : v.toStringAsFixed(2);

Future<String> _cpu() async {
  if (Platform.isMacOS) {
    final name = (await _capture('sysctl', [
      '-n',
      'machdep.cpu.brand_string',
    ])).trim();
    if (name.isNotEmpty && name != 'n/a') return name;
  }
  try {
    final info = File('/proc/cpuinfo').readAsLinesSync();
    final line = info.firstWhere((l) => l.startsWith('model name'));
    return line.split(':').last.trim();
  } catch (_) {
    return 'unknown CPU';
  }
}

Future<String> _capture(String exe, List<String> args) async {
  try {
    return (await Process.run(exe, args)).stdout as String;
  } catch (_) {
    return 'n/a';
  }
}

Future<String> _run(
  String exe,
  List<String> args, {
  required String cwd,
  Map<String, String>? environment,
}) async {
  stderr.writeln('\$ $exe ${args.join(' ')}');
  final p = await Process.start(
    exe,
    args,
    workingDirectory: cwd,
    environment: environment,
  );
  final buf = StringBuffer();
  final done = Future.wait([
    p.stdout.transform(utf8.decoder).forEach((s) {
      buf.write(s);
      stderr.write(s);
    }),
    p.stderr.transform(utf8.decoder).forEach(stderr.write),
  ]);
  final code = await p.exitCode;
  await done;
  if (code != 0) {
    stderr.writeln('$exe exited with $code');
    exit(code);
  }
  return buf.toString();
}
