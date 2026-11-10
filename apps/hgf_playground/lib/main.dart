import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:hgf_fit/hgf_fit.dart';

import 'src/app.dart';
import 'src/bench.dart';
import 'src/model/fit_config.dart';
import 'src/model/fit_host.dart';
import 'src/model/fitting.dart';
import 'src/model/playground.dart';
import 'src/settings.dart';
import 'src/ui/shell.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = Settings();
  final bench = benchTrials();
  final fitBench = fitBenchRequested();
  if (bench == null && !fitBench) await settings.load();
  // Links to a preset: `?preset=dp&trials=200` (over the saved settings).
  final query = kIsWeb ? Uri.base.queryParameters : const <String, String>{};
  final preset = query['preset'] ?? settings.preset;
  final playground = Playground(
    presetId: bench != null ? 'binary3' : (preset.isEmpty ? null : preset),
    nTrials: bench ?? int.tryParse(query['trials'] ?? '') ?? settings.nTrials,
    coalesce: true,
  );
  // Bench overrides: `&workers=0` fits on the main thread (web),
  // `&device=gpu|cpu|auto` and `&chains=N` (or the dart-defines
  // HGF_FIT_DEVICE and HGF_FIT_CHAINS on desktop).
  var fitConfig = settings.fit;
  if (fitBench) {
    final device =
        query['device'] ?? const String.fromEnvironment('HGF_FIT_DEVICE');
    final chains =
        query['chains'] ?? const String.fromEnvironment('HGF_FIT_CHAINS');
    fitConfig = FitConfig(
      device: FitDevice.values.asNameMap()[device] ?? FitDevice.cpu,
      workers: 4,
      chains: int.tryParse(chains) ?? 4,
    );
  }
  // Fits run in a host (pool, GPU and drivers): an isolate on desktop, a
  // Web Worker on the web, so the UI only gets progress and results.
  // `HGF_FIT_HOST=false` (a dart-define) or `&fithost=0` (web) keeps them
  // on the UI thread, for comparison.
  final mainThread = query['workers'] == '0';
  final fitting = Fitting(
    playground,
    config: fitConfig,
    engine:
        mainThread ||
            query['fithost'] == '0' ||
            !const bool.fromEnvironment('HGF_FIT_HOST', defaultValue: true)
        ? null
        : spawnFitHost(),
    spawnWorkers: mainThread
        ? (_) async => await FitWorkers.spawn(count: 0)
            ..yieldInterval = const Duration(milliseconds: 12)
        : null,
  );
  // `?page=deep` (or the dart-define HGF_PAGE=deep) opens the deep network
  // page.
  final page = query['page'] ?? const String.fromEnvironment('HGF_PAGE');
  runApp(
    PlaygroundApp(
      settings: settings,
      playground: playground,
      fitting: fitting,
      initialPage: page == 'deep' ? ShellPage.deep : ShellPage.hgf,
    ),
  );
  if (fitBench) {
    WidgetsBinding.instance.addPostFrameCallback(
      (_) => runFitBench(playground, fitting),
    );
  }
  if (bench != null) {
    // Scripted slider drags; see bench.dart.
    WidgetsBinding.instance.addPostFrameCallback(
      (_) => runBench(playground, label: 'slider sweep'),
    );
  }
}
