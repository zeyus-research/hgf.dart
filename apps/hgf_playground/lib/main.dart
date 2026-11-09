import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:hgf_fit/hgf_fit.dart';

import 'src/app.dart';
import 'src/bench.dart';
import 'src/model/fit_config.dart';
import 'src/model/fitting.dart';
import 'src/model/playground.dart';
import 'src/settings.dart';

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
  final fitting = Fitting(
    playground,
    config: fitConfig,
    spawnWorkers: query['workers'] == '0'
        ? (_) async => await FitWorkers.spawn(count: 0)
            ..yieldInterval = const Duration(milliseconds: 12)
        : null,
  );
  runApp(
    PlaygroundApp(settings: settings, playground: playground, fitting: fitting),
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
