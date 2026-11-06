import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'src/app.dart';
import 'src/bench.dart';
import 'src/model/playground.dart';
import 'src/settings.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = Settings();
  final bench = benchTrials();
  if (bench == null) await settings.load();
  // Links to a preset: `?preset=dp&trials=200` (over the saved settings).
  final query = kIsWeb ? Uri.base.queryParameters : const <String, String>{};
  final preset = query['preset'] ?? settings.preset;
  final playground = Playground(
    presetId: bench != null ? 'binary3' : (preset.isEmpty ? null : preset),
    nTrials: bench ?? int.tryParse(query['trials'] ?? '') ?? settings.nTrials,
    coalesce: true,
  );
  runApp(PlaygroundApp(settings: settings, playground: playground));
  if (bench != null) {
    // Scripted slider drags; see bench.dart.
    WidgetsBinding.instance.addPostFrameCallback(
      (_) => runBench(playground, label: 'slider sweep'),
    );
  }
}
