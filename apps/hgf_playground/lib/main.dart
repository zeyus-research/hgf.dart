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
  final playground = Playground(
    presetId: bench != null
        ? 'binary3'
        : (settings.preset.isEmpty ? null : settings.preset),
    nTrials: bench ?? settings.nTrials,
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
