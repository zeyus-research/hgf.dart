import 'package:material_ui/material_ui.dart';

import 'deep/deep_playground.dart';
import 'model/fitting.dart';
import 'model/playground.dart';
import 'settings.dart';
import 'ui/shell.dart';

/// The seed of both themes: the belief colour of the plots.
const _seed = Color(0xFF2A78D6);

ThemeData buildTheme(Brightness brightness) {
  final scheme = ColorScheme.fromSeed(seedColor: _seed, brightness: brightness);
  return ThemeData(
    colorScheme: scheme,
    visualDensity: VisualDensity.compact,
    sliderTheme: const SliderThemeData(
      showValueIndicator: ShowValueIndicator.never,
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: scheme.surfaceContainerLow,
      margin: EdgeInsets.zero,
    ),
  );
}

class PlaygroundApp extends StatelessWidget {
  const PlaygroundApp({
    super.key,
    required this.settings,
    required this.playground,
    required this.fitting,
    this.deep,
    this.initialPage = ShellPage.hgf,
  });

  final Settings settings;
  final Playground playground;
  final Fitting fitting;

  /// The deep network page's model (made on first visit when `null`).
  final DeepPlayground? deep;
  final ShellPage initialPage;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: settings,
    builder: (context, _) => MaterialApp(
      title: 'HGF playground',
      debugShowCheckedModeBanner: false,
      theme: buildTheme(Brightness.light),
      darkTheme: buildTheme(Brightness.dark),
      // Follows the system unless the user picked one.
      themeMode: settings.themeMode,
      home: AppShell(
        settings: settings,
        playground: playground,
        fitting: fitting,
        deep: deep,
        initialPage: initialPage,
      ),
    ),
  );
}
