import 'package:material_ui/material_ui.dart';

import '../builder/builder_fit.dart';
import '../builder/builder_model.dart';
import '../builder/ui/builder_page.dart';
import '../deep/deep_playground.dart';
import '../deep/ui/deep_page.dart';
import '../model/fitting.dart';
import '../model/playground.dart';
import '../settings.dart';
import 'home.dart';

/// The playground's pages.
enum ShellPage { hgf, builder, deep }

/// The top-level navigation: the HGF page, the model builder and the deep
/// network page, as a rail on wide screens and a bottom bar on narrow ones.
/// Pages keep their state when hidden; a hidden page's tickers are off.
class AppShell extends StatefulWidget {
  const AppShell({
    super.key,
    required this.settings,
    required this.playground,
    required this.fitting,
    this.deep,
    this.builder,
    this.initialPage = ShellPage.hgf,
  });

  final Settings settings;
  final Playground playground;
  final Fitting fitting;

  /// The deep network page's model (made on first visit when `null`).
  final DeepPlayground? deep;

  /// The builder's model (made on first visit when `null`, from the HGF
  /// page's network).
  final BuilderModel? builder;
  final ShellPage initialPage;

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  late ShellPage _page = widget.initialPage;
  DeepPlayground? _deep;
  bool _ownsDeep = false;

  DeepPlayground get deep {
    if (_deep == null) {
      _deep = widget.deep ?? DeepPlayground();
      _ownsDeep = widget.deep == null;
    }
    return _deep!;
  }

  BuilderModel? _builder;
  bool _ownsBuilder = false;

  BuilderModel get builder {
    if (_builder == null) {
      _ownsBuilder = widget.builder == null;
      _builder = widget.builder ?? BuilderModel(coalesce: true)
        ..openNetwork(
          widget.playground.network,
          name: widget.playground.preset.name,
        );
      if (_builder!.fitting == null) {
        final fitting = widget.fitting;
        BuilderFitting(
          _builder!,
          engine: fitting.engine,
          config: () => fitting.config,
        );
      }
    }
    return _builder!;
  }

  @override
  void dispose() {
    if (_ownsDeep) _deep?.dispose();
    if (_ownsBuilder) _builder?.dispose();
    super.dispose();
  }

  void _go(int i) => setState(() => _page = ShellPage.values[i]);

  /// Open the HGF page's network (with its edits) in the builder.
  void _editInBuilder() {
    final pg = widget.playground;
    builder.openNetwork(pg.network, name: pg.preset.name);
    setState(() => _page = ShellPage.builder);
  }

  @override
  Widget build(BuildContext context) {
    final pages = [
      TickerMode(
        enabled: _page == ShellPage.hgf,
        child: HomePage(
          settings: widget.settings,
          playground: widget.playground,
          fitting: widget.fitting,
          onEditInBuilder: _editInBuilder,
        ),
      ),
      // Built on first visit only.
      if (_page == ShellPage.builder || _builder != null)
        TickerMode(
          enabled: _page == ShellPage.builder,
          child: BuilderPage(model: builder),
        )
      else
        const SizedBox.shrink(),
      if (_page == ShellPage.deep || _deep != null)
        TickerMode(
          enabled: _page == ShellPage.deep,
          child: DeepPage(deep: deep),
        )
      else
        const SizedBox.shrink(),
    ];
    final stack = IndexedStack(index: _page.index, children: pages);
    return LayoutBuilder(
      builder: (context, box) {
        if (box.maxWidth >= 900) {
          return Row(
            children: [
              NavigationRail(
                selectedIndex: _page.index,
                onDestinationSelected: _go,
                labelType: NavigationRailLabelType.all,
                destinations: const [
                  NavigationRailDestination(
                    icon: Icon(Icons.show_chart),
                    label: Text('HGF'),
                  ),
                  NavigationRailDestination(
                    icon: Icon(Icons.schema_outlined),
                    selectedIcon: Icon(Icons.schema),
                    label: Text('Builder'),
                  ),
                  NavigationRailDestination(
                    icon: Icon(Icons.hub_outlined),
                    selectedIcon: Icon(Icons.hub),
                    label: Text('Deep network'),
                  ),
                ],
              ),
              const VerticalDivider(width: 1),
              Expanded(child: stack),
            ],
          );
        }
        return Scaffold(
          body: stack,
          bottomNavigationBar: NavigationBar(
            selectedIndex: _page.index,
            onDestinationSelected: _go,
            height: 60,
            destinations: const [
              NavigationDestination(icon: Icon(Icons.show_chart), label: 'HGF'),
              NavigationDestination(
                icon: Icon(Icons.schema_outlined),
                selectedIcon: Icon(Icons.schema),
                label: 'Builder',
              ),
              NavigationDestination(
                icon: Icon(Icons.hub_outlined),
                selectedIcon: Icon(Icons.hub),
                label: 'Deep network',
              ),
            ],
          ),
        );
      },
    );
  }
}
