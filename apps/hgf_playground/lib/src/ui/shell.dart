import 'package:flutter/material.dart';

import '../deep/deep_playground.dart';
import '../deep/ui/deep_page.dart';
import '../model/fitting.dart';
import '../model/playground.dart';
import '../settings.dart';
import 'home.dart';

/// The playground's pages.
enum ShellPage { hgf, deep }

/// The top-level navigation: the HGF page and the deep network page, as a
/// rail on wide screens and a bottom bar on narrow ones. Pages keep their
/// state when hidden; a hidden page's tickers are off.
class AppShell extends StatefulWidget {
  const AppShell({
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

  @override
  void dispose() {
    if (_ownsDeep) _deep?.dispose();
    super.dispose();
  }

  void _go(int i) => setState(() => _page = ShellPage.values[i]);

  @override
  Widget build(BuildContext context) {
    final pages = [
      TickerMode(
        enabled: _page == ShellPage.hgf,
        child: HomePage(
          settings: widget.settings,
          playground: widget.playground,
          fitting: widget.fitting,
        ),
      ),
      // Built on first visit only.
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
