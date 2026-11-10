import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

import '../deep_playground.dart';
import '../deep_view.dart';
import 'decision_map.dart';
import 'deep_cards.dart';
import 'deep_graph.dart';

/// Build a network, train it, and follow one sample through it.
class DeepPage extends StatefulWidget {
  const DeepPage({super.key, required this.deep});

  final DeepPlayground deep;

  @override
  State<DeepPage> createState() => _DeepPageState();
}

class _DeepPageState extends State<DeepPage>
    with SingleTickerProviderStateMixin {
  late final Ticker _ticker = createTicker((_) => deep.tick());
  NodeField _field = NodeField.mean;
  NodeRef? _selected;

  DeepPlayground get deep => widget.deep;

  @override
  void initState() {
    super.initState();
    deep.addListener(_sync);
  }

  @override
  void dispose() {
    deep.removeListener(_sync);
    _ticker.dispose();
    super.dispose();
  }

  void _sync() {
    if (deep.playing && !_ticker.isActive) {
      _ticker.start();
    } else if (!deep.playing && _ticker.isActive) {
      _ticker.stop();
    }
    // A rebuilt network may no longer have the selected node.
    final s = _selected;
    if (s != null &&
        (s.layer >= deep.network.nLayers ||
            s.node >= deep.network.layerSizes[s.layer])) {
      _selected = null;
    }
  }

  DeepViewModel _view() {
    final t = deep.trace;
    if (t == null) {
      final net = deep.network;
      return DeepViewModel(
        network: net,
        states: [for (var l = 0; l < net.nLayers; l++) net.state(l)],
        weights: net.weights,
      );
    }
    final step = t.steps[deep.cursor];
    return DeepViewModel(
      network: deep.network,
      states: step.states,
      weights: t.weightsAt(deep.cursor),
      step: step,
    );
  }

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: deep,
    builder: (context, _) => Scaffold(
      appBar: AppBar(
        title: Text(
          MediaQuery.sizeOf(context).width < 600
              ? 'Deep network'
              : 'Deep HGF network',
        ),
      ),
      body: LayoutBuilder(
        builder: (context, box) =>
            box.maxWidth >= 900 ? _wide(context) : _narrow(context),
      ),
    ),
  );

  List<Widget> _cards() => [
    TraceCard(
      deep: deep,
      field: _field,
      onField: (f) => setState(() => _field = f),
      selected: _selected,
      onSelect: (n) => setState(() => _selected = n),
    ),
    const SizedBox(height: 12),
    TrainingCard(deep: deep),
    const SizedBox(height: 12),
    ArchitectureCard(deep: deep),
    const SizedBox(height: 12),
    const DeepAboutCard(),
  ];

  Widget _graph() {
    final view = _view();
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Expanded(
          child: DeepGraphView(
            model: view,
            field: _field,
            selected: _selected,
            revision: deep.revision * 1000 + deep.cursor,
            onSelect: (n) => setState(() => _selected = n),
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
          child: Text(
            [
              'Nodes: ${_field.symbol} (${_field.description})',
              if (view.mode != ViewMode.full) view.modeReason,
            ].join(' · '),
            style: theme.textTheme.labelSmall,
          ),
        ),
      ],
    );
  }

  Widget _plots() => Row(
    children: [
      Expanded(child: ScoresPlot(deep: deep)),
      const SizedBox(width: 8),
      Expanded(child: ScoresPlot(deep: deep, loss: true)),
    ],
  );

  Widget _wide(BuildContext context) => Row(
    crossAxisAlignment: CrossAxisAlignment.stretch,
    children: [
      SizedBox(
        width: 380,
        child: ListView(padding: const EdgeInsets.all(12), children: _cards()),
      ),
      const VerticalDivider(width: 1),
      Expanded(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(8, 8, 12, 8),
          child: Column(
            children: [
              Expanded(flex: 3, child: _graph()),
              const SizedBox(height: 8),
              Expanded(
                flex: 2,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    AspectRatio(
                      aspectRatio: 0.95,
                      child: DecisionMap(deep: deep),
                    ),
                    const SizedBox(width: 12),
                    Expanded(child: _plots()),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    ],
  );

  Widget _narrow(BuildContext context) => ListView(
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
    children: [
      SizedBox(height: 380, child: _graph()),
      const SizedBox(height: 8),
      SizedBox(height: 300, child: DecisionMap(deep: deep)),
      const SizedBox(height: 8),
      SizedBox(height: 150, child: ScoresPlot(deep: deep)),
      const SizedBox(height: 8),
      SizedBox(height: 150, child: ScoresPlot(deep: deep, loss: true)),
      const SizedBox(height: 12),
      ..._cards(),
    ],
  );
}
