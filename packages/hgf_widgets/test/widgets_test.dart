import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/gestures.dart';
import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

Network threeLevelBinary() => Network()
  ..addNodes(kind: NodeKind.binaryState)
  ..addNodes(valueChildren: [0], tonicVolatility: -3)
  ..addNodes(volatilityChildren: [1], tonicVolatility: -4, mean: 1);

Float64List reversal(int n, int seed) {
  final rng = math.Random(seed);
  return Float64List.fromList([
    for (var t = 0; t < n; t++)
      rng.nextDouble() < ((t ~/ 40).isEven ? 0.8 : 0.2) ? 1.0 : 0.0,
  ]);
}

Widget host(Widget child, {Brightness brightness = Brightness.light}) =>
    MaterialApp(
      theme: ThemeData(brightness: brightness),
      home: Scaffold(body: child),
    );

void main() {
  testWidgets('trajectory view draws every plottable node + surprise', (
    tester,
  ) async {
    final net = threeLevelBinary();
    final traj = net.inputData(reversal(300, 1))!;
    await tester.pumpWidget(
      host(HgfTrajectoryView(network: net, trajectories: traj)),
    );
    expect(find.byType(TrajectoryPlot), findsNWidgets(4));

    // Hovering shows the readout without errors.
    final gesture = await tester.createGesture(kind: PointerDeviceKind.mouse);
    await gesture.addPointer(
      location: tester.getCenter(find.byType(TrajectoryPlot).first),
    );
    await tester.pump();
    await gesture.removePointer();
    expect(tester.takeException(), isNull);
  });

  testWidgets('dark theme and NaN-filled series render', (tester) async {
    final y = Float64List(50)..fillRange(0, 50, double.nan);
    await tester.pumpWidget(
      host(
        TrajectoryPlot(
          layers: [
            LineLayer(y),
            BandLayer.precision(y, y),
            PointLayer(y),
            const ReferenceLine(0),
          ],
        ),
        brightness: Brightness.dark,
      ),
    );
    expect(tester.takeException(), isNull);
  });

  testWidgets('graph: tapping a node selects it', (tester) async {
    final net = Network()..addCategoricalState(nCategories: 3);
    int? tapped;
    await tester.pumpWidget(
      host(
        SizedBox(
          width: 400,
          height: 300,
          child: NetworkGraph(network: net, onNodeTap: (i) => tapped = i),
        ),
      ),
    );
    final layout = NetworkLayout.of(net);
    final box = tester.getRect(find.byType(NetworkGraph));
    // The shared volatility parent is alone on the top row, centred.
    final top = layout.topDown.first;
    expect(layout.level[top], 3);
    await tester.tapAt(Offset(box.center.dx, box.top + 32));
    expect(tapped, top);
  });

  testWidgets('10k trials: layout + paint per re-run', (tester) async {
    tester.view.physicalSize = const Size(1600, 1000);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.reset);
    final net = threeLevelBinary();
    final u = reversal(10000, 2);
    var traj = net.inputData(u)!;
    var revision = 0;
    late StateSetter set;
    await tester.pumpWidget(
      host(
        StatefulBuilder(
          builder: (context, s) {
            set = s;
            return HgfTrajectoryView(
              network: net,
              trajectories: traj,
              revision: revision,
            );
          },
        ),
      ),
    );
    const frames = 60;
    final sw = Stopwatch()..start();
    for (var f = 0; f < frames; f++) {
      net.setInitial(1, NodeField.tonicVolatility, -3 + f / frames);
      traj = net.inputData(u, into: traj)!;
      set(() => revision++);
      await tester.pump();
    }
    final perFrame = sw.elapsedMicroseconds / frames / 1000;
    // ignore: avoid_print
    print(
      '10k trials, 4 panels: ${perFrame.toStringAsFixed(2)} ms/frame '
      '(filter + build + layout + paint, no raster)',
    );
    expect(perFrame, lessThan(16));
  }, tags: ['performance']);
}
