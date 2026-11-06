import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';

/// Frame times, live in the app: build and raster times from the engine's
/// own timings (`SchedulerBinding.addTimingsCallback`), and the interval
/// between consecutive frames while something animates. On the web the
/// engine reports only some frames, with unreliable durations, so the
/// interval is the number to read there.
class FrameStats extends ChangeNotifier {
  FrameStats() {
    SchedulerBinding.instance
      ..addTimingsCallback(_onTimings)
      ..addPersistentFrameCallback(_onFrame);
  }

  bool _disposed = false;
  Duration? _lastFrame;
  final List<double> _intervals = [];

  /// Mean interval between consecutive frames (while animating), in ms.
  double intervalMs = 0;

  void _onFrame(Duration timeStamp) {
    if (_disposed) return; // persistent callbacks can't be removed
    final last = _lastFrame;
    _lastFrame = timeStamp;
    if (last == null) return;
    final ms = (timeStamp - last).inMicroseconds / 1000;
    if (ms <= 0 || ms > 250) return; // idle gap, not a frame time
    _intervals.add(ms);
    if (_intervals.length > _window) _intervals.removeAt(0);
    intervalMs = _intervals.reduce((a, b) => a + b) / _intervals.length;
  }

  static const _window = 60;
  final List<FrameTiming> _recent = [];
  final Stopwatch _sinceNotify = Stopwatch()..start();

  /// Mean build (UI thread) and raster time over the last frames, in ms.
  double buildMs = 0, rasterMs = 0;

  /// The slowest of the last frames (build + raster), in ms.
  double worstMs = 0;

  int get frames => _recent.length;

  void _onTimings(List<FrameTiming> timings) {
    _recent.addAll(timings);
    if (_recent.length > _window) {
      _recent.removeRange(0, _recent.length - _window);
    }
    var b = 0.0, r = 0.0, w = 0.0;
    for (final t in _recent) {
      final bm = t.buildDuration.inMicroseconds / 1000;
      final rm = t.rasterDuration.inMicroseconds / 1000;
      b += bm;
      r += rm;
      if (bm + rm > w) w = bm + rm;
    }
    buildMs = b / _recent.length;
    rasterMs = r / _recent.length;
    worstMs = w;
    // At most a few updates a second, so the readout doesn't cause frames.
    if (_sinceNotify.elapsedMilliseconds > 400) {
      _sinceNotify.reset();
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    SchedulerBinding.instance.removeTimingsCallback(_onTimings);
    super.dispose();
  }
}
