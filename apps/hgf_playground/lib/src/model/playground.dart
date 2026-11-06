import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/scheduler.dart';
import 'package:hgf/hgf.dart';
import 'package:hgf_widgets/hgf_widgets.dart';

import 'data.dart';
import 'presets.dart';

/// Where the observations come from.
enum DataSource { simulated, loaded }

/// The playground's state: a network built from a [Preset], its data, and
/// the trajectories, re-filtered on every change.
///
/// Parameter edits go straight to the compiled network (`setInitial`,
/// `setValueCoupling`), then the whole series is filtered again into the
/// same [Trajectories] buffers (`inputData(into:)`), so a slider drag
/// allocates nothing per frame. With [coalesce], edits made between two
/// frames (a slider sends several per frame) cost one re-filter, at the
/// start of the next frame.
class Playground extends ChangeNotifier {
  Playground({String? presetId, int? nTrials, this.coalesce = false}) {
    _preset = presetById(presetId ?? presets.first.id);
    data.nTrials = (nTrials != null && nTrials > 0)
        ? nTrials
        : _preset.defaultTrials;
    _specs = _preset.nodes();
    _rebuild();
    _simulate();
    _rerun();
  }

  /// Re-filter once per frame instead of on every edit (needs a running
  /// Flutter binding).
  final bool coalesce;
  bool _rerunScheduled = false;

  /// The fields recorded: what the plots, captions and surprise read.
  static const recordedFields = [
    NodeField.mean,
    NodeField.expectedMean,
    NodeField.expectedPrecision,
  ];

  late Preset _preset;
  Preset get preset => _preset;
  late List<NodeSpec> _specs;

  VolatilityUpdates _volatilityUpdates = VolatilityUpdates.unbounded;
  VolatilityUpdates get volatilityUpdates => _volatilityUpdates;

  late Network network;
  late NetworkLayout layout;
  Trajectories? trajectories;

  /// Bumped whenever [trajectories] changed in place.
  int revision = 0;

  int? selected;

  final DataSettings data = DataSettings();
  DataSource source = DataSource.simulated;
  String? loadedName;

  /// The observations being filtered.
  Float64List observations = Float64List(0);

  /// `1` observed / `0` missing, per trial (null: all observed).
  Float64List? observed;

  /// What generated simulated data (P(u = 1) or the hidden value).
  Float64List? hidden;

  /// Wall time of the last full re-filter, in microseconds.
  int filterMicros = 0;

  /// Sum of the per-trial total surprise (NaN-skipping).
  double totalSurprise = double.nan;

  /// Error from the last run (e.g. NaN beliefs), shown to the user.
  String? error;

  // ---------------------------------------------------------------------------
  // Structure
  // ---------------------------------------------------------------------------

  void setPreset(String id) {
    final p = presetById(id);
    if (p.id == _preset.id) return;
    player = null;
    _preset = p;
    _specs = p.nodes();
    selected = null;
    _rebuild();
    if (source == DataSource.loaded && !_loadedFits()) {
      source = DataSource.simulated;
      loadedName = null;
    }
    if (source == DataSource.simulated) _simulate();
    _rerun();
  }

  void setVolatilityUpdates(VolatilityUpdates v) {
    if (v == _volatilityUpdates) return;
    player = null;
    _volatilityUpdates = v;
    _rebuild();
    _rerun();
  }

  void select(int? node) {
    selected = node == selected ? null : node;
    notifyListeners();
  }

  void _rebuild() {
    network = _preset.build(
      volatilityUpdates: _volatilityUpdates,
      specs: _specs,
    );
    network.build();
    layout = NetworkLayout.of(network);
    trajectories = null; // shape may have changed
  }

  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------

  double param(int node, NodeField field) => network.getInitial(node, field);

  void setParam(int node, NodeField field, double value) {
    _specs[node].params[field] = value;
    network.setInitial(node, field, value);
    _afterEdit();
  }

  /// Reset [node]'s parameters to the preset's.
  void resetNode(int node) {
    final fresh = _preset.nodes()[node];
    _specs[node] = fresh;
    _rebuild();
    _afterEdit();
  }

  void setCoupling(GraphEdge edge, double strength) {
    final spec = _specs[edge.parent];
    if (edge.kind == EdgeKind.value) {
      spec.valueStrengths[spec.valueChildren.indexOf(edge.child)] = strength;
      network.setValueCoupling(edge.parent, edge.child, strength);
    } else {
      spec.volatilityStrengths[spec.volatilityChildren.indexOf(edge.child)] =
          strength;
      network.setVolatilityCoupling(edge.parent, edge.child, strength);
    }
    // The layout caches strengths for the edge labels.
    layout = NetworkLayout.of(network);
    _afterEdit();
  }

  void _afterEdit() {
    player = null; // edits show the whole series again
    _requestRerun();
  }

  void _requestRerun() {
    if (!coalesce) {
      _rerun();
      return;
    }
    if (_rerunScheduled) return;
    _rerunScheduled = true;
    SchedulerBinding.instance
      ..scheduleFrameCallback((_) {
        _rerunScheduled = false;
        _rerun();
      })
      ..scheduleFrame();
  }

  // ---------------------------------------------------------------------------
  // Data
  // ---------------------------------------------------------------------------

  void updateData(void Function(DataSettings d) edit) {
    edit(data);
    source = DataSource.simulated;
    loadedName = null;
    player = null;
    _simulate();
    _requestRerun();
  }

  void reseed() => updateData((d) => d.seed = math.Random().nextInt(1 << 30));

  /// Use [values] as the observations (NaN marks a missing one).
  void load(Float64List values, String name) {
    player = null;
    source = DataSource.loaded;
    loadedName = name;
    hidden = null;
    _setObservations(values);
    trajectories = null;
    _rerun();
  }

  bool _loadedFits() =>
      _preset.input == InputKind.continuous ||
      observations.every((v) => v.isNaN || v == 0 || v == 1);

  void _simulate() {
    final sim = simulate(_preset.input, data);
    hidden = sim.hidden;
    _setObservations(sim.observations);
    trajectories = null;
  }

  void _setObservations(Float64List values) {
    final missing = values.any((v) => !v.isFinite);
    if (!missing) {
      observations = values;
      observed = null;
      return;
    }
    observations = Float64List(values.length);
    final obs = Float64List(values.length);
    for (var t = 0; t < values.length; t++) {
      final v = values[t];
      observations[t] = v.isFinite ? v : 0;
      obs[t] = v.isFinite ? 1 : 0;
    }
    observed = obs;
  }

  // ---------------------------------------------------------------------------
  // Filtering
  // ---------------------------------------------------------------------------

  void _rerun() {
    if (observations.isEmpty) {
      trajectories = null;
      notifyListeners();
      return;
    }
    final sw = Stopwatch()..start();
    try {
      final into = trajectories;
      trajectories = network.inputData(
        observations,
        observed: observed,
        // A reused buffer keeps the fields it was made with.
        fields: into == null ? recordedFields : defaultRecordedFields,
        into: into,
      );
      error = null;
    } catch (e) {
      trajectories = null;
      error = '$e';
    }
    filterMicros = sw.elapsedMicroseconds;
    revision++;
    _updateSurprise();
    notifyListeners();
  }

  void _updateSurprise() {
    final traj = trajectories;
    if (traj == null) {
      totalSurprise = double.nan;
      return;
    }
    var sum = 0.0;
    final s = traj.totalSurprise();
    final end = player?.t ?? traj.nTime;
    for (var t = 0; t < end; t++) {
      if (!s[t].isNaN) sum += s[t];
    }
    totalSurprise = sum;
    error = _divergence(traj, end);
  }

  /// A message if a belief became non-finite or ran away: a mean beyond
  /// 1000 times the data's scale.
  String? _divergence(Trajectories traj, int end) {
    var scale = 1.0;
    for (final v in observations) {
      if (v.abs() > scale) scale = v.abs();
    }
    final limit = 1000 * scale;
    for (var i = 0; i < traj.nNodes; i++) {
      final m = traj.series(i, NodeField.mean);
      for (var t = 0; t < end; t++) {
        final v = m[t];
        if (v.abs() <= limit) continue;
        final what = v.isFinite
            ? 'ran away (μ = ${v.toStringAsExponential(1)})'
            : 'became ${v.isNaN ? 'NaN' : 'infinite'}';
        return '${defaultNodeLabel(i)} $what at trial $t. Try the standard '
            'volatility update, or a lower ω.';
      }
    }
    return null;
  }

  // ---------------------------------------------------------------------------
  // Trial-by-trial playback (predict / step, as in a real-time experiment)
  // ---------------------------------------------------------------------------

  /// The running playback, or null when showing the whole series.
  Player? player;

  bool get playing => player?.playing ?? false;

  /// Trials per second during playback.
  double speed = 20;

  void setSpeed(double trialsPerSecond) {
    speed = trialsPerSecond.clamp(1, 1000);
    notifyListeners();
  }

  /// Start (or resume) trial-by-trial playback from the first trial.
  void play() {
    final p = player ??= _newPlayer();
    if (p.t >= observations.length) {
      p.restart();
    }
    p.playing = true;
    notifyListeners();
  }

  void pause() {
    player?.playing = false;
    notifyListeners();
  }

  /// One trial forward (pauses playback).
  void stepOnce() {
    final p = player ??= _newPlayer();
    p.playing = false;
    if (p.t < observations.length) p.advance(1);
    revision++;
    _updateSurprise();
    notifyListeners();
  }

  /// Back to the whole-series view.
  void stop() {
    if (player == null) return;
    player = null; // inputData starts from the initial beliefs again
    _rerun();
  }

  /// Advance playback by [seconds] of wall time; called every frame.
  void tick(double seconds) {
    final p = player;
    if (p == null || !p.playing) return;
    p.carry += seconds * speed;
    final n = p.carry.floor();
    if (n == 0) return;
    p.carry -= n;
    p.advance(n);
    if (p.t >= observations.length) p.playing = false;
    revision++;
    _updateSurprise();
    notifyListeners();
  }

  Player _newPlayer() {
    final traj =
        trajectories ??
        Trajectories(
          observations.length,
          network.nNodes,
          recordedFields,
          kinds: [for (var i = 0; i < network.nNodes; i++) network.kindOf(i)],
        );
    trajectories = traj;
    return Player(this, traj)..restart();
  }
}

/// Filters one trial at a time with `predict` + `step`, writing what
/// `inputData` would have recorded into the shown trajectories.
class Player {
  Player(this.owner, this.traj);

  final Playground owner;
  final Trajectories traj;

  /// Trials done.
  int t = 0;
  bool playing = false;
  double carry = 0;

  final Float64List _obs = Float64List(1);
  final Float64List _flag = Float64List(1);

  void restart() {
    owner.network.reset();
    for (final f in traj.fields) {
      traj.raw(f).fillRange(0, traj.raw(f).length, double.nan);
    }
    t = 0;
    carry = 0;
  }

  void advance(int n) {
    final net = owner.network;
    final end = math.min(t + n, owner.observations.length);
    final observed = owner.observed;
    for (; t < end; t++) {
      net.predict();
      _obs[0] = owner.observations[t];
      if (observed != null) {
        _flag[0] = observed[t];
        net.step(_obs, observed: _flag);
      } else {
        net.step(_obs);
      }
      for (final f in traj.fields) {
        final col = traj.raw(f);
        for (var i = 0; i < traj.nNodes; i++) {
          col[i * traj.nTime + t] = net.get(i, f);
        }
      }
    }
  }
}
