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
    data.nTrials = math.min(
      (nTrials != null && nTrials > 0) ? nTrials : _preset.defaultTrials,
      _preset.maxTrials,
    );
    _volatilityUpdates = _preset.volatilityUpdates ?? _volatilityUpdates;
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

  // Parameter and coupling edits on top of the preset, re-applied whenever
  // the network is rebuilt.
  final Map<int, Map<NodeField, double>> _paramEdits = {};
  final Map<(int, int, EdgeKind), double> _couplingEdits = {};

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

  /// The observations being filtered, one per trial (a category index for
  /// a categorical input).
  Float64List observations = Float64List(0);

  /// What the network takes: [observations], or their one-hot rows for a
  /// categorical input ([Network.nInputColumns] values per trial).
  Float64List _input = Float64List(0);

  /// `1` observed / `0` missing, per trial (null: all observed).
  Float64List? observed;

  /// What generated simulated data: one series (P(u = 1), the hidden
  /// value or the source mean), or one per category.
  List<Float64List>? hidden;

  /// The number of categories of a categorical input (else 0).
  int get nCategories =>
      _preset.input == InputKind.categorical ? network.nInputColumns : 0;

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
    _paramEdits.clear();
    _couplingEdits.clear();
    selected = null;
    // Slow presets (a lower cap) start at their own default.
    data.nTrials = p.maxTrials < presets.first.maxTrials
        ? p.defaultTrials
        : math.min(data.nTrials, p.maxTrials);
    _volatilityUpdates = p.volatilityUpdates ?? _volatilityUpdates;
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
    network = _preset.build(volatilityUpdates: _volatilityUpdates);
    network.build();
    _paramEdits.forEach((node, edits) {
      edits.forEach((f, v) => network.setInitial(node, f, v));
    });
    _couplingEdits.forEach((edge, v) {
      final (parent, child, kind) = edge;
      if (kind == EdgeKind.value) {
        network.setValueCoupling(parent, child, v);
      } else {
        network.setVolatilityCoupling(parent, child, v);
      }
    });
    layout = NetworkLayout.of(network);
    trajectories = null; // shape may have changed
  }

  // ---------------------------------------------------------------------------
  // Parameters
  // ---------------------------------------------------------------------------

  double param(int node, NodeField field) => network.getInitial(node, field);

  void setParam(int node, NodeField field, double value) {
    (_paramEdits[node] ??= {})[field] = value;
    network.setInitial(node, field, value);
    _afterEdit();
  }

  /// Reset [node]'s parameters and outgoing couplings to the preset's.
  void resetNode(int node) {
    _paramEdits.remove(node);
    _couplingEdits.removeWhere((e, _) => e.$1 == node);
    _rebuild();
    _afterEdit();
  }

  void setCoupling(GraphEdge edge, double strength) {
    _couplingEdits[(edge.parent, edge.child, edge.kind)] = strength;
    if (edge.kind == EdgeKind.value) {
      network.setValueCoupling(edge.parent, edge.child, strength);
    } else {
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
    data.nTrials = data.nTrials.clamp(1, _preset.maxTrials);
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

  bool _loadedFits() => loadedFits(_preset, observations, nCategories);

  /// Whether [values] suit [preset]'s input: anything for a continuous
  /// one, 0/1 for a binary one, category indices for a categorical one
  /// (non-finite values are missing observations).
  static bool loadedFits(
    Preset preset,
    Float64List values,
    int nCategories,
  ) => switch (preset.input) {
    InputKind.continuous => true,
    InputKind.binary => values.every((v) => !v.isFinite || v == 0 || v == 1),
    InputKind.categorical => values.every(
      (v) =>
          !v.isFinite || (v == v.roundToDouble() && v >= 0 && v < nCategories),
    ),
  };

  void _simulate() {
    final sim = simulate(_preset.data, data, nCategories: nCategories);
    hidden = sim.hidden;
    _setObservations(sim.observations);
    trajectories = null;
  }

  void _setObservations(Float64List values) {
    final missing = values.any((v) => !v.isFinite);
    final n = nCategories;
    if (!missing) {
      observations = values;
      observed = null;
    } else {
      observations = Float64List(values.length);
      final obs = Float64List(values.length);
      for (var t = 0; t < values.length; t++) {
        final v = values[t];
        observations[t] = v.isFinite ? v : (n > 0 ? double.nan : 0);
        obs[t] = v.isFinite ? 1 : 0;
      }
      observed = obs;
    }
    _input = n > 0 ? oneHot(observations, n) : observations;
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
        _input,
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
    // NaN when no node defines a surprise (ef-state and dp-state only).
    var sum = double.nan;
    final s = traj.totalSurprise();
    final end = player?.t ?? traj.nTime;
    for (var t = 0; t < end; t++) {
      if (!s[t].isNaN) sum = (sum.isNaN ? 0 : sum) + s[t];
    }
    totalSurprise = sum;
    error = _divergence(traj, end);
  }

  /// A message if a belief became non-finite or ran away: a mean beyond
  /// 1000 times the data's scale.
  String? _divergence(Trajectories traj, int end) {
    var scale = 1.0;
    for (final v in observations) {
      if (v.isFinite && v.abs() > scale) scale = v.abs();
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
          vectorStarts: {
            for (final f in VectorField.values)
              if (_hasVector(f)) f: _vectorStarts(f),
          },
        );
    trajectories = traj;
    return Player(this, traj)..restart();
  }

  bool _hasVector(VectorField f) {
    for (var i = 0; i < network.nNodes; i++) {
      if (network.vectorLength(i, f) > 0) return true;
    }
    return false;
  }

  Int32List _vectorStarts(VectorField f) {
    final starts = Int32List(network.nNodes + 1);
    for (var i = 0; i < network.nNodes; i++) {
      starts[i + 1] = starts[i] + network.vectorLength(i, f);
    }
    return starts;
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

  late final Float64List _obs = Float64List(owner.network.nInputColumns);
  final Float64List _flag = Float64List(1);

  void restart() {
    owner.network.reset();
    for (final f in traj.fields) {
      traj.raw(f).fillRange(0, traj.raw(f).length, double.nan);
    }
    for (final f in traj.vectorFields) {
      traj.rawVector(f).fillRange(0, traj.rawVector(f).length, double.nan);
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
      final w = _obs.length;
      _obs.setRange(0, w, owner._input, t * w);
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
      for (final f in traj.vectorFields) {
        final raw = traj.rawVector(f);
        final starts = traj.vectorStart(f);
        for (var i = 0; i < traj.nNodes; i++) {
          final len = starts[i + 1] - starts[i];
          if (len == 0) continue;
          raw.setAll(starts[i] * traj.nTime + t * len, net.getVector(i, f));
        }
      }
    }
  }
}
