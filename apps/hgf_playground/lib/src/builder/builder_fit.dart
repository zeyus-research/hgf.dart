import 'dart:convert';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:hgf_fit/document.dart';
import 'package:hgf_fit/hgf_fit.dart'
    show FitCancelled, FitCodec, ParticipantEstimates, Subject;

import '../model/fit_config.dart';
import '../model/fit_engine.dart';
import '../model/fit_jobs.dart';
import 'builder_model.dart';

/// Fitting from the model builder: every participant of the builder's data
/// (loaded, or simulated participants) with the document's free
/// parameters, group level ([ModelDocument.group]) and fit settings
/// ([DocFit]), as a [DocumentFitJob] on the playground's [FitEngine] (the
/// same pool, GPU and host as the HGF page's fits).
///
/// Changes are announced through the builder ([BuilderModel.fitChanged]),
/// so the builder's listeners see them.
class BuilderFitting {
  BuilderFitting(this.builder, {required this.engine, required this.config}) {
    builder.fitting = this;
  }

  final BuilderModel builder;
  final FitEngine engine;

  /// The playground's fit settings: the device and the worker pool (the
  /// document says the rest).
  final FitConfig Function() config;

  bool running = false;
  String? stage;

  /// NUTS progress (completed / total iterations), while sampling.
  double? progress;
  String? error;

  DocumentFitOutcome? result;

  /// The model and data [result] is for (see [stale]).
  String? _resultKey;

  /// Preview each participant at its estimates (else at the parameters'
  /// starting values, or a simulated participant's own values).
  bool previewAtEstimates = true;

  /// Why the document can't be fitted now, or null.
  String? get unavailable {
    final c = builder.compiled;
    if (c == null) return 'Fix the errors first.';
    final m = c.model;
    if (m == null) {
      return 'Add a response (or fit to the observations) to fit the model.';
    }
    if (m.dimension == 0) {
      return 'Free some parameters: tick them in the network\'s and the '
          'response\'s inspectors, or add them here.';
    }
    final method = builder.document.fit.method;
    if (method == 'none') return 'The fit method is "none".';
    if (c.hierarchical != null && builder.participants.length < 2) {
      return 'The group level needs several participants: load data with a '
          'participant column, or simulate more participants.';
    }
    return null;
  }

  /// Whether the model or the data changed since [result].
  bool get stale => result != null && _key() != _resultKey;

  String? _key() {
    final c = builder.compiled;
    final m = c?.model;
    if (c == null || m == null) return null;
    return jsonEncode([
      const FitCodec().encodeModel(m),
      builder.document.group.toJson(),
      builder.dataKey,
    ]);
  }

  /// [participant]'s estimates for the preview, when [previewAtEstimates]
  /// and [result] fits the current model's parameters.
  Float64List? estimateFor(String participant, List<String> names) {
    final r = result;
    if (r == null || !previewAtEstimates || !listEquals(r.names, names)) {
      return null;
    }
    final k = r.participants.indexOf(participant);
    if (k < 0) return null;
    final x = r.estimate[k];
    return x.every((v) => v.isFinite) ? x : null;
  }

  void setPreviewAtEstimates(bool on) {
    if (on == previewAtEstimates) return;
    previewAtEstimates = on;
    builder.refreshPreview();
  }

  /// Fits every participant (see [DocumentFitJob]).
  Future<void> fit() async {
    if (running || unavailable != null) return;
    final c = builder.compiled!;
    final m = c.model!;
    final List<Subject> subjects;
    try {
      subjects = builder.allSubjects();
    } on Object catch (e) {
      error = '$e';
      builder.fitChanged();
      return;
    }
    final d = builder.document.fit;
    final base = config();
    final cfg = base.copyWith(
      restarts: d.restarts,
      chains: d.chains,
      warmup: d.warmup,
      draws: d.draws,
      targetAccept: d.targetAccept,
      maxTreeDepth: d.maxTreeDepth,
    );
    final nuts = d.method == 'nuts';
    final h = c.hierarchical;
    final n = subjects.length;
    final gpu = switch (base.device) {
      FitDevice.gpu => true,
      FitDevice.cpu => false,
      FitDevice.auto =>
        nuts
            ? d.chains * (h == null ? n : 1) >= FitConfig.autoGpuChains
            : n * (d.restarts + 1) >= FitConfig.autoGpuMapRuns,
    };
    final job = DocumentFitJob(
      config: cfg,
      gpu: gpu,
      method: nuts ? FitMethod.nuts : FitMethod.map,
      model: m,
      subjects: subjects,
      pooled: h?.pooled ?? const {},
      shared: h?.shared ?? const {},
      centered: h?.centered ?? false,
      seed: d.seed ?? (nuts ? 0 : 1),
    );
    final key = _key();
    running = true;
    stage = 'Starting…';
    progress = null;
    error = null;
    builder.fitChanged();
    try {
      final r = await engine.run(job, (e) {
        switch (e) {
          case JobStage(:final text):
            stage = text;
            progress = null;
          case JobProgress(:final completed, :final total):
            progress = total == 0 ? null : completed / total;
        }
        builder.fitChanged();
      });
      result = r;
      _resultKey = key;
    } on FitCancelled {
      // keep the last result
    } on Object catch (e) {
      error = '$e';
    } finally {
      running = false;
      stage = null;
      progress = null;
    }
    builder.refreshPreview();
  }

  void cancel() {
    if (running) engine.cancel();
  }

  void clear() {
    result = null;
    _resultKey = null;
    builder.refreshPreview();
  }

  /// [result] as CSV: a row per participant with each parameter's estimate
  /// and 95 % interval (`name`, `name_lower`, `name_upper`; simulated
  /// participants add their true values, `name_true`). With a group level,
  /// a second table follows a blank line: a row per group estimate
  /// (`name_typical`, `name_spread`, or a shared parameter's name).
  String? estimatesCsv() {
    final r = result;
    if (r == null) return null;
    return ParticipantEstimates(
      names: r.names,
      participants: r.participants,
      estimate: r.estimate,
      lower: r.lower,
      upper: r.upper,
      group: r.group,
    ).toCsv(truth: builder.simulatedTruth);
  }

  /// The estimates per participant for the outputs (null: no usable
  /// result).
  Map<String, List<double>>? estimatesByParticipant(List<String> names) {
    final r = result;
    if (r == null || !listEquals(r.names, names)) return null;
    return {
      for (final (k, id) in r.participants.indexed)
        if (r.estimate[k].every((v) => v.isFinite)) id: r.estimate[k],
    };
  }
}

/// A number for the fit's tables: 3 significant digits, or an exponent.
String fitNumber(double v) {
  if (!v.isFinite) return '–';
  final a = v.abs();
  if (a != 0 && (a >= 1e4 || a < 1e-3)) return v.toStringAsExponential(2);
  final digits = a >= 100 ? 1 : (a >= 10 ? 2 : 3);
  return v.toStringAsFixed(math.max(0, digits));
}
