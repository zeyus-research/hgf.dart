import 'dart:typed_data';

import 'empirical_bayes.dart';
import 'hierarchical.dart';
import 'laplace.dart';
import 'model.dart';
import 'sample/diagnostics.dart';
import 'sample/sample.dart';

/// What a [GroupEstimate] row is.
enum GroupRow {
  /// A pooled parameter's population mean, on the natural scale (the
  /// typical participant's value).
  typical,

  /// A pooled parameter's population sd, on its unconstrained scale (log
  /// scale for a positive parameter: 0.5 means participants differ by
  /// factors of about e^0.5).
  spread,

  /// A parameter shared by all participants.
  shared,
}

/// A group-level estimate: a pooled parameter's typical value (its
/// population mean, mapped to the natural scale) or spread (the population
/// sd, on the unconstrained scale), or a shared parameter's value. Bounds
/// are NaN where the method gives none.
typedef GroupEstimate = ({
  String name,
  GroupRow kind,
  double value,
  double lower,
  double upper,
});

/// A fit of several participants as one table: per participant a value of
/// every parameter (the mode, or the posterior mean; shared ones repeated)
/// with a 95 % interval, and the group-level estimates. What the
/// playground's builder shows and the exported scripts print and write
/// ([toCsv]), whatever the method.
final class ParticipantEstimates {
  ParticipantEstimates({
    required this.names,
    required this.participants,
    required this.estimate,
    required this.lower,
    required this.upper,
    this.group = const [],
  });

  /// MAP per participant with Laplace intervals ([laplace] of their
  /// `MapFit`).
  factory ParticipantEstimates.fromLaplace(
    LaplaceFit laplace, {
    List<String>? participants,
  }) {
    final rs = laplace.results;
    final model = rs.isEmpty ? null : rs.first.model;
    final lo = <Float64List>[], hi = <Float64List>[];
    for (final r in rs) {
      final (a, b) = r.interval();
      lo.add(a);
      hi.add(b);
    }
    return ParticipantEstimates(
      names: model?.names ?? const [],
      participants:
          participants ??
          [for (final (s, r) in rs.indexed) _id(r.map.subject.id, s)],
      estimate: [for (final r in rs) Float64List.fromList(r.map.x)],
      lower: lo,
      upper: hi,
    );
  }

  /// Empirical Bayes ([fitEmpiricalBayes]): each participant's mode under
  /// the populations, with `y ± 1.96 sd` intervals on the unconstrained
  /// scale; the populations as typical values and spreads (no intervals:
  /// they are point estimates).
  factory ParticipantEstimates.fromEmpiricalBayes(
    EmpiricalBayesFit fit, {
    List<String>? participants,
  }) {
    final model = fit.model.model;
    final k = model.dimension;
    final lo = <Float64List>[], hi = <Float64List>[];
    for (final e in fit.subjects) {
      final se = e.unconstrainedStandardErrors;
      final a = Float64List(k), b = Float64List(k);
      for (var i = 0; i < k; i++) {
        final t = model.parameters[i].transform;
        a[i] = t.toConstrained(e.y[i] - _z95 * se[i]);
        b[i] = t.toConstrained(e.y[i] + _z95 * se[i]);
      }
      lo.add(a);
      hi.add(b);
    }
    return ParticipantEstimates(
      names: model.names,
      participants:
          participants ??
          [for (final (s, e) in fit.subjects.indexed) _id(e.subject.id, s)],
      estimate: [for (final e in fit.subjects) Float64List.fromList(e.x)],
      lower: lo,
      upper: hi,
      group: [
        for (final MapEntry(key: name, value: p)
            in fit.populations.entries) ...[
          (
            name: name,
            kind: GroupRow.typical,
            value: model.parameters[model.names.indexOf(name)].transform
                .toConstrained(p.mean),
            lower: double.nan,
            upper: double.nan,
          ),
          (
            name: name,
            kind: GroupRow.spread,
            value: p.sd,
            lower: double.nan,
            upper: double.nan,
          ),
        ],
      ],
    );
  }

  /// NUTS per participant ([sampleNuts]): posterior means and central 95 %
  /// intervals.
  factory ParticipantEstimates.fromNuts(
    NutsFit fit, {
    List<String>? participants,
  }) {
    final rs = fit.results;
    final est = <Float64List>[], lo = <Float64List>[], hi = <Float64List>[];
    for (final d in rs) {
      final k = d.dimension;
      final m = Float64List(k), a = Float64List(k), b = Float64List(k);
      for (var i = 0; i < k; i++) {
        final (mean, lo, hi) = summariseDraws(d.parameter(i));
        m[i] = mean;
        a[i] = lo;
        b[i] = hi;
      }
      est.add(m);
      lo.add(a);
      hi.add(b);
    }
    return ParticipantEstimates(
      names: rs.isEmpty ? const [] : rs.first.names,
      participants:
          participants ??
          [for (final (s, d) in rs.indexed) _id(d.subject?.id, s)],
      estimate: est,
      lower: lo,
      upper: hi,
    );
  }

  /// The joint posterior of a [HierarchicalModel] for [subjects]
  /// ([sampleHierarchical]): per participant and parameter the posterior
  /// mean and central 95 % interval; per pooled parameter its population's
  /// typical value and spread, per shared one its value.
  factory ParticipantEstimates.fromHierarchical(
    HierarchicalModel model,
    HierarchicalFit fit,
    List<Subject> subjects, {
    List<String>? participants,
  }) {
    final names = model.model.names;
    final draws = fit.draws;
    final n = subjects.length, k = names.length;
    final est = <Float64List>[], lo = <Float64List>[], hi = <Float64List>[];
    for (var s = 0; s < n; s++) {
      final m = Float64List(k), a = Float64List(k), b = Float64List(k);
      for (var i = 0; i < k; i++) {
        final name = names[i];
        final idx = model.shared.contains(name)
            ? model.indexOf(name)
            : model.indexOf(name, subject: s);
        final (mean, lo, hi) = summariseDraws(draws.parameter(idx));
        m[i] = mean;
        a[i] = lo;
        b[i] = hi;
      }
      est.add(m);
      lo.add(a);
      hi.add(b);
    }
    final group = <GroupEstimate>[];
    for (final (i, name) in names.indexed) {
      if (model.pooled.containsKey(name)) {
        final t = model.model.parameters[i].transform;
        final (m, a, b) = summariseDraws(
          draws.parameter(model.indexOf(name, group: 'mu')),
        );
        group.add((
          name: name,
          kind: GroupRow.typical,
          value: t.toConstrained(m),
          lower: t.toConstrained(a),
          upper: t.toConstrained(b),
        ));
        final (sm, sa, sb) = summariseDraws(
          draws.parameter(model.indexOf(name, group: 'sigma')),
        );
        group.add((
          name: name,
          kind: GroupRow.spread,
          value: sm,
          lower: sa,
          upper: sb,
        ));
      } else if (model.shared.contains(name)) {
        final (m, a, b) = summariseDraws(draws.parameter(model.indexOf(name)));
        group.add((
          name: name,
          kind: GroupRow.shared,
          value: m,
          lower: a,
          upper: b,
        ));
      }
    }
    return ParticipantEstimates(
      names: names,
      participants:
          participants ?? [for (var s = 0; s < n; s++) _id(subjects[s].id, s)],
      estimate: est,
      lower: lo,
      upper: hi,
      group: group,
    );
  }

  /// The model's parameters, in order.
  final List<String> names;

  /// The participants' ids, in the subjects' order (`Subject.id`, else 1,
  /// 2, …).
  final List<String> participants;

  /// Per participant, per parameter (natural scale): the estimate and the
  /// 95 % interval (NaN: none).
  final List<Float64List> estimate, lower, upper;

  final List<GroupEstimate> group;

  /// Participant [id]'s estimates by parameter name (null if unknown).
  Map<String, double>? of(String id) {
    final s = participants.indexOf(id);
    if (s < 0) return null;
    return {for (final (i, n) in names.indexed) n: estimate[s][i]};
  }

  /// The estimates by participant id (each in [names] order), e.g. for
  /// `CompiledDocument.withOutputs`.
  Map<String, List<double>> get byParticipant => {
    for (final (s, id) in participants.indexed) id: estimate[s],
  };

  /// The table as CSV: a row per participant (`participant`, then per
  /// parameter `<name>`, `<name>_lower`, `<name>_upper`, and with [truth]
  /// (true values by participant id, e.g. of simulated participants)
  /// `<name>_true`); with group-level estimates a blank line and a second
  /// table (`parameter,value,lower,upper`: `<name>_typical`,
  /// `<name>_spread`, or a shared parameter's name). Missing values are
  /// empty cells.
  String toCsv({Map<String, List<double>>? truth}) {
    final hasTruth = truth != null && truth.isNotEmpty;
    final header = [
      'participant',
      for (final n in names) ...[
        n,
        '${n}_lower',
        '${n}_upper',
        if (hasTruth) '${n}_true',
      ],
    ];
    final rows = <List<String>>[
      for (final (k, id) in participants.indexed)
        [
          id,
          for (var i = 0; i < names.length; i++) ...[
            _cell(estimate[k][i]),
            _cell(lower[k][i]),
            _cell(upper[k][i]),
            if (hasTruth) _cell(truth[id]?[i] ?? double.nan),
          ],
        ],
    ];
    final table = _csvText(header, rows);
    if (group.isEmpty) return table;
    return '$table\n\n${_csvText(const ['parameter', 'value', 'lower', 'upper'], [
      for (final e in group) [switch (e.kind) {
          GroupRow.typical => '${e.name}_typical',
          GroupRow.spread => '${e.name}_spread',
          GroupRow.shared => e.name,
        }, _cell(e.value), _cell(e.lower), _cell(e.upper)],
    ])}';
  }

  /// A summary for the console: per parameter the participants' estimates
  /// with intervals, then the group level.
  @override
  String toString() {
    final b = StringBuffer();
    final w = participants.fold(11, (m, p) => p.length > m ? p.length : m);
    for (final (i, n) in names.indexed) {
      b.writeln(n);
      for (final (s, id) in participants.indexed) {
        b.writeln(
          '  ${id.padRight(w)}  ${_num(estimate[s][i])}'
          '${lower[s][i].isFinite ? '  [${_num(lower[s][i])}, ${_num(upper[s][i])}]' : ''}',
        );
      }
    }
    if (group.isNotEmpty) {
      b.writeln('group level');
      for (final e in group) {
        final label = switch (e.kind) {
          GroupRow.typical => '${e.name} typical',
          GroupRow.spread => '${e.name} spread',
          GroupRow.shared => '${e.name} shared',
        };
        b.writeln(
          '  ${label.padRight(w)}  ${_num(e.value)}'
          '${e.lower.isFinite ? '  [${_num(e.lower)}, ${_num(e.upper)}]' : ''}',
        );
      }
    }
    return b.toString();
  }

  static String _num(double v) => v.isFinite ? v.toStringAsPrecision(5) : '—';
}

const _z95 = 1.959963984540054;

String _id(Object? id, int s) => id == null ? '${s + 1}' : '$id';

String _cell(double v) => v.isFinite ? '$v' : '';

/// Rows as CSV, quoting cells that need it.
String _csvText(List<String> header, List<List<String>> rows) {
  String q(String c) =>
      c.contains(RegExp('[,"\n]')) ? '"${c.replaceAll('"', '""')}"' : c;
  return [
    header.map(q).join(','),
    for (final r in rows) r.map(q).join(','),
  ].join('\n');
}

/// The mean and central 95 % interval of [chains]' draws (NaN if none).
(double, double, double) summariseDraws(List<Float64List> chains) {
  final all = Float64List.fromList(chains.expand((c) => c).toList()..sort());
  if (all.isEmpty) return (double.nan, double.nan, double.nan);
  var sum = 0.0;
  for (final v in all) {
    sum += v;
  }
  return (sum / all.length, quantileOf(all, 0.025), quantileOf(all, 0.975));
}
