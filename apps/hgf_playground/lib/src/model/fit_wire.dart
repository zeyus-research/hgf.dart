import 'dart:convert';
import 'dart:typed_data';

import 'package:hgf_fit/hgf_fit.dart';

import 'fit_config.dart';
import 'fit_engine.dart';
import 'fit_jobs.dart';

/// Fit jobs, their events and outcomes as data, for the web's fit host
/// worker: Web Workers only exchange structured clones, and the host is a
/// program of its own (`worker/fit_host.dart`), so nothing with code in it
/// (closures, priors' methods) can cross as it is.
///
/// A message is a JSON string plus the `Float64List`s pulled out of it
/// ([pack]), which travel as typed arrays: series, draws, recovery tables.
/// Models are `FitCodec` JSON, as for the pool's workers.

/// What the fit host worker posts once it listens, or when it fails to
/// start (followed by the error).
const fitHostReady = 'hgf fit host ready';
const fitHostFailed = 'hgf fit host failed: ';

/// A JSON string and the arrays it refers to.
typedef Packed = ({String json, List<Float64List> arrays});

/// [value] (maps, lists, numbers, strings, bools, null and `Float64List`s)
/// as a [Packed]: arrays are replaced by references, and non-finite doubles
/// by markers (JSON has neither).
Packed pack(Object? value) {
  final arrays = <Float64List>[];
  Object? out(Object? v) => switch (v) {
    Float64List() => {r'$f64': (arrays..add(v)).length - 1},
    double() when !v.isFinite => {r'$num': '$v'},
    Map() => {for (final e in v.entries) e.key as String: out(e.value)},
    List() => [for (final x in v) out(x)],
    _ => v,
  };
  return (json: jsonEncode(out(value)), arrays: arrays);
}

/// [pack]'s input back.
Object? unpack(String json, List<Float64List> arrays) => jsonDecode(
  json,
  reviver: (_, v) => switch (v) {
    {r'$f64': final int i} => arrays[i],
    {r'$num': final String x} => double.parse(x),
    _ => v,
  },
);

typedef JsonMap = Map<String, Object?>;

double _d(Object? v) => (v! as num).toDouble();
double? _dn(Object? v) => v == null ? null : _d(v);
int _i(Object? v) => (v! as num).toInt();
int? _in(Object? v) => v == null ? null : _i(v);

const _codec = FitCodec();

// -----------------------------------------------------------------------------
// Jobs
// -----------------------------------------------------------------------------

JsonMap encodeConfig(FitConfig c) => {
  'device': c.device.name,
  'workers': c.workers,
  'restarts': c.restarts,
  'chains': c.chains,
  'warmup': c.warmup,
  'draws': c.draws,
  'targetAccept': c.targetAccept,
  'maxTreeDepth': c.maxTreeDepth,
};

FitConfig decodeConfig(JsonMap j) => FitConfig(
  device: FitDevice.values.byName(j['device']! as String),
  workers: _i(j['workers']),
  restarts: _i(j['restarts']),
  chains: _i(j['chains']),
  warmup: _i(j['warmup']),
  draws: _i(j['draws']),
  targetAccept: _d(j['targetAccept']),
  maxTreeDepth: _i(j['maxTreeDepth']),
);

JsonMap _encodeParam(FreeParam p) => {
  'id': p.id,
  'symbol': p.symbol,
  'help': p.help,
  'target': p.target == null ? null : FitCodec.encodeTarget(p.target!),
  'prior': _codec.encodePrior(p.prior),
  'priorText': p.priorText,
  'defaultFree': p.defaultFree,
  'also': [for (final t in p.alsoTargets) FitCodec.encodeTarget(t)],
};

FreeParam _decodeParam(JsonMap j) => FreeParam(
  id: j['id']! as String,
  symbol: j['symbol']! as String,
  help: j['help']! as String,
  target: j['target'] == null
      ? null
      : FitCodec.decodeTarget(j['target']! as JsonMap),
  prior: _codec.decodePrior(j['prior']! as JsonMap),
  priorText: j['priorText']! as String,
  defaultFree: j['defaultFree']! as bool,
  alsoTargets: [
    for (final t in j['also'] as List<Object?>? ?? const [])
      FitCodec.decodeTarget(t! as JsonMap),
  ],
);

JsonMap _encodeSubject(Subject s) => {
  'inputs': s.inputs,
  'responses': s.responses,
  'observed': s.observed,
  'timeSteps': s.timeSteps,
  'id': s.id is num || s.id is String ? s.id : s.id?.toString(),
};

Subject _decodeSubject(JsonMap j) => Subject(
  inputs: j['inputs']! as Float64List,
  responses: j['responses'] as Float64List?,
  observed: j['observed'] as Float64List?,
  timeSteps: j['timeSteps'] as Float64List?,
  id: j['id'],
);

/// [job] as data ([decodeJob] reads it back).
Packed encodeJob(FitJobSpec<Object?> job) => pack({
  'config': encodeConfig(job.config),
  'gpu': job.gpu,
  ...switch (job) {
    ParameterFitJob() => {
      'kind': 'fit',
      'method': job.method.name,
      'target': job.target.name,
      'params': [for (final p in job.params) _encodeParam(p)],
      'model': _codec.encodeModel(job.model),
      'subject': _encodeSubject(job.subject),
      'truth': job.truth,
    },
    RecoveryJob() => {
      'kind': 'recovery',
      'params': [for (final p in job.params) _encodeParam(p)],
      'model': _codec.encodeModel(job.model),
      'base': _encodeSubject(job.base),
      'agents': job.agents,
      'seed': job.seed,
      'reveals': job.reveals,
    },
  },
});

FitJobSpec<Object?> decodeJob(String json, List<Float64List> arrays) {
  final j = unpack(json, arrays)! as JsonMap;
  final config = decodeConfig(j['config']! as JsonMap);
  final gpu = j['gpu']! as bool;
  final params = [
    for (final p in j['params']! as List<Object?>) _decodeParam(p! as JsonMap),
  ];
  final model = _codec.decodeModel(j['model']! as JsonMap);
  return switch (j['kind']) {
    'fit' => ParameterFitJob(
      config: config,
      gpu: gpu,
      method: FitMethod.values.byName(j['method']! as String),
      target: FitTarget.values.byName(j['target']! as String),
      params: params,
      model: model,
      subject: _decodeSubject(j['subject']! as JsonMap),
      truth: (j['truth'] as JsonMap?)?.map((k, v) => MapEntry(k, _d(v))),
    ),
    'recovery' => RecoveryJob(
      config: config,
      gpu: gpu,
      params: params,
      model: model,
      base: _decodeSubject(j['base']! as JsonMap),
      agents: _i(j['agents']),
      seed: _i(j['seed']),
      reveals: switch (j['reveals']) {
        final List<Object?> rs => [
          for (final r in rs) [for (final i in r! as List<Object?>) _i(i)],
        ],
        _ => null,
      },
    ),
    final kind => throw FormatException('Bad fit job kind: $kind'),
  };
}

// -----------------------------------------------------------------------------
// Events and results
// -----------------------------------------------------------------------------

String encodeEvent(JobEvent e) => jsonEncode(switch (e) {
  JobStage(:final text) => {'stage': text},
  JobProgress(:final completed, :final total) => {
    'completed': completed,
    'total': total,
  },
});

JobEvent decodeEvent(String json) => switch (jsonDecode(json)) {
  {'stage': final String text} => JobStage(text),
  {'completed': final int completed, 'total': final int total} => JobProgress(
    completed,
    total,
  ),
  final other => throw FormatException('Bad job event: $other'),
};

String encodeGpuInfo(GpuInfo g) =>
    jsonEncode({'adapter': g.adapter, 'reason': g.reason});

GpuInfo decodeGpuInfo(String json) {
  final j = jsonDecode(json) as JsonMap;
  return (adapter: j['adapter'] as String?, reason: j['reason'] as String?);
}

/// A job's result ([decodeResult] reads it back, given the job).
Packed encodeResult(Object? result) => pack(switch (result) {
  FitOutcome r => {
    'kind': 'fit',
    'method': r.method.name,
    'target': r.target.name,
    'millis': r.millis,
    'where': r.where,
    'logEvidence': r.logEvidence,
    'converged': r.converged,
    'divergences': r.divergences,
    'chains': r.chains,
    'note': r.note,
    'deviceNote': r.deviceNote,
    // Parameters by position: the job has them.
    'estimates': [
      for (final e in r.estimates)
        {
          'estimate': e.estimate,
          'lower': e.lower,
          'upper': e.upper,
          'truth': e.truth,
          if (e.laplace case final l?)
            'laplace': {
              'transform': _codec.encodeTransform(l.transform),
              'mode': l.mode,
              'sd': l.sd,
            },
          'draws': e.draws,
          'rhat': e.rhat,
          'ess': e.ess,
        },
    ],
  },
  RecoveryOutcome r => {
    'kind': 'recovery',
    'truth': r.truth,
    'estimate': r.estimate,
    'ok': r.ok,
    'dropped': r.dropped,
    'redrawn': r.redrawn,
    'unconverged': r.unconverged,
    'millis': r.millis,
    'where': r.where,
    'deviceNote': r.deviceNote,
  },
  _ => throw ArgumentError('Not a fit job result: ${result.runtimeType}'),
});

/// The result of [job] from [encodeResult]'s output.
R decodeResult<R>(FitJobSpec<R> job, String json, List<Float64List> arrays) {
  final j = unpack(json, arrays)! as JsonMap;
  final Object result = switch (job) {
    ParameterFitJob(:final params) => FitOutcome(
      method: FitMethod.values.byName(j['method']! as String),
      target: FitTarget.values.byName(j['target']! as String),
      millis: _i(j['millis']),
      where: j['where']! as String,
      logEvidence: _dn(j['logEvidence']),
      converged: j['converged']! as bool,
      divergences: _in(j['divergences']),
      chains: _in(j['chains']),
      note: j['note'] as String?,
      deviceNote: j['deviceNote'] as String?,
      estimates: [
        for (final (i, e as JsonMap) in (j['estimates']! as List).indexed)
          ParamEstimate(
            param: params[i],
            estimate: _d(e['estimate']),
            lower: _d(e['lower']),
            upper: _d(e['upper']),
            truth: _dn(e['truth']),
            laplace: switch (e['laplace']) {
              final JsonMap l => LaplaceMarginal(
                _codec.decodeTransform(l['transform']! as JsonMap),
                _d(l['mode']),
                _d(l['sd']),
              ),
              _ => null,
            },
            draws: e['draws'] as Float64List?,
            rhat: _dn(e['rhat']),
            ess: _dn(e['ess']),
          ),
      ],
    ),
    RecoveryJob(:final params) => RecoveryOutcome(
      params: params,
      truth: [for (final t in j['truth']! as List) t as Float64List],
      estimate: [for (final t in j['estimate']! as List) t as Float64List],
      ok: [for (final o in j['ok']! as List) o as bool],
      dropped: _i(j['dropped']),
      redrawn: _i(j['redrawn']),
      unconverged: _i(j['unconverged']),
      millis: _i(j['millis']),
      where: j['where']! as String,
      deviceNote: j['deviceNote'] as String?,
    ),
  };
  return result as R;
}
