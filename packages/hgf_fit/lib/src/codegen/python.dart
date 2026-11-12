import 'package:hgf/hgf.dart';

import '../codec.dart';
import '../parameter.dart';
import 'export_spec.dart';
import 'plan.dart';

/// Generates a Python script that uses pyhgf (and PyMC to fit), so the
/// work can continue in the lab's usual stack.
///
/// The network is built with pyhgf's `Network().add_nodes(...)` (the same
/// builder calls as the Dart export), then whatever the calls don't set is
/// written to `net.attributes`. The fit is a PyMC model with the same
/// priors, the network's log-likelihood wrapped from JAX with
/// `pytensor.wrap_jax` (pyhgf's recommended route), and `pm.find_MAP` or
/// `pm.sample`.
///
/// hgf.dart extensions pyhgf doesn't have are flagged in comments: the
/// eHGF MGF correction, the dp-state surprise, a non-default maximum
/// posterior precision. Custom couplings, priors and response models
/// become stubs that raise `NotImplementedError`.
final class PythonExport {
  PythonExport(this.spec);

  final ExportSpec spec;

  /// pyhgf ignores `binary_parameters` and an ef-state `xis` argument, so
  /// those become attribute writes.
  late final NetworkPlan plan = NetworkPlan.fromJson(
    spec.network,
    categoricalArguments: false,
    efXis: false,
  );

  final List<String> _stubs = [];
  final Set<String> _stubNames = {};
  final List<String> _warnings = [];

  /// `build_network()`.
  /// Whether an input with noise parents has a constant-state value parent
  /// ([NetworkFixes.constantNoiseVariance] changes its updates).
  bool _biasIntoNoiseChild() {
    final net = plan.build();
    for (final i in net.inputIdxs) {
      if (net.kindOf(i) != NodeKind.continuousState ||
          net.volatilityParentsOf(i).isEmpty) {
        continue;
      }
      if (net
          .valueParentsOf(i)
          .any((p) => net.kindOf(p) == NodeKind.constantState)) {
        return true;
      }
    }
    return false;
  }

  String buildNetwork() {
    final b = StringBuffer()
      ..writeln('def build_network():')
      ..writeln(
        '    """The perceptual model: ${plan.nNodes} '
        '${plan.nNodes == 1 ? 'node' : 'nodes'}, input '
        '${plan.inputs.length == 1 ? 'node ${plan.inputs.single}' : 'nodes ${plan.inputs.join(', ')}'}."""',
      );
    final settings = [
      if (plan.volatilityUpdates != VolatilityUpdates.unbounded)
        'volatility_updates="${plan.volatilityUpdates.pyhgfName}"',
      if (plan.meanFieldUpdates) 'mean_field_updates=True',
    ];
    if (plan.ehgfMgfCorrection) {
      _warnings.add(
        'The network uses hgf.dart\'s eHGF MGF correction '
        '(ehgfMgfCorrection), which pyhgf leaves out: its eHGF updates '
        'differ.',
      );
    }
    if (plan.fixes.constantNoiseVariance && _biasIntoNoiseChild()) {
      _warnings.add(
        'An input with a noise parent has a bias (constant-state) value '
        'parent: pyhgf adds (t·α)² of variance for it to the noise '
        'prediction error, which hgf.dart corrects '
        '(NetworkFixes.constantNoiseVariance): its noise updates differ.',
      );
    }
    if (plan.maxPosteriorPrecision != 1e10) {
      _warnings.add(
        'hgf.dart caps posterior precisions at '
        '${pyFloat(plan.maxPosteriorPrecision)} here (default 1e10); pyhgf '
        'has no such setting.',
      );
    }
    final adds = plan.calls.where(_isAdd).toList();
    final rest = plan.calls.where((c) => !_isAdd(c)).toList();
    b.writeln('    net = (');
    b.writeln('        Network(${settings.join(', ')})');
    for (final c in adds) {
      final (code, comment) = _add(c);
      b.writeln('        .$code${comment == null ? '' : '  # $comment'}');
    }
    b.writeln('    )');
    if (plan.precisionClippingValue != 1e-6) {
      b.writeln(
        '    net.attributes[-1]["precision_clipping_value"] = '
        '${pyFloat(plan.precisionClippingValue)}',
      );
    }
    if (rest.isNotEmpty) {
      b.writeln('    # Values set after building (e.g. with the sliders).');
    }
    for (final c in rest) {
      switch (c) {
        case SetInitial(:final node, :final field, :final value):
          if (plan.inputs.contains(node) &&
              plan.kinds[node] == NodeKind.continuousState &&
              (field == NodeField.tonicVolatility ||
                  field == NodeField.autoconnectionStrength) &&
              value != 0) {
            _warnings.add(_inputResetWarning(node, field));
          }
          b.writeln(
            '    set_attribute(net, $node, "${field.pyhgfName}", '
            '${pyFloat(value)})',
          );
        case SetInitialVector(:final node, :final field, :final values):
          b.writeln(
            '    set_attribute(net, $node, "${field.pyhgfName}", '
            'jnp.array([${values.map(pyFloat).join(', ')}]))',
          );
        case SetCoupling(
          :final parent,
          :final child,
          :final strength,
          :final volatility,
        ):
          b.writeln(
            '    set_coupling(net, $parent, $child, ${pyFloat(strength)}, '
            '"${volatility ? 'volatility' : 'value'}")',
          );
        default:
          throw StateError('Unexpected call $c');
      }
    }
    b.writeln('    return net');
    return b.toString();
  }

  /// pyhgf sets every input node's tonic volatility and autoconnection
  /// strength to 0 when it filters; hgf.dart only when it builds.
  static String _inputResetWarning(int node, NodeField field) =>
      'pyhgf resets the ${field.pyhgfName} of input node $node to 0 in '
      'input_data; hgf.dart keeps the value set here, so the results differ.';

  static bool _isAdd(BuildCall c) =>
      c is AddNodes ||
      c is AddEfState ||
      c is AddCategoricalState ||
      c is AddDpState;

  (String, String?) _add(BuildCall c) {
    String edge(List<PlanEdge> e) {
      final idx = e.length == 1
          ? '${e.single.node}'
          : '[${e.map((x) => x.node).join(', ')}]';
      if (e.every((x) => x.strength == 1.0)) return idx;
      return '([${e.map((x) => x.node).join(', ')}], '
          '[${e.map((x) => pyFloat(x.strength)).join(', ')}])';
    }

    switch (c) {
      case AddNodes():
        final args = [
          if (c.kind != NodeKind.continuousState) 'kind="${c.kind.pyhgfName}"',
          if (c.nNodes != 1) 'n_nodes=${c.nNodes}',
          if (c.valueParents.isNotEmpty)
            'value_parents=${edge(c.valueParents)}',
          if (c.valueChildren.isNotEmpty)
            'value_children=${edge(c.valueChildren)}',
          if (c.volatilityParents.isNotEmpty)
            'volatility_parents=${edge(c.volatilityParents)}',
          if (c.volatilityChildren.isNotEmpty)
            'volatility_children=${edge(c.volatilityChildren)}',
          if (c.customCoupling case final name?)
            'coupling_fn=(${_couplingStub(name)},)'
          else if (c.coupling != CouplingFn.linear)
            'coupling_fn=(${_coupling(c.coupling)},)',
          for (final MapEntry(:key, :value) in c.parameters.entries)
            '${key.pyhgfName}=${pyFloat(value)}',
        ];
        return (
          'add_nodes(${args.join(', ')})',
          c.nNodes == 1 ? '${c.first}' : '${c.first}-${c.first + c.nNodes - 1}',
        );
      case AddEfState():
        final args = [
          'kind="ef-state"',
          if (c.nNodes != 1) 'n_nodes=${c.nNodes}',
          if (c.dimension != 1) ...[
            'dimension=${c.dimension}',
            'distribution="multivariate-normal"',
          ],
          if (c.learning != EfLearning.fixed)
            'learning="${c.learning.pyhgfName}"',
          if (c.nus != 3.0) 'nus=${pyFloat(c.nus)}',
          if (c.valueChildren case final vc?)
            'value_children=${vc.length == 1 ? '${vc.single}' : '[${vc.join(', ')}]'}',
        ];
        final last = c.first + c.nNodes - 1;
        return (
          'add_nodes(${args.join(', ')})',
          '${c.nNodes == 1 ? '${c.first}' : '${c.first}-$last'}'
              '${c.impliedNodes == 0 ? '' : ' (${last + 1}-${last + c.impliedNodes}: learning chains)'}',
        );
      case AddCategoricalState():
        return (
          'add_nodes(kind="categorical-state", n_categories=${c.nCategories})',
          '${c.node} (${c.node + 1}-${c.node + c.impliedNodes}: its binary '
              'HGFs)',
        );
      case AddDpState():
        final args = [
          'kind="dp-state"',
          'batch_size=${c.batchSize}',
          if (c.alpha != 1.0) 'alpha=${pyFloat(c.alpha)}',
          if (c.sensoryPrecision != 1.0)
            'sensory_precision=${pyFloat(c.sensoryPrecision)}',
        ];
        return ('add_nodes(${args.join(', ')})', '${c.node}');
      default:
        throw StateError('Not a builder call: $c');
    }
  }

  static String _coupling(CouplingFn f) => switch (f) {
    CouplingFn.linear => 'None',
    CouplingFn.relu => 'jax.nn.relu',
    CouplingFn.sigmoid => 'jax.nn.sigmoid',
    CouplingFn.tanh => 'jnp.tanh',
    CouplingFn.leakyRelu => 'jax.nn.leaky_relu',
    // hgf.dart's gelu is the exact one (erf), not JAX's default tanh form.
    CouplingFn.gelu => 'partial(jax.nn.gelu, approximate=False)',
  };

  String _couplingStub(String name) {
    final id = '${pyIdentifier(name)}_coupling';
    if (_stubNames.add(id)) {
      _stubs.add(
        'def $id(x):\n'
        '    """The custom coupling "$name" (TODO: hgf.dart can\'t export '
        'code;\n'
        '    write it in JAX, which differentiates it)."""\n'
        '    raise NotImplementedError("$name")\n',
      );
    }
    return id;
  }

  /// The fit: `log_likelihood(params, data)` and `fit(data)`.
  String _fitCode(Map<String, Object?> model, ExportFit fit) {
    final params = [
      for (final p in model['parameters'] as List<Object?>)
        p as Map<String, Object?>,
    ];
    final response = model['response'] as Map<String, Object?>;
    final responseParams = _responseParameters(response);
    final b = StringBuffer()
      ..writeln('def negative_log_likelihood(params, data):')
      ..writeln(
        '    """The surprise of the data (minus the log-likelihood) under '
        '[params],',
      )
      ..writeln('    a dict of the free parameters by name."""')
      ..writeln('    net = build_network()');
    final responseValues = <String, String>{
      for (final MapEntry(:key, :value) in responseParams.entries)
        key: pyFloat(value),
    };
    for (final p in params) {
      final name = p['name'] as String;
      final v = 'params["$name"]';
      for (final t in p['targets'] as List<Object?>) {
        switch (FitCodec.decodeTarget(t as Map<String, Object?>)) {
          case NodeTarget(:final node, :final field):
            b.writeln(
              '    set_attribute(net, $node, "${field.pyhgfName}", $v)',
            );
          case ValueCouplingTarget(:final parent, :final child):
            b.writeln('    set_coupling(net, $parent, $child, $v, "value")');
          case VolatilityCouplingTarget(:final parent, :final child):
            b.writeln(
              '    set_coupling(net, $parent, $child, $v, "volatility")',
            );
          case ResponseTarget(name: final r):
            responseValues[r] = v;
        }
      }
    }
    b
      ..writeln('    net.input_data(')
      ..writeln('        input_data=data["inputs"],')
      ..writeln('        observed=data["observed"],')
      ..writeln('        time_steps=data["time_steps"],')
      ..writeln('    )')
      ..writeln('    return ${_responseSurprise(response, responseValues)}')
      ..writeln()
      ..writeln()
      ..writeln('def fit(data):')
      ..writeln(
        '    """${fit is ExportNuts ? 'NUTS' : 'The MAP estimate'} with '
        'PyMC: the priors, and the network\'s',
      )
      ..writeln(
        '    log-likelihood (JAX, wrapped for PyTensor) as a potential."""',
      )
      ..writeln('    import pymc as pm')
      ..writeln('    from pytensor import wrap_jax')
      ..writeln()
      ..writeln(
        '    names = [${params.map((p) => '"${p['name']}"').join(', ')}]',
      )
      ..writeln()
      ..writeln('    @wrap_jax')
      ..writeln('    def log_likelihood(*values):')
      ..writeln(
        '        return -negative_log_likelihood(dict(zip(names, values)), '
        'data)',
      )
      ..writeln()
      ..writeln('    with pm.Model():')
      ..writeln('        values = [');
    for (final p in params) {
      b.writeln(
        '            ${_prior(p['name'] as String, p['prior'] as Map<String, Object?>, p['initial'])},',
      );
    }
    b
      ..writeln('        ]')
      ..writeln('        pm.Potential("hgf", log_likelihood(*values))');
    switch (fit) {
      case ExportMap():
        b
          ..writeln('        estimate = pm.find_MAP()')
          ..writeln('    for name in names:')
          ..writeln('        print(f"{name} = {float(estimate[name]):.6g}")');
      case ExportNuts():
        b
          ..writeln('        idata = pm.sample(')
          ..writeln('            draws=${fit.draws},')
          ..writeln('            tune=${fit.warmup},')
          ..writeln('            chains=${fit.chains},')
          ..writeln('            target_accept=${pyFloat(fit.targetAccept)},')
          ..writeln('            nuts={"max_treedepth": ${fit.maxTreeDepth}},')
          ..writeln('            random_seed=${fit.seed},')
          ..writeln('        )')
          ..writeln('    import arviz as az')
          ..writeln()
          ..writeln('    print(az.summary(idata, var_names=names))');
    }
    b.writeln('    return ${fit is ExportMap ? 'estimate' : 'idata'}');
    return b.toString();
  }

  static Map<String, double> _responseParameters(Map<String, Object?> j) =>
      switch (j['type']) {
        'GaussianReadout' => {'precision': readJsonDouble(j['precision'])},
        'BinarySoftmaxReadout' ||
        'CategoricalSoftmaxReadout' ||
        'SoftmaxReadout' => {
          'inverseTemperature': readJsonDouble(j['inverseTemperature']),
        },
        _ => const {},
      };

  /// The total surprise under the response model, as an expression.
  String _responseSurprise(Map<String, Object?> j, Map<String, String> values) {
    String field() => '"${NodeField.parse(j['field'] as String).pyhgfName}"';
    switch (j['type']) {
      case 'GaussianReadout':
        return 'gaussian_readout_surprise(net, data["responses"], '
            '${j['node']}, ${field()}, ${values['precision']})';
      case 'BinarySoftmaxReadout':
        return 'binary_softmax_surprise(net, data["responses"], '
            '${j['node']}, ${values['inverseTemperature']})';
      case 'CategoricalSoftmaxReadout':
        return 'categorical_softmax_surprise(net, data["responses"], '
            '${j['node']}, ${values['inverseTemperature']})';
      case 'SoftmaxReadout':
        final options = [
          for (final o in j['options'] as List<Object?>)
            '{${[for (final t in (o as List<Object?>).cast<Map<String, Object?>>()) '${t['node']}: ${pyFloat(readJsonDouble(t['weight']))}'].join(', ')}}',
        ];
        return 'softmax_surprise(net, data["responses"], '
            '[${options.join(', ')}], ${field()}, '
            '${values['inverseTemperature']})';
      case 'ObservationSurprise':
        return switch (j['function']) {
          'FirstLevelGaussianSurprise' =>
            'jnp.sum(first_level_gaussian_surprise(net))',
          'TotalGaussianSurprise' => 'jnp.sum(total_gaussian_surprise(net))',
          'FirstLevelBinarySurprise' =>
            'jnp.sum(first_level_binary_surprise(net))',
          'CategoricalSurprise' => 'categorical_surprise(net)',
          'DpSurprise' => () {
            _warnings.add(
              'The model fits a dp-state node by its surprise, an hgf.dart '
              'extension (DpSurprise): pyhgf defines none.',
            );
            return '${_stub('dp_surprise', 'the dp-state surprise (an hgf.dart extension, not in pyhgf)')}(net)';
          }(),
          final f =>
            '${_stub(pyIdentifier('$f'), 'the response function "$f"')}(net)',
        };
      case final type:
        return '${_stub(pyIdentifier('$type'), 'the response model "$type"')}(net, data["responses"])';
    }
  }

  String _stub(String id, String what) {
    if (_stubNames.add(id)) {
      _stubs.add(
        'def $id(*args):\n'
        '    """TODO: $what; hgf.dart can\'t export it."""\n'
        '    raise NotImplementedError("$id")\n',
      );
    }
    return id;
  }

  String _prior(String name, Map<String, Object?> j, Object? initial) {
    String d(String key) => pyFloat(readJsonDouble(j[key]));
    bool finite(String key) => readJsonDouble(j[key]).isFinite;
    final init = initial == null
        ? ''
        : ', initval=${pyFloat(readJsonDouble(initial))}';
    final n = '"$name"';
    return switch (j['type']) {
      'Normal' => 'pm.Normal($n, mu=${d('mean')}, sigma=${d('sd')}$init)',
      'TruncatedNormal' =>
        'pm.TruncatedNormal($n, mu=${d('mean')}, sigma=${d('sd')}'
            '${finite('lower') ? ', lower=${d('lower')}' : ''}'
            '${finite('upper') ? ', upper=${d('upper')}' : ''}$init)',
      'Uniform' =>
        'pm.Uniform($n, lower=${d('lower')}, upper=${d('upper')}$init)',
      'HalfNormal' => 'pm.HalfNormal($n, sigma=${d('sd')}$init)',
      'LogNormal' =>
        'pm.LogNormal($n, mu=${d('mu')}, sigma=${d('sigma')}$init)',
      'Gamma' => 'pm.Gamma($n, alpha=${d('shape')}, beta=${d('rate')}$init)',
      'Flat' when finite('lower') && finite('upper') =>
        'pm.Uniform($n, lower=${d('lower')}, upper=${d('upper')}$init)',
      'Flat' when finite('lower') => 'pm.HalfFlat($n$init) + ${d('lower')}',
      'Flat' when !finite('upper') => 'pm.Flat($n$init)',
      final type =>
        '${_stub('${pyIdentifier(name)}_prior', 'the prior "$type" of $name (a PyMC distribution)')}()',
    };
  }

  /// The whole script.
  String script() {
    final model = spec.model;
    final network = buildNetwork();
    final fit = spec.fit;
    final fitCode = model == null || fit == null ? null : _fitCode(model, fit);
    final data = spec.data;
    final width = plan.nInputColumns;
    final nTime = data?.nTime(width) ?? 0;
    final responseType = (model?['response'] as Map<String, Object?>?)?['type'];
    final r =
        data?.responsesPerTrial(nTime) ??
        (responseType == null || responseType == 'ObservationSurprise' ? 0 : 1);
    final cols = ExportData.columns(
      plan,
      responsesPerTrial: r,
      observed: data?.observed != null,
      timeSteps: data?.timeSteps != null,
    );
    final b = StringBuffer()
      ..writeln(
        '"""${spec.title ?? (model == null ? 'An HGF network' : 'An HGF model')}, '
        'exported from hgf.dart.',
      )
      ..writeln()
      ..writeln(
        'Needs pyhgf${fitCode == null ? '' : ' and PyMC'} '
        '(pip install pyhgf${fitCode == null ? '' : ' pymc'}):',
      )
      ..writeln()
      ..writeln(
        spec.dataFile != null
            ? '    python <this file> [data.csv]   (default: ${spec.dataFile})'
            : data != null
            ? '    python <this file> [data.csv]   (default: the data below)'
            : '    python <this file> data.csv',
      )
      ..writeln()
      ..writeln(
        'The CSV has a header row and the columns ${cols.header.join(', ')}.',
      )
      ..writeln(
        'The observed flags are optional: without them, an input with an empty',
      )
      ..writeln(
        "cell wasn't observed on that trial. An empty response is a missed one.",
      );
    final notes = [..._warnings];
    if (plan.customCouplings.isNotEmpty || _stubs.isNotEmpty) {
      notes.add('Before running: fill in the stubs marked TODO.');
    }
    if (notes.isNotEmpty) {
      b.writeln();
      for (final w in notes) {
        b.writeln('Note: $w');
      }
    }
    b
      ..writeln('"""')
      ..writeln()
      ..writeln('import csv')
      ..writeln('import os')
      ..writeln('import sys')
      ..writeln('from functools import partial  # noqa: F401')
      ..writeln()
      ..writeln('# float64, as hgf.dart (set before JAX is imported).')
      ..writeln('os.environ.setdefault("JAX_ENABLE_X64", "1")')
      ..writeln()
      ..writeln('import jax  # noqa: E402,F401')
      ..writeln('import jax.numpy as jnp  # noqa: E402')
      ..writeln('import numpy as np  # noqa: E402')
      ..writeln('from pyhgf.model import Network  # noqa: E402');
    if (fitCode != null) {
      b.writeln(
        'from pyhgf.response import (  # noqa: E402,F401\n'
        '    first_level_binary_surprise,\n'
        '    first_level_gaussian_surprise,\n'
        '    total_gaussian_surprise,\n'
        ')',
      );
    }
    b
      ..writeln()
      ..writeln()
      ..write(network);
    for (final s in _stubs) {
      b
        ..writeln()
        ..writeln()
        ..write(s);
    }
    b
      ..writeln()
      ..writeln()
      ..write(_helpers);
    if (fitCode != null) {
      b
        ..writeln()
        ..writeln()
        ..write(_readouts)
        ..writeln()
        ..writeln()
        ..write(fitCode);
    }
    b
      ..writeln()
      ..writeln()
      ..writeln(
        '# The CSV columns: each input node\'s (one per data column), its',
      )
      ..writeln(
        '# observed flag (optional: without them, an empty cell is a missing',
      )
      ..writeln('# observation), the responses.')
      ..writeln(
        'INPUT_COLUMNS = [${cols.inputs.map((c) => '[${c.map((n) => '"$n"').join(', ')}]').join(', ')}]',
      )
      ..writeln(
        'OBSERVED_COLUMNS = [${cols.observed.map((n) => '"$n"').join(', ')}]',
      )
      ..writeln(
        'RESPONSE_COLUMNS = [${cols.responses.map((n) => '"$n"').join(', ')}]',
      )
      ..writeln(
        'TIME_STEP_COLUMN = ${cols.timeStep == null ? 'None' : '"${cols.timeStep}"'}',
      )
      ..writeln()
      ..writeln()
      ..write(_readCsv);
    if (data != null) {
      String arr(List<double>? v, int cols) => v == null
          ? 'None'
          : 'np.array(\n        [\n${_rows(v, '            ')}        ],\n'
                '        dtype=np.float64,\n'
                '    )${cols > 0 ? '.reshape(-1, $cols)' : ''}';
      b
        ..writeln()
        ..writeln()
        ..writeln('nan, inf = float("nan"), float("inf")')
        ..writeln()
        ..writeln()
        ..writeln('def embedded_data():')
        ..writeln('    """The data the network was exported with."""')
        ..writeln('    return dict(')
        ..writeln('        inputs=${arr(data.inputs, width)},')
        ..writeln('        observed=${arr(data.observed, plan.inputs.length)},')
        ..writeln('        time_steps=${arr(data.timeSteps, 0)},')
        ..writeln('        responses=${arr(data.responses, r > 1 ? r : 0)},')
        ..writeln('    )');
    }
    final load = spec.dataFile != null
        ? 'read_csv(sys.argv[1] if len(sys.argv) > 1 else "${spec.dataFile}")'
        : data != null
        ? 'read_csv(sys.argv[1]) if len(sys.argv) > 1 else embedded_data()'
        : 'read_csv(sys.argv[1])';
    b
      ..writeln()
      ..writeln()
      ..writeln('def main():')
      ..writeln('    data = $load')
      ..writeln('    net = build_network()')
      ..writeln('    net.input_data(')
      ..writeln('        input_data=data["inputs"],')
      ..writeln('        observed=data["observed"],')
      ..writeln('        time_steps=data["time_steps"],')
      ..writeln('    )')
      ..writeln(
        '    print(f"{len(data[\'inputs\'])} trials; final beliefs (mean, '
        'precision):")',
      )
      ..writeln('    for node, traj in net.node_trajectories.items():')
      ..writeln('        if node < 0 or "precision" not in traj:')
      ..writeln('            continue')
      ..writeln(
        '        print(f"  node {node}: {float(traj[\'mean\'][-1])}, '
        '{float(traj[\'precision\'][-1])}")',
      );
    if (fitCode != null) b.writeln('    fit(data)');
    b
      ..writeln()
      ..writeln()
      ..writeln('if __name__ == "__main__":')
      ..writeln('    main()');
    return b.toString();
  }

  static String _rows(List<double> v, String indent) {
    final b = StringBuffer();
    for (var i = 0; i < v.length; i += 8) {
      b
        ..write(indent)
        ..write(
          v
              .sublist(i, i + 8 > v.length ? v.length : i + 8)
              .map(_pyValue)
              .join(', '),
        )
        ..writeln(',');
    }
    return b.toString();
  }

  static String _pyValue(double v) => v.isNaN
      ? 'nan'
      : v.isInfinite
      ? (v > 0 ? 'inf' : '-inf')
      : pyFloat(v);

  static const _helpers = r'''
def set_attribute(net, node, name, value):
    """Set a node's initial value (in its "temp" store where pyhgf keeps it
    there)."""
    attributes = net.attributes[node]
    if name in attributes:
        attributes[name] = value
    elif name in attributes.get("temp", {}):
        attributes["temp"][name] = value
    else:
        print(f"pyhgf's node {node} has no {name}; not set", file=sys.stderr)


def set_coupling(net, parent, child, strength, kind):
    """Set the [kind] ("value" or "volatility") coupling strength from
    [parent] to [child], on both sides of the edge."""
    k = list(getattr(net.edges[parent], f"{kind}_children")).index(child)
    a = net.attributes[parent]
    a[f"{kind}_coupling_children"] = (
        jnp.asarray(a[f"{kind}_coupling_children"]).at[k].set(strength)
    )
    k = list(getattr(net.edges[child], f"{kind}_parents")).index(parent)
    a = net.attributes[child]
    a[f"{kind}_coupling_parents"] = (
        jnp.asarray(a[f"{kind}_coupling_parents"]).at[k].set(strength)
    )
''';

  /// hgf_fit's response models, on pyhgf's trajectories (NaN responses are
  /// missed ones; a NaN surprise counts as infinite, as in hgf_fit).
  static const _readouts = r'''
def _total(surprise, responses):
    surprise = jnp.where(jnp.isnan(responses), 0.0, surprise)
    return jnp.sum(jnp.where(jnp.isnan(surprise), jnp.inf, surprise))


def gaussian_readout_surprise(net, responses, node, field, precision):
    """Responses ~ N(belief, 1 / precision) (hgf_fit's GaussianReadout)."""
    d = responses - net.node_trajectories[node][field]
    s = 0.5 * (jnp.log(2 * jnp.pi) - jnp.log(precision) + precision * d * d)
    return _total(s, responses)


def _binary_surprise(x, p):
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    return jnp.where(x != 0, -jnp.log(p), -jnp.log(1 - p))


def binary_softmax_surprise(net, responses, node, beta):
    """Binary choices, P(1) = m^β / (m^β + (1 − m)^β) with m the node's
    prediction (hgf_fit's BinarySoftmaxReadout; pyhgf's
    binary_softmax_inverse_temperature)."""
    m = net.node_trajectories[node]["expected_mean"]
    p = m**beta / (m**beta + (1 - m) ** beta)
    return _total(_binary_surprise(responses, p), responses)


def categorical_softmax_surprise(net, responses, node, beta):
    """Category k chosen with probability ∝ μ̂ₖ^β, μ̂ₖ the prediction of its
    binary node (hgf_fit's CategoricalSoftmaxReadout)."""
    binary = net.edges[node].value_parents
    logits = beta * jnp.log(
        jnp.stack([net.node_trajectories[b]["expected_mean"] for b in binary], 1)
    )
    logp = jax.nn.log_softmax(logits, axis=1)
    k = jnp.nan_to_num(responses).astype(int)
    s = -jnp.take_along_axis(logp, k[:, None], axis=1)[:, 0]
    return _total(s, responses)


def softmax_surprise(net, responses, options, field, beta):
    """Option k chosen with probability ∝ exp(β vₖ), vₖ = Σ weight · belief
    over its nodes (hgf_fit's SoftmaxReadout)."""
    values = jnp.stack(
        [
            sum(w * net.node_trajectories[n][field] for n, w in option.items())
            for option in options
        ],
        1,
    )
    logp = jax.nn.log_softmax(beta * values, axis=1)
    k = jnp.nan_to_num(responses).astype(int)
    s = -jnp.take_along_axis(logp, k[:, None], axis=1)[:, 0]
    return _total(s, responses)


def categorical_surprise(net):
    """The surprise of the categorical inputs (hgf.dart's
    CategoricalSurprise: the categorical nodes' own surprise)."""
    s = sum(
        net.node_trajectories[i]["surprise"]
        for i in net.input_idxs
        if "binary_idxs" in net.attributes[i]
    )
    return jnp.sum(jnp.where(jnp.isnan(s), jnp.inf, s))
''';

  static const _readCsv = r'''
def read_csv(path):
    """Reads a CSV with the columns above (a header row, then one row per
    trial; `#` lines are comments)."""
    with open(path, newline="") as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    header = [h.strip() for h in rows[0]]
    rows = rows[1:]

    def column(name):
        if name not in header:
            raise ValueError(f"{path} has no column {name!r}")
        return header.index(name)

    def cell(row, c):
        s = row[c].strip() if c < len(row) else ""
        return float(s) if s else float("nan")

    in_cols = [[column(n) for n in c] for c in INPUT_COLUMNS]
    flags = all(n in header for n in OBSERVED_COLUMNS)
    inputs = np.array([[cell(r, c) for cols in in_cols for c in cols] for r in rows])
    if flags:
        observed = np.array([[cell(r, column(n)) for n in OBSERVED_COLUMNS] for r in rows])
    else:
        observed = np.array(
            [[0.0 if any(np.isnan(cell(r, c)) for c in cols) else 1.0 for cols in in_cols]
             for r in rows]
        )
        inputs = np.nan_to_num(inputs, nan=0.0)
    responses = None
    if RESPONSE_COLUMNS:
        rc = [column(n) for n in RESPONSE_COLUMNS]
        responses = np.array([[cell(r, c) for c in rc] for r in rows])
        if len(rc) == 1:
            responses = responses[:, 0]
    time_steps = None
    if TIME_STEP_COLUMN is not None:
        time_steps = np.array([cell(r, column(TIME_STEP_COLUMN)) for r in rows])
    return dict(
        inputs=inputs,
        observed=observed if flags or (observed == 0).any() else None,
        time_steps=time_steps,
        responses=responses,
    )
''';
}

/// [v] as a Python float literal (`float("nan")` etc. when not finite).
String pyFloat(double v) {
  if (v.isNaN) return 'float("nan")';
  if (v == double.infinity) return 'float("inf")';
  if (v == double.negativeInfinity) return '-float("inf")';
  var s = v.toString().replaceFirst('e+', 'e');
  if (v == v.roundToDouble() && v.abs() >= 1e6) {
    final e = v.toStringAsExponential().replaceFirst('e+', 'e');
    if (double.parse(e) == v && e.length < s.length) s = e;
  }
  if (!s.contains('.') && !s.contains('e')) s = '$s.0';
  return s;
}

/// [name] as a snake-case Python identifier.
String pyIdentifier(String name) {
  final s = name
      .replaceAllMapped(RegExp('([a-z0-9])([A-Z])'), (m) => '${m[1]}_${m[2]}')
      .toLowerCase()
      .replaceAll(RegExp('[^a-z0-9]+'), '_')
      .replaceAll(RegExp('^_+|_+\$'), '');
  if (s.isEmpty) return 'unnamed';
  return RegExp('^[0-9]').hasMatch(s) ? 'x_$s' : s;
}
