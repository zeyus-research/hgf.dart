part of 'python.dart';

/// The script of a model document ([ExportSpec.layout]): every participant
/// of the CSV read as hgf.dart reads it (labels, participants, time, the
/// transforms), one PyMC model over all of them (individual parameters per
/// participant, pooled ones around a population, shared ones once), the
/// estimates and the outputs per row written next to the data.
extension _PythonDocument on PythonExport {
  String _documentScript() {
    final layout = spec.layout!;
    final model = spec.model;
    final fit = spec.fit;
    final group = spec.group;
    final grouped = group != null && !group.isEmpty;
    final outputs = spec.outputs;
    final network = buildNetwork();
    final params = [
      for (final p in (model?['parameters'] as List<Object?>?) ?? const [])
        p as Map<String, Object?>,
    ];
    final fitCode = model == null || fit == null
        ? null
        : _documentFitCode(model, params, fit, group);
    final transforms = _transformCode(layout);

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
            : spec.csv != null
            ? '    python <this file> [data.csv]   (default: the data below)'
            : '    python <this file> data.csv',
      )
      ..writeln()
      ..writeln(
        'It reads every participant of the CSV (one row per time step, the '
        'columns',
      )
      ..writeln(
        'in the LAYOUT below) as hgf.dart does${fitCode == null ? '.' : ','}',
      );
    if (fitCode != null) {
      b
        ..writeln(switch (fit) {
          ExportMap() when grouped && group.shared.isNotEmpty =>
            'samples the joint posterior of the participants and the group '
                'level\n(NUTS; shared parameters have no empirical Bayes '
                'fit),',
          ExportMap() when grouped =>
            'fits the populations by empirical Bayes (Laplace-EM, as '
                'hgf.dart\'s\nfitEmpiricalBayes) and each participant\'s '
                'mode under them,',
          ExportMap() =>
            'finds each participant\'s MAP estimate (with Laplace intervals),',
          ExportNuts() when grouped =>
            'samples the joint posterior of the participants and the group '
                'level (NUTS),',
          ExportNuts() => 'samples each participant\'s posterior (NUTS),',
          null => '',
        })
        ..writeln('and writes <data>_estimates.csv.');
    }
    if (outputs.isNotEmpty) {
      b.writeln(
        'Then <data>_outputs.csv: the data with a column per named output, '
        'each\nparticipant filtered at '
        '${fitCode == null ? 'the parameters set here' : 'their estimates'}.',
      );
    }
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
      ..writeln('import math  # noqa: F401')
      ..writeln('import os')
      ..writeln('import re')
      ..writeln('import sys')
      ..writeln('import warnings')
      ..writeln('from functools import partial  # noqa: F401')
      ..writeln()
      ..writeln('# float64, as hgf.dart (set before JAX is imported).')
      ..writeln('os.environ.setdefault("JAX_ENABLE_X64", "1")')
      ..writeln()
      ..writeln('import jax  # noqa: E402,F401')
      ..writeln('import jax.numpy as jnp  # noqa: E402')
      ..writeln('import numpy as np  # noqa: E402')
      ..writeln(
        layout.deep.isEmpty
            ? 'from pyhgf.model import Network  # noqa: E402'
            : 'import optax  # noqa: E402\n'
                  'from pyhgf.model import DeepNetwork, Network  # noqa: E402',
      );
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
      ..writeln('warnings.filterwarnings("ignore", category=FutureWarning)')
      ..writeln(
        '# PyTensor runs the wrapped JAX in Python ("object mode"): '
        'expected.',
      )
      ..writeln('warnings.filterwarnings("ignore", message=".*object mode.*")')
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
      ..write(PythonExport._helpers);
    if (model?['response']
        case {'type': 'ExprReadout'} && final Map<String, Object?> r) {
      b
        ..writeln()
        ..writeln()
        ..write(PythonExport.exprReadoutFunction(ExprReadout.fromJson(r)));
    }
    if (params.isNotEmpty) {
      b
        ..writeln()
        ..writeln()
        ..write(_setParameters(params));
    }
    if (fitCode != null) {
      b
        ..writeln()
        ..writeln()
        ..write(PythonExport._readouts)
        ..writeln()
        ..writeln()
        ..write(fitCode);
    }

    // The layout.
    String strs(List<String> xs) => '[${xs.map(_pyStr).join(', ')}]';
    String opt(String? s) => s == null ? 'None' : _pyStr(s);
    b
      ..writeln()
      ..writeln()
      ..writeln(
        '# How the CSV is read: per input node (in the network\'s input '
        'order) its',
      )
      ..writeln(
        '# columns, or one column of category labels (as an index, or '
        'one-hot); an',
      )
      ..writeln(
        '# empty cell is an unobserved input, unless "observed" names a '
        'column of',
      )
      ..writeln(
        '# 0/1 flags. Responses: option labels (or codes) where there are '
        'options;',
      )
      ..writeln('# an empty cell is a missed response.')
      ..writeln('LAYOUT = dict(')
      ..writeln('    inputs=[');
    for (final i in layout.inputs) {
      b.writeln(
        '        dict(name=${_pyStr(i.name)}, columns=${strs(i.columns)}, '
        'categories=${i.categories == null ? 'None' : strs(i.categories!)}, '
        'one_hot=${i.oneHot ? 'True' : 'False'}, '
        'observed=${opt(i.observedColumn)}),',
      );
    }
    b
      ..writeln('    ],')
      ..writeln('    responses=${strs(layout.responseColumns)},')
      ..writeln(
        '    options=${layout.options == null ? 'None' : strs(layout.options!)},',
      )
      ..writeln('    participant=${opt(layout.participantColumn)},')
      ..writeln('    time=${opt(layout.timeColumn)},')
      ..writeln('    time_step=${opt(layout.timeStepColumn)},')
      ..writeln('    group=${opt(layout.groupColumn)},')
      ..writeln(')')
      ..writeln()
      ..writeln()
      ..write(_dataHelpers)
      ..writeln()
      ..writeln()
      ..write(transforms);

    if (outputs.isNotEmpty) {
      final net = plan.build();
      b
        ..writeln()
        ..writeln()
        ..writeln('def outputs(net):')
        ..writeln(
          '    """The named outputs of one participant\'s beliefs (after '
          'input_data)."""',
        )
        ..writeln('    return {');
      for (final o in outputs) {
        final String code;
        if (o.field == 'surprise') {
          code = switch (net.kindOf(o.node)) {
            NodeKind.continuousState =>
              'gaussian_node_surprise(net, ${o.node})',
            NodeKind.binaryState => 'binary_node_surprise(net, ${o.node})',
            _ =>
              'np.full(len(net.node_trajectories[${o.node}]["mean"]), np.nan)',
          };
        } else {
          code =
              'trajectory(net, ${o.node}, '
              '"${NodeField.parse(o.field).pyhgfName}")';
        }
        b.writeln('        ${_pyStr(o.name)}: $code,');
      }
      b
        ..writeln('    }')
        ..writeln()
        ..writeln()
        ..write(_outputHelpers);
    }

    if (spec.csv case final csv?) {
      b
        ..writeln()
        ..writeln()
        ..writeln('# The data the model was exported with.')
        ..writeln('EMBEDDED_CSV = """\\')
        ..write(csv.replaceAll(r'\', r'\\').replaceAll('"""', r'"""'));
      if (!csv.endsWith('\n')) b.writeln();
      b.writeln('"""');
    }

    final load = spec.dataFile != null
        ? 'path = sys.argv[1] if len(sys.argv) > 1 else "${spec.dataFile}"\n'
              '    with open(path, newline="") as f:\n'
              '        table = parse_csv(f.read())'
        : spec.csv != null
        ? 'path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"\n'
              '    if len(sys.argv) > 1:\n'
              '        with open(path, newline="") as f:\n'
              '            table = parse_csv(f.read())\n'
              '    else:\n'
              '        table = parse_csv(EMBEDDED_CSV)'
        : 'path = sys.argv[1]\n'
              '    with open(path, newline="") as f:\n'
              '        table = parse_csv(f.read())';
    b
      ..writeln()
      ..writeln()
      ..writeln('def main():')
      ..writeln('    $load')
      ..writeln('    base = re.sub(r"\\.(csv|tsv|txt)\$", "", path)')
      ..writeln('    table = transform(table)')
      ..writeln('    ids, subjects = participants(table)')
      ..writeln(
        '    print(f"{len(ids)} participant(s), {len(table[\'rows\'])} rows.")',
      );
    if (fitCode == null) {
      b
        ..writeln('    for pid, data in zip(ids, subjects):')
        ..writeln('        net = build_network()')
        ..writeln('        net.input_data(')
        ..writeln('            input_data=data["inputs"],')
        ..writeln('            observed=data["observed"],')
        ..writeln('            time_steps=data["time_steps"],')
        ..writeln('        )')
        ..writeln('        means = [')
        ..writeln('            float(traj["mean"][-1])')
        ..writeln('            for node, traj in net.node_trajectories.items()')
        ..writeln(
          '            if node >= 0 and "mean" in traj and np.ndim(traj["mean"][-1]) == 0',
        )
        ..writeln('        ]')
        ..writeln(
          '        print(f"{pid}: {len(data[\'inputs\'])} rows; final means: '
          '{means}")',
        );
    } else {
      b
        ..writeln('    estimates = fit(ids, subjects)')
        ..writeln('    print_estimates(estimates)')
        ..writeln('    with open(base + "_estimates.csv", "w") as f:')
        ..writeln('        f.write(estimates_csv(estimates))')
        ..writeln('    print(f"Wrote {base}_estimates.csv")');
    }
    if (outputs.isNotEmpty) {
      final at = fitCode == null
          ? 'build_network()'
          : 'set_parameters(build_network(), '
                'estimates["participants"][pid]["estimate"])';
      b
        ..writeln('    columns = []')
        ..writeln('    for pid, data in zip(ids, subjects):')
        ..writeln('        net = $at')
        ..writeln('        net.input_data(')
        ..writeln('            input_data=data["inputs"],')
        ..writeln('            observed=data["observed"],')
        ..writeln('            time_steps=data["time_steps"],')
        ..writeln('        )')
        ..writeln('        columns.append(outputs(net))')
        ..writeln('    with open(base + "_outputs.csv", "w") as f:')
        ..writeln('        f.write(with_columns(table, columns))')
        ..writeln('    print(f"Wrote {base}_outputs.csv")');
    }
    b
      ..writeln()
      ..writeln()
      ..writeln('if __name__ == "__main__":')
      ..writeln('    main()');
    return b.toString();
  }

  /// `set_parameters(net, params)`: the network targets of [params].
  String _setParameters(List<Map<String, Object?>> params) {
    final b = StringBuffer()
      ..writeln('def set_parameters(net, params):')
      ..writeln(
        '    """[net] with the free parameters (a dict by name) set where '
        'they',
      )
      ..writeln('    target the network."""');
    for (final p in params) {
      final v = 'params["${p['name']}"]';
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
          case ResponseTarget():
        }
      }
    }
    b.writeln('    return net');
    return b.toString();
  }

  /// `negative_log_likelihood`, the PyMC model over all participants, and
  /// `fit(ids, subjects)`.
  String _documentFitCode(
    Map<String, Object?> model,
    List<Map<String, Object?>> params,
    ExportFit fit,
    DocGroup? group,
  ) {
    final pooled = group?.pooled ?? const <String, DocPopulation>{};
    final shared = group?.shared ?? const <String>[];
    final grouped = pooled.isNotEmpty || shared.isNotEmpty;
    final names = [for (final p in params) p['name'] as String];
    // The single-subject likelihood (and its fit, unused here).
    final single = _fitCode(model, fit);
    final nll = single.substring(0, single.indexOf('\n\ndef fit(data):'));
    final b = StringBuffer()
      ..writeln(nll)
      ..writeln()
      ..writeln('PARAMETERS = [${names.map(_pyStr).join(', ')}]')
      ..writeln(
        'POOLED = [${[for (final n in names)
          if (pooled.containsKey(n)) _pyStr(n)].join(', ')}]',
      )
      ..writeln(
        'SHARED = [${[for (final n in names)
          if (shared.contains(n)) _pyStr(n)].join(', ')}]',
      )
      ..writeln()
      ..writeln()
      ..writeln('def likelihood_of(data):')
      ..writeln(
        '    """One participant\'s log-likelihood as a PyTensor op: JAX, '
        'compiled once',
      )
      ..writeln('    (jit) and wrapped, gradients included."""')
      ..writeln('    from pytensor import wrap_jax')
      ..writeln()
      ..writeln('    @jax.jit')
      ..writeln('    def f(*xs):')
      ..writeln(
        '        return -negative_log_likelihood(dict(zip(PARAMETERS, xs)), '
        'data)',
      )
      ..writeln()
      ..writeln('    @wrap_jax')
      ..writeln('    def log_likelihood(*xs):')
      ..writeln('        return f(*xs)')
      ..writeln()
      ..writeln('    return log_likelihood')
      ..writeln()
      ..writeln()
      ..writeln('def build_model(ids, subjects, populations=None, pool=True):')
      ..writeln(
        '    """One PyMC model over [subjects] (participants [ids]): '
        'individual',
      )
      ..writeln(
        '    parameters per participant with their priors, pooled ones '
        'around a',
      )
      ..writeln(
        '    population (mean and sd on the parameter\'s unconstrained scale, '
        'as',
      )
      ..writeln(
        '    hgf.dart), shared ones once; each participant\'s log-likelihood '
        '(JAX,',
      )
      ..writeln(
        '    wrapped for PyTensor) as a potential. With [populations] '
        '(name: (mean,',
      )
      ..writeln(
        '    sd), fixed) the pooled parameters have those instead of a '
        'population',
      )
      ..writeln(
        '    of their own; with pool=False they have their own priors, as '
        'if not',
      )
      ..writeln('    pooled."""')
      ..writeln('    import pymc as pm')
      ..writeln()
      ..writeln('    n = len(ids)  # noqa: F841')
      ..writeln('    with pm.Model(coords={"participant": ids}) as model:')
      ..writeln('        values = {}');
    for (final p in params) {
      final name = p['name'] as String;
      final prior = p['prior'] as Map<String, Object?>;
      final n = _pyStr(name);
      if (shared.contains(name)) {
        b.writeln('        values[$n] = ${_prior(name, prior, p['initial'])}');
      } else if (pooled[name] case final pop?) {
        final t = p['transform'] as Map<String, Object?>;
        final centered = group!.centered;
        final init = p['initial'] == null
            ? null
            : 'np.full(n, ${pyFloat(readJsonDouble(p['initial']))})';
        b
          ..writeln('        if not pool:')
          ..writeln(
            '            values[$n] = '
            '${_prior(name, prior, null, initval: init, dims: 'participant')}',
          )
          ..writeln('        else:')
          ..writeln('            # $name varies around a population.')
          ..writeln('            if populations is not None:')
          ..writeln('                mu, sigma = populations[$n]')
          ..writeln(
            '                y = pm.Normal("${name}_y", mu=mu, sigma=sigma, '
            'dims="participant")',
          )
          ..writeln('            else:')
          ..writeln(
            '                mu = ${_prior('${name}_mu', pop.mean, null)}',
          )
          ..writeln(
            '                sigma = ${_prior('${name}_sigma', pop.sd, null)}',
          );
        if (centered) {
          b.writeln(
            '                y = pm.Normal("${name}_y", mu=mu, sigma=sigma, '
            'dims="participant")',
          );
        } else {
          b
            ..writeln(
              '                z = pm.Normal("${name}_z", mu=0.0, sigma=1.0, '
              'dims="participant")',
            )
            ..writeln('                y = mu + sigma * z');
        }
        b.writeln(
          '            values[$n] = pm.Deterministic($n, '
          '${_constrained(t, 'y')}, dims="participant")',
        );
      } else {
        final init = p['initial'] == null
            ? null
            : 'np.full(n, ${pyFloat(readJsonDouble(p['initial']))})';
        b.writeln(
          '        values[$n] = '
          '${_prior(name, prior, null, initval: init, dims: 'participant')}',
        );
      }
    }
    b
      ..writeln('        for s, data in enumerate(subjects):')
      ..writeln(
        '            xs = [values[p] if p in SHARED else values[p][s] for p in '
        'PARAMETERS]',
      )
      ..writeln(
        '            pm.Potential(f"hgf_{s}", likelihood_of(data)(*xs))',
      )
      ..writeln('    return model')
      ..writeln()
      ..writeln();
    final transforms = {
      for (final p in params)
        p['name'] as String: _constrained(
          p['transform'] as Map<String, Object?>,
          'y',
          numpy: true,
        ),
    };
    b
      ..writeln('# Each parameter on its natural scale from its unconstrained')
      ..writeln("# value y (hgf.dart's transforms; a population lives on y).")
      ..writeln('TO_CONSTRAINED = {');
    for (final MapEntry(:key, :value) in transforms.entries) {
      b.writeln('    ${_pyStr(key)}: lambda y: $value,');
    }
    b
      ..writeln('}')
      ..writeln('TO_UNCONSTRAINED = {');
    for (final p in params) {
      b.writeln(
        '    ${_pyStr(p['name'] as String)}: lambda x: '
        '${_unconstrained(p['transform'] as Map<String, Object?>, 'x')},',
      );
    }
    final (restarts, seed) = switch (fit) {
      ExportMap(:final restarts, :final seed) => (restarts, seed),
      ExportNuts(:final restarts) => (restarts, 1),
    };
    b
      ..writeln('}')
      ..writeln()
      ..writeln(
        '# MAP: from the starting values and RESTARTS more starts drawn '
        'from the',
      )
      ..writeln('# priors (seeded by SEED); the best run is kept.')
      ..writeln('RESTARTS = $restarts')
      ..writeln('SEED = $seed')
      ..writeln()
      ..writeln()
      ..write(_optimiseHelpers);
    switch (fit) {
      case ExportMap() when !grouped:
        b
          ..writeln()
          ..writeln()
          ..write(_fitMapCode(fit));
      case ExportMap() when shared.isEmpty:
        b
          ..writeln()
          ..writeln()
          ..write(_fitEmpiricalBayesCode);
      case ExportMap():
      case ExportNuts():
        final nuts = fit is ExportNuts ? fit : const ExportNuts();
        b
          ..writeln()
          ..writeln()
          ..write(_fitNutsCode(nuts, grouped));
    }
    b
      ..writeln()
      ..writeln()
      ..write(_estimateHelpers);
    return b.toString();
  }

  String _fitMapCode(ExportMap fit) => '''
def fit(ids, subjects):
    """Each participant's MAP estimate (as hgf.dart, without a Jacobian),
    then 95 % intervals from the curvature there on the unconstrained scale
    (Laplace; central differences of the gradient)."""
    out = {"participants": {}, "group": []}
    for pid, data in zip(ids, subjects):
        model = build_model([pid], [data])
        x, info, _ = optimise(model, starts(model), jacobian=False)
        cov = laplace_covariance(model, x, info, jacobian=False)
        out["participants"][pid] = estimates_at(model, x, info, cov)
    return out
''';

  /// `fit(ids, subjects)` by empirical Bayes, as hgf_fit's
  /// `fitEmpiricalBayes` (pooled parameters only).
  static const _fitEmpiricalBayesCode = r'''
def fit(ids, subjects, max_iterations=200, tolerance=1e-4, min_sd=1e-3):
    """Empirical Bayes (Laplace-EM, as hgf.dart's fitEmpiricalBayes): each
    participant's mode under the current populations (density on the
    unconstrained scale), the Laplace covariance there, then each population
    set to the moments of the participants' approximate posteriors (mean of
    the modes; variance of the modes plus the mean posterior variance),
    until the summed Laplace log evidence changes by less than [tolerance].
    The populations start from each participant's MAP estimate under the
    parameters' own priors. No group-level intervals (type-II maximum
    likelihood)."""
    import pytensor

    n = len(ids)
    # Each participant's MAP estimate under the parameters' own priors.
    first = []
    for pid, data in zip(ids, subjects):
        model = build_model([pid], [data], pool=False)
        x, info, nlp = optimise(model, starts(model), jacobian=False)
        first.append(estimates_at(model, x, info)["estimate"] if np.isfinite(nlp) else None)
    populations = {}
    for p in POOLED:
        ys = np.array([TO_UNCONSTRAINED[p](e[p]) for e in first if e is not None])
        mu = float(np.mean(ys)) if len(ys) else 0.0
        sd = max(float(np.std(ys, ddof=1)), 0.1) if len(ys) > 1 else 1.0
        populations[p] = (pytensor.shared(mu), pytensor.shared(sd))

    models = [
        build_model([pid], [data], populations=populations)
        for pid, data in zip(ids, subjects)
    ]
    # Warm starts: the first estimates, pooled ones on their y scale.
    points = []
    for model, e in zip(models, first):
        if e is None:
            points.append(model.initial_point())
            continue
        overrides = {}
        for p in PARAMETERS:
            if p in POOLED:
                overrides[p + "_y"] = np.full(1, TO_UNCONSTRAINED[p](e[p]))
            else:
                overrides[p] = np.full(1, e[p])
        points.append(initial_point(model, overrides))

    history = []
    for _ in range(max_iterations):
        found = [optimise(m, [pt], jacobian=True) for m, pt in zip(models, points)]
        points = [to_point(x, info) for x, info, _ in found]
        covs, total = [], 0.0
        for m, (x, info, nlp) in zip(models, found):
            h = negative_hessian(m, x, info, jacobian=True)
            try:
                c = np.linalg.cholesky(h)
            except np.linalg.LinAlgError:
                covs.append(None)
                continue
            covs.append(np.linalg.inv(h))
            total += -nlp + len(x) * math.log(2 * math.pi) / 2 - np.sum(np.log(np.diag(c)))
        history.append(total)
        for p in POOLED:
            ys, var = [], []
            for (x, info, _), cov in zip(found, covs):
                at = position(info, p + "_y")
                ys.append(x[at])
                var.append(0.0 if cov is None else cov[at, at])
            ys = np.array(ys)
            mu = float(np.mean(ys))
            sd = max(math.sqrt(float(np.mean((ys - mu) ** 2 + np.array(var)))), min_sd)
            populations[p][0].set_value(mu)
            populations[p][1].set_value(sd)
        if len(history) >= 2 and abs(history[-1] - history[-2]) < tolerance:
            break
    print(
        f"empirical Bayes: {len(history)} iterations, log evidence {history[-1]:.3f}"
        f"{'' if len(history) >= 2 and abs(history[-1] - history[-2]) < tolerance else ' (not converged)'}"
    )

    out = {"participants": {}, "group": []}
    for pid, m, (x, info, _), cov in zip(ids, models, found, covs):
        out["participants"][pid] = estimates_at(m, x, info, cov)
    for p in PARAMETERS:
        if p in POOLED:
            mu, sd = (float(v.get_value()) for v in populations[p])
            out["group"].append((p + "_typical", TO_CONSTRAINED[p](mu), np.nan, np.nan))
            out["group"].append((p + "_spread", sd, np.nan, np.nan))
    return out
''';

  /// The optimiser and Laplace helpers the MAP and empirical Bayes fits
  /// share.
  static const _optimiseHelpers = r'''
_COMPILED = {}


def compiled(model, jacobian):
    """The model's log density and its gradient, compiled once per model."""
    key = (id(model), jacobian)
    if key not in _COMPILED:
        _COMPILED[key] = (
            model,
            model.compile_logp(jacobian=jacobian),
            model.compile_dlogp(jacobian=jacobian),
        )
    return _COMPILED[key][1:]


def to_point(x, info):
    from pymc.blocking import DictToArrayBijection, RaveledVars

    return DictToArrayBijection.rmap(RaveledVars(x, info))


def position(info, name):
    """Where the value variable [name] starts in a raveled point."""
    at = 0
    for entry in info:
        if entry[0] == name:
            return at
        at += int(np.prod(entry[1]))
    raise KeyError(name)


def initial_point(model, overrides=None, rng=None):
    """The model's starting point on the value variables' scale, with
    [overrides] (natural values by variable name)."""
    from pymc.initial_point import make_initial_point_fn

    fn = make_initial_point_fn(
        model=model, overrides=overrides or {}, jitter_rvs=set(), return_transformed=True
    )
    return fn(0 if rng is None else rng)


def starts(model):
    """The starting values, then RESTARTS draws from the priors."""
    import pymc as pm

    out = [model.initial_point()]
    rng = np.random.default_rng(SEED)
    for _ in range(RESTARTS):
        draws = pm.draw(model.free_RVs, random_seed=rng)
        out.append(initial_point(model, {rv.name: d for rv, d in zip(model.free_RVs, draws)}))
    return out


def optimise(model, points, jacobian):
    """L-BFGS-B on the model's value variables from each of [points]; the
    best run as (x, info, negative log density)."""
    from pymc.blocking import DictToArrayBijection
    from scipy.optimize import minimize

    logp, dlogp = compiled(model, jacobian)
    best = None
    for start in points:
        x0 = DictToArrayBijection.map({v.name: start[v.name] for v in model.value_vars})
        info = x0.point_map_info

        def f(x, info=info):
            v = float(logp(to_point(x, info)))
            return -v if np.isfinite(v) else np.inf

        def g(x, info=info):
            return -np.asarray(dlogp(to_point(x, info)), dtype=float)

        if not np.isfinite(f(x0.data)):
            continue
        r = minimize(f, x0.data, jac=g, method="L-BFGS-B", options={"maxiter": 1000})
        if best is None or r.fun < best[2]:
            best = (np.asarray(r.x, dtype=float), info, float(r.fun))
    if best is None:
        x0 = DictToArrayBijection.map({v.name: points[0][v.name] for v in model.value_vars})
        best = (x0.data, x0.point_map_info, np.inf)
    return best


def negative_hessian(model, x, info, jacobian):
    """The Hessian of the negative log density at [x] (central differences of
    the gradient), symmetrised."""
    _, dlogp = compiled(model, jacobian)

    def grad(x):
        return np.asarray(dlogp(to_point(x, info)), dtype=float)

    k = len(x)
    h = 1e-5 * np.maximum(1.0, np.abs(x))
    hess = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h[j]
        hess[:, j] = -(grad(x + e) - grad(x - e)) / (2 * h[j])
    return 0.5 * (hess + hess.T)


def laplace_covariance(model, x, info, jacobian):
    """The Laplace covariance at [x], or None without a positive definite
    Hessian."""
    h = negative_hessian(model, x, info, jacobian)
    try:
        np.linalg.cholesky(h)
    except np.linalg.LinAlgError:
        return None
    return np.linalg.inv(h)


def estimates_at(model, x, info, cov=None):
    """Each parameter at [x] on its natural scale, with 95 % intervals from
    [cov] (the value variables' covariance) mapped through its transform."""
    import pytensor.tensor as pt

    point = to_point(x, info)
    sd = np.sqrt(np.diag(cov)) if cov is not None else np.full(len(x), np.nan)
    estimate, lower, upper = {}, {}, {}
    for rv in model.free_RVs:
        v = model.rvs_to_values[rv].name
        transform = model.rvs_to_transforms.get(rv)
        at = position(info, v)
        y, s = float(np.ravel(point[v])[0]), float(sd[at])

        def natural(y, rv=rv, transform=transform):
            if transform is None:
                return float(y)
            return float(
                transform.backward(pt.as_tensor(np.asarray(y)), *rv.owner.inputs).eval()
            )

        if rv.name.endswith("_y") and rv.name[:-2] in POOLED:
            # A pooled parameter's y, on hgf.dart's unconstrained scale.
            name, to = rv.name[:-2], TO_CONSTRAINED[rv.name[:-2]]
        else:
            name, to = rv.name, natural
        estimate[name] = float(to(y))
        if np.isfinite(s) and s > 0:
            lower[name] = float(to(y - 1.959963984540054 * s))
            upper[name] = float(to(y + 1.959963984540054 * s))
        else:
            lower[name] = upper[name] = np.nan
    return dict(
        estimate={p: estimate[p] for p in PARAMETERS},
        lower={p: lower[p] for p in PARAMETERS},
        upper={p: upper[p] for p in PARAMETERS},
    )
''';

  String _fitNutsCode(ExportNuts fit, bool grouped) =>
      '''
def fit(ids, subjects):
    """NUTS (PyMC) over ${grouped ? 'the participants and the group level jointly' : 'every participant (their posteriors are independent)'}:
    posterior means and central 95 % intervals."""
    import pymc as pm

    model = build_model(ids, subjects)
    with model:
        idata = pm.sample(
            draws=${fit.draws},
            tune=${fit.warmup},
            chains=${fit.chains},
            target_accept=${pyFloat(fit.targetAccept)},
            nuts={"max_treedepth": ${fit.maxTreeDepth}},
            random_seed=${fit.seed},
            # One process: JAX deadlocks in forked chain processes.
            cores=1,
            progressbar=False,
        )
    post = idata.posterior
    divergences = int(idata.sample_stats["diverging"].sum())
    print(f"{divergences} divergent transitions")

    def summary(draws):
        draws = np.ravel(np.asarray(draws, dtype=float))
        return (
            float(np.mean(draws)),
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        )

    out = {"participants": {}, "group": []}
    for s, pid in enumerate(ids):
        est, lo, hi = {}, {}, {}
        for p in PARAMETERS:
            draws = post[p].values
            if p not in SHARED:
                draws = draws[..., s]
            est[p], lo[p], hi[p] = summary(draws)
        out["participants"][pid] = dict(estimate=est, lower=lo, upper=hi)
    for p in PARAMETERS:
        if p in POOLED:
            m, a, b = summary(post[p + "_mu"].values)
            t = TO_CONSTRAINED[p]
            out["group"].append((p + "_typical", t(m), t(a), t(b)))
            out["group"].append((p + "_spread",) + summary(post[p + "_sigma"].values))
        elif p in SHARED:
            out["group"].append((p,) + summary(post[p].values))
    return out
''';

  /// [t] (a `FitCodec` transform) applied to the unconstrained [y], in
  /// PyMC (or numpy with [numpy]).
  String _constrained(Map<String, Object?> t, String y, {bool numpy = false}) {
    String d(String key) => pyFloat(readJsonDouble(t[key]));
    final exp = numpy ? 'np.exp' : 'pm.math.exp';
    final sigmoid = numpy ? '1.0 / (1.0 + np.exp(-$y))' : 'pm.math.sigmoid($y)';
    return switch (t['type']) {
      'Identity' => y,
      'LowerBound' => '${d('lower')} + $exp($y)',
      'UpperBound' => '${d('upper')} - $exp($y)',
      'Interval' =>
        '${d('lower')} + ${pyFloat(readJsonDouble(t['upper']) - readJsonDouble(t['lower']))} * ${numpy ? '($sigmoid)' : sigmoid}',
      final type =>
        '${_stub('${pyIdentifier('$type')}_transform', 'the transform "$type" (natural value from unconstrained y)')}($y)',
    };
  }

  /// [t] (a `FitCodec` transform) inverted: the unconstrained value of the
  /// natural [x], in numpy.
  String _unconstrained(Map<String, Object?> t, String x) {
    String d(String key) => pyFloat(readJsonDouble(t[key]));
    return switch (t['type']) {
      'Identity' => x,
      'LowerBound' => 'np.log($x - ${d('lower')})',
      'UpperBound' => 'np.log(${d('upper')} - $x)',
      'Interval' => 'np.log(($x - ${d('lower')}) / (${d('upper')} - $x))',
      final type =>
        '${_stub('${pyIdentifier('$type')}_untransform', 'the inverse of the transform "$type" (unconstrained y from natural value)')}($x)',
    };
  }

  /// `transform(table)`: the document's data transforms, as hgf.dart
  /// computes them.
  static String _transformCode(DataLayout layout) {
    final b = StringBuffer()
      ..writeln('def transform(table):')
      ..writeln(
        '    """The table with the data transforms\' columns appended (as '
        'hgf.dart',
      )
      ..writeln('    computes them)."""');
    if (layout.clock != null) b.writeln('    table = on_clock(table)');
    if (layout.steps.isNotEmpty) b.writeln('    table = run_steps(table)');
    if (layout.transforms.isNotEmpty) {
      b.writeln(
        '    t = Transforms(table, LAYOUT["participant"], LAYOUT["group"], '
        'LAYOUT["time"])',
      );
      for (final t in layout.transforms) {
        b
          ..writeln('    # ${t.name} = ${t.expression}')
          ..writeln('    t.add(${_pyStr(t.name)}, ${_dataExpr(t.expression)})');
      }
      b.writeln('    table = t.table');
    }
    if (layout.deep.isNotEmpty) {
      b.writeln(
        '    # The deep networks, trained after the transforms: their '
        'predictions are columns.',
      );
      for (final d in layout.deep) {
        b.writeln('    table = ${pyDeepTrainFunction(d)}(table)');
      }
    }
    b.writeln('    return table');
    return _clockCode(layout) +
        _stepsCode(layout) +
        (layout.deep.isEmpty ? '' : '${pyDeepFunctions(layout.deep)}\n\n') +
        b.toString();
  }

  /// `STEPS` and `run_steps(table)`: the data-pipeline steps, as hgf.dart's
  /// `DocStep`s (empty without steps).
  static String _stepsCode(DataLayout layout) {
    if (layout.steps.isEmpty) return '';
    String fn(Expr e) => 'lambda t: ${_dataExpr(e)}';
    String opt(Expr? e) => e == null ? 'None' : fn(e);
    String strs(List<String> xs) => '[${xs.map(_pyStr).join(', ')}]';
    final b = StringBuffer()
      ..writeln("# The data-pipeline steps (hgf.dart's DocStep), in order.")
      ..writeln('STEPS = [');
    for (final st in layout.steps) {
      final args = switch (st) {
        DecimateStep() => [
          'factor=${st.factor}',
          'how=${_pyStr(st.how.name)}',
          'columns={${[for (final MapEntry(:key, :value) in st.columns.entries) '${_pyStr(key)}: ${_pyStr(value.name)}'].join(', ')}}',
          'within=${strs(st.within)}',
        ],
        SegmentStep() => [
          'at=${fn(st.at)}',
          'before=${st.before}',
          'after=${st.after ?? 'None'}',
          'column=${_pyStr(st.column)}',
          'offset=${st.offsetColumn == null ? 'None' : _pyStr(st.offsetColumn!)}',
        ],
        FilterStep() => ['keep=${fn(st.keep)}'],
        StandardiseStep() => [
          'columns=${strs(st.columns)}',
          'per_participant=${st.perParticipant ? 'True' : 'False'}',
          'fit_on=${opt(st.fitOn)}',
        ],
        SplitStep() => [
          'column=${_pyStr(st.column)}',
          'parts=[${[for (final p in st.parts) '(${_pyStr(p.label)}, ${pyFloat(p.fraction)})'].join(', ')}]',
          'by_participant=${st.byParticipant ? 'True' : 'False'}',
          'seed=${st.seed}',
        ],
      };
      b
        ..writeln('    # ${st.kind}: ${st.summary}')
        ..writeln('    dict(kind=${_pyStr(st.kind)}, ${args.join(', ')}),');
    }
    b
      ..writeln(']')
      ..writeln()
      ..writeln()
      ..write(_stepHelpers)
      ..writeln()
      ..writeln();
    return b.toString();
  }

  static const _stepHelpers = r'''
def _step_index(table, name, what):
    if name not in table["header"]:
        raise ValueError(f'No column "{name}" ({what}) in the data')
    return table["header"].index(name)


def _step_units(table):
    if LAYOUT["participant"] is None:
        return [list(range(len(table["rows"])))]
    c = _step_index(table, LAYOUT["participant"], "participants")
    groups = {}
    for r, row in enumerate(table["rows"]):
        groups.setdefault(row[c].strip(), []).append(r)
    return list(groups.values())


def _step_eval(table, f):
    return f(Transforms(table, LAYOUT["participant"], LAYOUT["group"], LAYOUT["time"])).v


def _on(v):
    return not np.isnan(v) and v != 0


def _aggregate(table, block, c, a):
    rows = table["rows"]
    if a == "first":
        return rows[block[0]][c]
    if a == "last":
        return rows[block[-1]][c]
    if a == "mode":
        counts = {}
        for r in block:
            s = rows[r][c].strip()
            if s.lower() not in Transforms._MISSING_TEXT:
                counts[s] = counts.get(s, 0) + 1
        best = None
        for k, n in counts.items():
            if best is None or n > counts[best]:
                best = k
        return "" if best is None else best
    xs = []
    for r in block:
        v = number(table, rows[r][c])
        if v is None:
            raise ValueError(f'Column "{table["header"][c]}" has text ("{rows[r][c]}"), which "{a}" cannot aggregate')
        if not np.isnan(v):
            xs.append(v)
    if not xs:
        return ""
    if a in ("mean", "sum"):
        s = 0.0
        for x in xs:
            s += x
        return format_number(s / len(xs) if a == "mean" else s)
    if a == "min":
        return format_number(min(xs))
    if a == "max":
        return format_number(max(xs))
    return format_number(1.0 if any(x != 0 for x in xs) else 0.0)


def _decimate(table, s):
    header, rows = table["header"], table["rows"]
    if s["factor"] == 1 and not s["columns"] and s["how"] == "first":
        return table
    aggs = [s["columns"].get(h, s["how"]) for h in header]
    for k, h in enumerate(header):
        if h in s["columns"] or aggs[k] in ("first", "last", "mode"):
            continue
        if h in (LAYOUT["participant"], LAYOUT["group"], LAYOUT["time"]) or any(
            number(table, row[k]) is None for row in rows
        ):
            aggs[k] = "first"
    for c in s["columns"]:
        _step_index(table, c, "to aggregate")
    wc = [_step_index(table, w, "blocks end where it changes") for w in s["within"]]
    out = []
    for unit in _step_units(table):
        start = 0
        while start < len(unit):
            end = start + 1
            while (
                end < len(unit)
                and end - start < s["factor"]
                and all(rows[unit[end]][c].strip() == rows[unit[start]][c].strip() for c in wc)
            ):
                end += 1
            block = unit[start:end]
            out.append([_aggregate(table, block, c, a) for c, a in enumerate(aggs)])
            start = end
    return dict(table, rows=out)


def _segment(table, s):
    v = _step_eval(table, s["at"])
    header = list(table["header"])
    for name in (s["column"], s["offset"]):
        if name is not None and name not in header:
            header.append(name)
    sc = header.index(s["column"])
    oc = -1 if s["offset"] is None else header.index(s["offset"])
    before, after = s["before"], s["after"]
    out = []
    for unit in _step_units(table):
        onsets = [i for i, r in enumerate(unit) if _on(v[r])]
        seg = [-1] * len(unit)
        for k, o in enumerate(onsets):
            if after is None:
                end = onsets[k + 1] - before - 1 if k + 1 < len(onsets) else len(unit) - 1
            else:
                end = o + after
            for i in range(max(0, o - before), min(end, len(unit) - 1) + 1):
                seg[i] = k
        for i, r in enumerate(unit):
            k = seg[i]
            if k < 0:
                continue
            row = list(table["rows"][r]) + [""] * (len(header) - len(table["header"]))
            row[sc] = str(k + 1)
            if oc >= 0:
                row[oc] = str(i - onsets[k])
            out.append(row)
    return dict(table, header=header, rows=out)


def _filter(table, s):
    v = _step_eval(table, s["keep"])
    return dict(table, rows=[row for r, row in enumerate(table["rows"]) if _on(v[r])])


def _standardise(table, s):
    cs = [_step_index(table, c, "to standardise") for c in s["columns"]]
    fit = None if s["fit_on"] is None else _step_eval(table, s["fit_on"])
    rows = [list(r) for r in table["rows"]]
    units = _step_units(table) if s["per_participant"] else [list(range(len(rows)))]
    for c in cs:
        for unit in units:
            xs = []
            for r in unit:
                x = number(table, table["rows"][r][c])
                if x is None:
                    raise ValueError(f'Column "{table["header"][c]}" has text ("{table["rows"][r][c]}")')
                xs.append(x)
            use = [i for i, r in enumerate(unit) if not np.isnan(xs[i]) and (fit is None or _on(fit[r]))]
            if not use:
                raise ValueError(f'No rows to fit the mean and sd of "{table["header"][c]}" on')
            total = 0.0
            for i in use:
                total += xs[i]
            mean = total / len(use)
            ss = 0.0
            for i in use:
                ss += (xs[i] - mean) * (xs[i] - mean)
            sd = math.sqrt(ss / len(use))
            scale = sd if sd > 1e-12 else 1.0
            for i, r in enumerate(unit):
                rows[r][c] = format_number((xs[i] - mean) / scale)
    return dict(table, rows=rows)


def _shuffled(n, seed):
    # 0..n-1 shuffled with mulberry32, as hgf.dart's SplitStep.shuffled.
    state = seed & 0xFFFFFFFF

    def imul(a, b):
        return (a * b) & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = imul(state ^ (state >> 15), 1 | state)
        t = ((t + imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    out = list(range(n))
    for i in range(n - 1, 0, -1):
        j = math.floor(nxt() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


def _split(table, s):
    header = list(table["header"])
    if s["column"] not in header:
        header.append(s["column"])
    c = header.index(s["column"])
    rows = [list(r) + [""] * (len(header) - len(r)) for r in table["rows"]]
    parts = s["parts"]
    total = 0.0
    for _, f in parts:
        total += f

    def label(i, n):
        cum = 0.0
        for name, f in parts:
            cum += f
            if i < math.floor(cum / total * n + 0.5):
                return name
        return parts[-1][0]

    units = _step_units(table)
    if s["by_participant"]:
        for i, u in enumerate(_shuffled(len(units), s["seed"])):
            for r in units[u]:
                rows[r][c] = label(i, len(units))
    else:
        for unit in units:
            for i, r in enumerate(unit):
                rows[r][c] = label(i, len(unit))
    return dict(table, header=header, rows=rows)


def run_steps(table):
    # The table through the data-pipeline steps, in order.
    run = dict(decimate=_decimate, segment=_segment, filter=_filter, standardise=_standardise, split=_split)
    for i, s in enumerate(STEPS):
        try:
            table = run[s["kind"]](table, s)
        except ValueError as e:
            raise ValueError(f'Data step {i + 1} ({s["kind"]}): {e}') from None
    return table
''';

  /// `CLOCK` and `on_clock(table)`: events onto the shared clock, as
  /// hgf.dart's `DocClock` (empty without a clock).
  static String _clockCode(DataLayout layout) {
    final k = layout.clock;
    if (k == null) return '';
    String fn(Expr e) => 'lambda t: ${_dataExpr(e)}';
    final b = StringBuffer()
      ..writeln("# Events onto the shared clock (hgf.dart's DocClock).")
      ..writeln('CLOCK = dict(')
      ..writeln('    onset=${_pyStr(k.onsetColumn)},')
      ..writeln('    step=${pyFloat(k.step)},')
      ..writeln(
        '    session=${k.sessionColumn == null ? 'None' : _pyStr(k.sessionColumn!)},',
      )
      ..writeln('    streams=[');
    for (final st in k.streams) {
      b
        ..writeln('        dict(')
        ..writeln(
          '            when=${st.when == null ? 'None' : fn(st.when!)},  '
          '# ${st.when ?? 'every event'}',
        )
        ..writeln('            hold=[');
      for (final MapEntry(:key, :value) in st.hold.entries) {
        b.writeln('                (${_pyStr(key)}, ${fn(value)}),  # $value');
      }
      b
        ..writeln('            ],')
        ..writeln('        ),');
    }
    b
      ..writeln('    ],')
      ..writeln(
        '    keep=${k.keep == null ? 'None' : fn(k.keep!)},'
        '${k.keep == null ? '' : '  # ${k.keep}'}',
      )
      ..writeln(')')
      ..writeln()
      ..writeln()
      ..write(_clockHelpers)
      ..writeln()
      ..writeln();
    return b.toString();
  }

  static const _clockHelpers = r'''
def _cell(col, r):
    return col.text[r] if col.text is not None else format_number(col.v[r])


def _decimals(step):
    for d in range(9):
        s = step * 10**d
        if abs(s - round(s)) < 1e-9 * max(1, s):
            return d
    return 9


def on_clock(table):
    """The event table on the shared clock (hgf.dart's DocClock): one row per
    participant per step, each stream's columns held from an event until the
    stream's next one (empty before the first), on a grid of whole multiples
    of the step from each session's first event to its last. A table that
    already has the time column and no onset column is on the clock."""
    c, time = CLOCK, LAYOUT["time"]
    header = table["header"]
    if time in header and c["onset"] not in header:
        return table
    t = Transforms(table, LAYOUT["participant"], None, None)
    streams = c["streams"]
    whens = [None if s["when"] is None else s["when"](t) for s in streams]
    holds = [[(name, f(t)) for name, f in s["hold"]] for s in streams]

    def index(name, what):
        if name not in header:
            raise ValueError(f'No column "{name}" ({what}) in the data')
        return header.index(name)

    oc = index(c["onset"], "event times")
    pc = None if LAYOUT["participant"] is None else index(LAYOUT["participant"], "participants")
    sc = None if c["session"] is None else index(c["session"], "sessions")
    sessions, session_of, events = {}, {}, {}
    for r, row in enumerate(table["rows"]):
        onset = number(table, row[oc])
        if onset is None or np.isnan(onset):
            continue
        p = "" if pc is None else row[pc].strip()
        ses = "" if sc is None else row[sc].strip()
        found = False
        for i in range(len(streams)):
            w = whens[i]
            if w is not None and (np.isnan(w.v[r]) or w.v[r] == 0):
                continue
            found = True
            events.setdefault(p, [[] for _ in streams])[i].append(
                (onset, r, [_cell(col, r) for _, col in holds[i]])
            )
        if not found:
            continue
        if session_of.setdefault(p, ses) != ses:
            raise ValueError(f'Participant "{p}" is in sessions "{session_of[p]}" and "{ses}"')
        s = sessions.setdefault(ses, dict(people=[], lo=onset, hi=onset))
        if p not in s["people"]:
            s["people"].append(p)
        s["lo"], s["hi"] = min(s["lo"], onset), max(s["hi"], onset)
    step, digits = c["step"], _decimals(c["step"])
    out = []
    for ses, s in sessions.items():
        k0, k1 = math.floor(s["lo"] / step), math.ceil(s["hi"] / step)
        times = [float(f"{k * step:.{digits}f}") for k in range(k0, k1 + 1)]
        for p in s["people"]:
            evs = [sorted(e, key=lambda x: (x[0], x[1])) for e in events[p]]
            at = [-1] * len(streams)
            for tm in times:
                row = ([] if pc is None else [p]) + ([] if sc is None else [ses]) + [format_number(tm)]
                for i, e in enumerate(evs):
                    while at[i] + 1 < len(e) and e[at[i] + 1][0] <= tm + step * 1e-9:
                        at[i] += 1
                    row += [""] * len(holds[i]) if at[i] < 0 else e[at[i]][2]
                out.append(row)
    names = (
        ([] if pc is None else [LAYOUT["participant"]])
        + ([] if sc is None else [c["session"]])
        + [time]
        + [name for h in holds for name, _ in h]
    )
    grid = dict(header=names, rows=out, decimal_comma=table["decimal_comma"])
    if c["keep"] is not None:
        k = c["keep"](Transforms(grid, None, None, None))
        grid["rows"] = [row for r, row in enumerate(out) if not np.isnan(k.v[r]) and k.v[r] != 0]
    return grid
''';

  /// [e] as a column of a data transform (see the `Transforms` helper).
  static String _dataExpr(Expr e) {
    switch (e) {
      case ExprNum(:final value):
        return 't.const(${pyFloat(value)})';
      case ExprVar(:final name):
        return 't.col(${_pyStr(name)})';
      case ExprStr(:final value):
        throw FormatException(
          'The string \'$value\' can only be compared with a column',
        );
      case ExprBinary(:final op, :final a, :final b)
          when (op == '==' || op == '!=') && (a is ExprStr || b is ExprStr):
        final s = (a is ExprStr ? a : b as ExprStr).value;
        final other = a is ExprStr ? b : a;
        return 't.eq(${_dataExpr(other)}, ${_pyStr(s)}, '
            '${op == '==' ? 'True' : 'False'})';
      case ExprCall(:final fn, :final args)
          when fn.dataOnly || fn.name == 'where':
        final xs = [for (final a in args) _dataExpr(a)];
        int k(int i) =>
            args.length > i ? (args[i] as ExprNum).value.toInt() : 1;
        return switch (fn.name) {
          'lag' => 't.lag(${xs[0]}, ${k(1)})',
          'diff' => 't.diff(${xs[0]})',
          'mate' => 't.mate(${xs[0]}, ${k(1)})',
          'mates_sum' => 't.mates(${xs[0]}, "sum")',
          'mates_mean' => 't.mates(${xs[0]}, "mean")',
          'mates_count' => 't.mates(${xs[0]}, "count")',
          'missing' => 't.missing(${xs[0]})',
          'fill' => 't.fill(${xs[0]}, ${xs[1]})',
          'where' => 't.where(${xs[0]}, ${xs[1]}, ${xs[2]})',
          _ => throw UnsupportedError('Unknown data function ${fn.name}'),
        };
      default:
        // Pure arithmetic: one function of its columns.
        final leaves = <String>[];
        final body = _fused(e, leaves);
        final params = [for (var i = 0; i < leaves.length; i++) 'a$i'];
        return 't.map(lambda ${params.join(', ')}: $body'
            '${leaves.isEmpty ? '' : ', ${leaves.join(', ')}'})';
    }
  }

  /// [e]'s arithmetic over numpy arrays, every column it reads (or data
  /// function it calls) added to [leaves] as `a<i>`. NaN propagates
  /// through everything, as in hgf.dart's transforms.
  static String _fused(Expr e, List<String> leaves) {
    String leaf(Expr x) {
      leaves.add(_dataExpr(x));
      return 'a${leaves.length - 1}';
    }

    switch (e) {
      case ExprNum(:final value):
        return pyFloat(value);
      case ExprVar():
        return leaf(e);
      case ExprStr():
        return leaf(e);
      case ExprNeg(:final a):
        return '(-${_fused(a, leaves)})';
      case ExprBinary(:final op, :final a, :final b)
          when (op == '==' || op == '!=') && (a is ExprStr || b is ExprStr):
        return leaf(e);
      case ExprBinary(:final op, :final a, :final b) when e.isComparison:
        return '_cmp("$op", ${_fused(a, leaves)}, ${_fused(b, leaves)})';
      case ExprBinary(op: '^', :final a, :final b):
        return '_pow(${_fused(a, leaves)}, ${_fused(b, leaves)})';
      case ExprBinary(:final op, :final a, :final b):
        return '(${_fused(a, leaves)} $op ${_fused(b, leaves)})';
      case ExprCall(:final fn) when fn.dataOnly || fn.name == 'where':
        return leaf(e);
      case ExprCall(:final fn, :final args):
        final xs = [for (final a in args) _fused(a, leaves)];
        return switch (fn.name) {
          'exp' ||
          'log' ||
          'sqrt' ||
          'abs' ||
          'sign' ||
          'tanh' => 'np.${fn.name}(${xs[0]})',
          'sigmoid' => '(1.0 / (1.0 + np.exp(-${xs[0]})))',
          'min' => 'np.minimum(${xs[0]}, ${xs[1]})',
          'max' => 'np.maximum(${xs[0]}, ${xs[1]})',
          'clip' => 'np.minimum(np.maximum(${xs[0]}, ${xs[1]}), ${xs[2]})',
          _ => throw UnsupportedError('Unknown function ${fn.name}'),
        };
    }
  }

  static String _pyStr(String s) =>
      '"${s.replaceAll(r'\', r'\\').replaceAll('"', r'\"')}"';

  static const _dataHelpers =
      '''
$pyCsvReader'''
      r'''def format_number(v):
    return "" if np.isnan(v) else str(int(v)) if v == int(v) and abs(v) < 1e15 else repr(float(v))


def participant_rows(table):
    """The rows of each participant, in order of first appearance."""
    groups = {}
    p = LAYOUT["participant"]
    if p is None:
        return {"": list(range(len(table["rows"])))}
    if p not in table["header"]:
        raise ValueError(f'No column "{p}" (participants) in the data')
    c = table["header"].index(p)
    for r, row in enumerate(table["rows"]):
        groups.setdefault(row[c].strip(), []).append(r)
    return groups


def participants(table):
    """The participants' ids and data (inputs, observed flags, time steps,
    responses), read by LAYOUT as hgf.dart reads them: a label is matched
    before a missing marker ("none" may be an option), an empty input cell is
    an unobserved input (read as 0), an empty response a missed one (NaN), and
    time steps come from the time-step column, or the differences of the time
    column (the first row takes the second's)."""
    header, rows = table["header"], table["rows"]

    def col(name, what):
        if name not in header:
            raise ValueError(
                f'No column "{name}" ({what}) in the data; its columns are '
                + ", ".join(header)
            )
        return header.index(name)

    def num(r, c):
        v = number(table, rows[r][c])
        if v is None:
            raise ValueError(f'Row {r + 2}, column "{header[c]}": "{rows[r][c]}" is not a number')
        return v

    def code(r, c, labels):
        cell = rows[r][c].strip()
        if cell in labels:
            return labels.index(cell)
        if cell.lower() in _MISSING:
            return None
        if re.fullmatch(r"[+-]?\d+", cell) and 0 <= int(cell) < len(labels):
            return int(cell)
        raise ValueError(f'Row {r + 2}, column "{header[c]}": "{cell}" is not one of {", ".join(labels)}')

    inputs = LAYOUT["inputs"]
    widths = [len(i["categories"]) if i["one_hot"] else len(i["columns"]) for i in inputs]
    width = sum(widths)
    in_cols = [[col(n, "input " + i["name"]) for n in i["columns"]] for i in inputs]
    obs_cols = [None if i["observed"] is None else col(i["observed"], "observed flags of " + i["name"]) for i in inputs]
    r_cols = [col(n, "responses") for n in LAYOUT["responses"]]
    t_col = None if LAYOUT["time"] is None else col(LAYOUT["time"], "time")
    dt_col = None if LAYOUT["time_step"] is None else col(LAYOUT["time_step"], "time steps")
    options = LAYOUT["options"]
    ids, subjects = [], []
    for pid, idx in participant_rows(table).items():
        n = len(idx)
        u = np.zeros((n, width))
        observed = np.ones((n, len(inputs)))
        masked = False
        for t, r in enumerate(idx):
            off = 0
            for k, i in enumerate(inputs):
                w = widths[k]
                seen = True
                if i["categories"] is not None:
                    c = code(r, in_cols[k][0], i["categories"])
                    if c is None:
                        seen = False
                    elif i["one_hot"]:
                        u[t, off + c] = 1.0
                    else:
                        u[t, off] = float(c)
                else:
                    for j in range(w):
                        v = num(r, in_cols[k][j])
                        if np.isnan(v):
                            seen = False
                        else:
                            u[t, off + j] = v
                if obs_cols[k] is not None:
                    flag = num(r, obs_cols[k]) != 0
                    if flag and not seen:
                        raise ValueError(f'Row {r + 2}: input "{i["name"]}" is empty but flagged observed')
                    seen = flag
                    masked = True
                if not seen:
                    observed[t, k] = 0.0
                    masked = True
                    if obs_cols[k] is None:
                        u[t, off : off + w] = 0.0
                off += w
        time_steps = None
        if dt_col is not None:
            time_steps = np.array([num(r, dt_col) for r in idx])
        elif t_col is not None:
            times = np.array([num(r, t_col) for r in idx])
            time_steps = np.zeros(n)
            time_steps[1:] = np.diff(times)
            if n > 0:
                time_steps[0] = time_steps[1] if n > 1 else 1.0
        responses = None
        if r_cols:
            responses = np.array(
                [
                    [
                        (float("nan") if (c := code(r, rc, options)) is None else float(c))
                        if options is not None
                        else num(r, rc)
                        for rc in r_cols
                    ]
                    for r in idx
                ]
            )
            if len(r_cols) == 1:
                responses = responses[:, 0]
        ids.append(pid if LAYOUT["participant"] is not None else str(len(ids) + 1))
        subjects.append(
            dict(
                inputs=u,
                observed=observed if masked else None,
                time_steps=time_steps,
                responses=responses,
            )
        )
    return ids, subjects


def _cmp(op, a, b):
    """A comparison as 1.0 / 0.0, NaN where either side is."""
    r = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
         "==": np.equal, "!=": np.not_equal}[op](a, b)
    return np.where(np.isnan(a) | np.isnan(b), np.nan, r.astype(float))


def _pow(a, b):
    return np.where(np.isnan(a) | np.isnan(b), np.nan, np.power(a, b))


class Col:
    """A column: numbers (NaN when missing or text) and, for a column of
    the data (or a lag/mate of one), the cells' text."""

    def __init__(self, v, text=None):
        self.v = np.asarray(v, dtype=float)
        self.text = text


class Transforms:
    """hgf.dart's data transforms: new columns from expressions over the
    data's columns, per participant (lag, diff), across a group's members at
    the same time (mate). A missing cell is NaN and stays NaN; text compares
    as written."""

    _MISSING_TEXT = {"", "na", "nan", "null", "n/a", "."}

    def __init__(self, table, participant, group, time):
        self.table = dict(table, header=list(table["header"]), rows=[list(r) for r in table["rows"]])
        self.n = len(table["rows"])
        self.participant, self.group, self.time = participant, group, time
        pc = None if participant is None else self._index(participant, "participants")
        self.participant_of = ["" if pc is None else r[pc].strip() for r in table["rows"]]
        self.previous = []
        last = {}
        for r, p in enumerate(self.participant_of):
            self.previous.append(last.get(p, -1))
            last[p] = r
        self.cols = {}
        self._mates = None

    def _index(self, name, what):
        if name not in self.table["header"]:
            raise ValueError(f'No column "{name}" ({what}) in the data')
        return self.table["header"].index(name)

    def add(self, name, col):
        if name in self.table["header"]:
            raise ValueError(f'Transform "{name}": the data already has a column of that name')
        self.cols[name] = col
        self.table["header"].append(name)
        for r, row in enumerate(self.table["rows"]):
            row.append(col.text[r] if col.text is not None else format_number(col.v[r]))

    def col(self, name):
        if name in self.cols:
            return self.cols[name]
        c = self._index(name, "a transform's variable")
        text = [row[c].strip() for row in self.table["rows"]]
        v = [number(self.table, row[c]) for row in self.table["rows"]]
        col = Col([np.nan if x is None else x for x in v], text)
        self.cols[name] = col
        return col

    def const(self, x):
        return Col(np.full(self.n, x))

    def map(self, f, *cols):
        if not cols:
            return Col(np.full(self.n, f()))
        with np.errstate(all="ignore"):
            v = np.asarray(f(*[c.v for c in cols]), dtype=float) * np.ones(self.n)
        missing = np.zeros(self.n, dtype=bool)
        for c in cols:
            missing |= np.isnan(c.v)
        return Col(np.where(missing, np.nan, v))

    def _shift(self, x, rows):
        v = np.array([np.nan if p < 0 else x.v[p] for p in rows])
        text = None if x.text is None else ["" if p < 0 else x.text[p] for p in rows]
        return Col(v, text)

    def lag(self, x, k=1):
        rows = []
        for r in range(self.n):
            p = r
            for _ in range(k):
                if p < 0:
                    break
                p = self.previous[p]
            rows.append(p)
        return self._shift(x, rows)

    def diff(self, x):
        return self.map(lambda a, b: a - b, x, self.lag(x, 1))

    def mate(self, x, k=1):
        if self._mates is None:
            self._mates = self._build_mates()
        return self._shift(x, [m[k - 1] if k <= len(m) else -1 for m in self._mates])

    def mates(self, x, fn):
        """Over the team-mates present at the same time whose x is not
        missing (for text, not empty): their sum or mean (NaN when there are
        none), or how many."""
        if self._mates is None:
            self._mates = self._build_mates()
        out = []
        for others in self._mates:
            if fn == "count" and x.text is not None:
                out.append(float(sum(1 for m in others if m >= 0 and x.text[m].lower() not in self._MISSING_TEXT)))
                continue
            vs = [x.v[m] for m in others if m >= 0 and not np.isnan(x.v[m])]
            if fn == "count":
                out.append(float(len(vs)))
            elif not vs:
                out.append(np.nan)
            else:
                out.append(sum(vs) if fn == "sum" else sum(vs) / len(vs))
        return Col(out)

    def _build_mates(self):
        if self.participant is None or self.group is None:
            raise ValueError("mate() needs the data's participant and group columns")
        gc = self._index(self.group, "groups")
        tc = None if self.time is None else self._index(self.time, "time")
        rows = self.table["rows"]
        group_of = [r[gc].strip() for r in rows]
        members, at, ordinal, keys = {}, {}, {}, []
        for r in range(self.n):
            p, g = self.participant_of[r], group_of[r]
            m = members.setdefault(g, [])
            if p not in m:
                m.append(p)
            if tc is None:
                ordinal[(p, g)] = ordinal.get((p, g), -1) + 1
                key = ordinal[(p, g)]
            else:
                x = number(self.table, rows[r][tc])
                key = rows[r][tc].strip() if x is None or np.isnan(x) else x
            keys.append(key)
            at[(p, g, key)] = r
        # Per row, the rows of the other members of its group at the same
        # time, in the order the members first appear (-1: absent); an empty
        # group cell has none.
        return [
            []
            if group_of[r].lower() in self._MISSING_TEXT
            else [at.get((q, group_of[r], keys[r]), -1) for q in members[group_of[r]] if q != self.participant_of[r]]
            for r in range(self.n)
        ]

    def eq(self, x, s, equal=True):
        if x.text is None:
            raise ValueError(f"Compare '{s}' with a column of the data (or a lag/mate of one)")
        return Col(
            [np.nan if t.lower() in self._MISSING_TEXT else float((t == s) == equal) for t in x.text]
        )

    def missing(self, x):
        if x.text is not None:
            return Col([float(t.lower() in self._MISSING_TEXT) for t in x.text])
        return Col(np.isnan(x.v).astype(float))

    def fill(self, x, y):
        return Col(np.where(np.isnan(x.v), y.v, x.v))

    def where(self, c, a, b):
        return Col(np.where(np.isnan(c.v), np.nan, np.where(c.v != 0, a.v, b.v)))
''';

  static const _outputHelpers = r'''
def trajectory(net, node, field):
    """A node's recorded field (some live in its "temp" store)."""
    traj = net.node_trajectories[node]
    return np.asarray(traj[field] if field in traj else traj["temp"][field], dtype=float)


def gaussian_node_surprise(net, node):
    """The surprise of a continuous node's mean under its prediction."""
    traj = net.node_trajectories[node]
    x, m, p = (np.asarray(traj[f], dtype=float) for f in ("mean", "expected_mean", "expected_precision"))
    return 0.5 * (np.log(2 * np.pi) - np.log(p) + p * (x - m) ** 2)


def binary_node_surprise(net, node):
    """The surprise of a binary node's value under its prediction."""
    traj = net.node_trajectories[node]
    x = np.asarray(traj["mean"], dtype=float)
    m = np.clip(np.asarray(traj["expected_mean"], dtype=float), 1e-6, 1 - 1e-6)
    return np.where(x != 0, -np.log(m), -np.log(1 - m))


def with_columns(table, columns):
    """The table as CSV text with columns appended: per participant (in
    order) a dict of column name to one value per row of theirs."""
    names = list(dict.fromkeys(n for c in columns for n in c))
    for n in names:
        if n in table["header"]:
            raise ValueError(f'The data already has a column "{n}"; rename the output')
    rows = [list(r) + [""] * len(names) for r in table["rows"]]
    base = len(table["header"])
    for cols, idx in zip(columns, participant_rows(table).values()):
        for j, n in enumerate(names):
            series = cols.get(n)
            if series is None:
                continue
            for t, r in enumerate(idx):
                if t < len(series) and not np.isnan(series[t]):
                    rows[r][base + j] = format_number(float(series[t]))
    return _csv_text(table["header"] + names, rows)


def _csv_text(header, rows):
    def q(c):
        return '"' + c.replace('"', '""') + '"' if any(x in c for x in ',"\n\r') else c

    return "\n".join(",".join(q(c) for c in r) for r in [header] + rows) + "\n"
''';

  static const _estimateHelpers = r'''
def estimates_csv(estimates):
    """The estimates as hgf.dart's builder writes them: a row per
    participant (each parameter, its 95 % interval), then the group level."""

    def cell(v):
        return repr(float(v)) if np.isfinite(v) else ""

    header = ["participant"] + [x for p in PARAMETERS for x in (p, p + "_lower", p + "_upper")]
    rows = [
        [pid] + [cell(e[k][p]) for p in PARAMETERS for k in ("estimate", "lower", "upper")]
        for pid, e in estimates["participants"].items()
    ]
    text = _csv_text(header, rows)
    if estimates["group"]:
        text += "\n" + _csv_text(
            ["parameter", "value", "lower", "upper"],
            [[name] + [cell(v) for v in values] for name, *values in estimates["group"]],
        )
    return text


def print_estimates(estimates):
    def fmt(v):
        return f"{v:.5g}" if np.isfinite(v) else "-"

    for p in PARAMETERS:
        print(p)
        for pid, e in estimates["participants"].items():
            lo, hi = e["lower"][p], e["upper"][p]
            iv = f"  [{fmt(lo)}, {fmt(hi)}]" if np.isfinite(lo) else ""
            print(f"  {pid:<11}  {fmt(e['estimate'][p])}{iv}")
    if estimates["group"]:
        print("group level")
        for name, v, lo, hi in estimates["group"]:
            iv = f"  [{fmt(lo)}, {fmt(hi)}]" if np.isfinite(lo) else ""
            print(f"  {name:<11}  {fmt(v)}{iv}")
''';
}

/// Python that reads CSV as hgf.dart does: `parse_csv(text)` and
/// `number(table, cell)` (NaN when missing, None for text).
const pyCsvReader = r'''
_MISSING = {"", "na", "nan", "null", "n/a", "none", "."}


def parse_csv(text):
    """A table (dict of "header" and "rows", cells as text) read as hgf.dart
    reads CSV: the separator is a tab, a semicolon or a comma (the first the
    header has; with semicolons a decimal comma reads as a number), blank and
    `#` lines are skipped, rows are padded to the header's width."""
    lines = [
        line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    first = lines[0]
    sep = "\t" if "\t" in first else ";" if ";" in first else ","
    split = [next(csv.reader([line], delimiter=sep)) for line in lines]
    header = [h.strip() for h in split[0]]
    rows = [(r + [""] * len(header))[: len(header)] for r in split[1:]]
    return dict(header=header, rows=rows, decimal_comma=sep == ";")


def number(table, cell):
    """A cell as a number: NaN when missing, None when it is text."""
    s = cell.strip()
    if s.lower() in _MISSING:
        return float("nan")
    if table["decimal_comma"]:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


''';

/// [e] as a column of the `Transforms` helper (a data expression).
String pyDataExpr(Expr e) => _PythonDocument._dataExpr(e);

/// The data pipeline of a script without an HGF network: `LAYOUT` (the
/// participant, time and group columns), the CSV and data helpers, and
/// `transform(table)` (the clock, the steps, the transforms and [layout]'s
/// deep networks).
String pyDataPipeline(DataLayout layout) {
  String opt(String? s) => s == null ? 'None' : _PythonDocument._pyStr(s);
  return """# How the CSV is read (hgf.dart's DataLayout): the participant, time and group
# columns.
LAYOUT = dict(
    inputs=[],
    responses=[],
    options=None,
    participant=${opt(layout.participantColumn)},
    time=${opt(layout.timeColumn)},
    time_step=${opt(layout.timeStepColumn)},
    group=${opt(layout.groupColumn)},
)


${_PythonDocument._dataHelpers}

${_PythonDocument._transformCode(layout)}""";
}
