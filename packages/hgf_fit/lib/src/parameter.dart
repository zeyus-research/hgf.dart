import 'package:hgf/hgf.dart';

import 'prior.dart';
import 'transform.dart';

/// Where a free parameter's value goes.
sealed class ParameterTarget {
  const ParameterTarget();
}

/// The initial value of [field] on [node] (a node parameter such as
/// `tonicVolatility`, or an initial belief such as `mean`).
final class NodeTarget extends ParameterTarget {
  const NodeTarget(this.node, this.field);

  final int node;
  final NodeField field;

  @override
  String toString() => 'node $node ${field.name}';
}

/// The value coupling strength from [parent] to [child].
final class ValueCouplingTarget extends ParameterTarget {
  const ValueCouplingTarget(this.parent, this.child);

  final int parent;
  final int child;

  @override
  String toString() => 'value coupling $parent→$child';
}

/// The volatility coupling strength from [parent] to [child].
final class VolatilityCouplingTarget extends ParameterTarget {
  const VolatilityCouplingTarget(this.parent, this.child);

  final int parent;
  final int child;

  @override
  String toString() => 'volatility coupling $parent→$child';
}

/// A parameter of the response model, by its name in
/// `ResponseModel.parameters`.
final class ResponseTarget extends ParameterTarget {
  const ResponseTarget(this.name);

  final String name;

  @override
  String toString() => 'response $name';
}

/// A parameter estimated per subject: a [prior] on its natural scale, the
/// [targets] it sets, and the [transform] to the unconstrained scale the
/// optimisers work on (by default the one matching the prior's support).
///
/// ```dart
/// final omega2 = Parameter.node(
///   'omega2', 1, NodeField.tonicVolatility,
///   prior: TruncatedNormal(-3, 5, lower: -8, upper: 2),
/// );
/// final beta = Parameter.response('inverseTemperature',
///     prior: const Uniform(0.2, 20));
/// ```
final class Parameter {
  /// A parameter that sets every one of [targets] to the same value (e.g.
  /// one tonic volatility shared by several nodes).
  Parameter(
    this.name, {
    required List<ParameterTarget> targets,
    required this.prior,
    Transform? transform,
    this.initial,
  }) : targets = List.unmodifiable(targets),
       transform = transform ?? prior.defaultTransform {
    if (targets.isEmpty) {
      throw ArgumentError.value(targets, 'targets', 'Must not be empty');
    }
    if (initial case final x? when !prior.contains(x)) {
      throw ArgumentError.value(x, 'initial', 'Outside the prior support');
    }
  }

  /// The initial value of [field] on [node].
  Parameter.node(
    String name,
    int node,
    NodeField field, {
    required Prior prior,
    Transform? transform,
    double? initial,
  }) : this(
         name,
         targets: [NodeTarget(node, field)],
         prior: prior,
         transform: transform,
         initial: initial,
       );

  /// The value coupling strength from [parent] to [child].
  Parameter.valueCoupling(
    String name,
    int parent,
    int child, {
    required Prior prior,
    Transform? transform,
    double? initial,
  }) : this(
         name,
         targets: [ValueCouplingTarget(parent, child)],
         prior: prior,
         transform: transform,
         initial: initial,
       );

  /// The volatility coupling strength from [parent] to [child].
  Parameter.volatilityCoupling(
    String name,
    int parent,
    int child, {
    required Prior prior,
    Transform? transform,
    double? initial,
  }) : this(
         name,
         targets: [VolatilityCouplingTarget(parent, child)],
         prior: prior,
         transform: transform,
         initial: initial,
       );

  /// The response model's parameter [responseName] (by default [name]).
  Parameter.response(
    String name, {
    String? responseName,
    required Prior prior,
    Transform? transform,
    double? initial,
  }) : this(
         name,
         targets: [ResponseTarget(responseName ?? name)],
         prior: prior,
         transform: transform,
         initial: initial,
       );

  final String name;
  final List<ParameterTarget> targets;
  final Prior prior;
  final Transform transform;

  /// The starting value for optimisation (default: the prior's typical
  /// value).
  final double? initial;

  /// [initial], or the prior's typical value.
  double get start => initial ?? prior.typical;

  @override
  String toString() => 'Parameter($name ~ $prior → ${targets.join(', ')})';
}
