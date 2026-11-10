import 'dart:typed_data';

import 'model.dart';

/// What a [Program] yields: a [Pass] to evaluate, or [Concurrent] streams
/// of steps.
sealed class Step {
  const Step();
}

/// Independent sequences of steps that a runner may interleave or run in
/// parallel; the program resumes once all of them have ended.
///
/// Drivers whose tasks don't interact (each subject's optimiser, each NUTS
/// chain) split them into streams when the runner is parallel, so that one
/// slow task doesn't hold every other one at each pass's barrier. Each
/// stream must only touch its own state.
final class Concurrent extends Step {
  const Concurrent(this.streams);

  final List<Iterable<Step>> streams;
}

/// One batched evaluation of [HgfModel.negLogPosterior]: its inputs, and
/// output buffers that a runner fills.
///
/// The fitting drivers (`fitMap`, `sampleNuts`, …) are written as `sync*`
/// generators that yield passes ([Program]s), the way optimisers yield
/// `EvalRequest`s. A runner answers each pass before resuming the program:
/// [runInline] on the calling isolate, `FitWorkers` by splitting the pass's
/// elements across worker isolates. Elements are independent (a batch
/// element's result doesn't depend on which others share its batch), so
/// both give bit-identical results.
final class Pass extends Step {
  Pass(
    this.model,
    this.subjects,
    this.subjectOf,
    this.y, {
    this.jacobian = false,
    this.includePrior,
    this.precise = false,
    bool negLogLikelihood = false,
    bool gradient = false,
  }) : out = Float64List(subjectOf.length),
       negLogLikelihood = negLogLikelihood
           ? Float64List(subjectOf.length)
           : null,
       gradient = gradient ? Float64List(y.length) : null;

  final HgfModel model;
  final List<Subject> subjects;
  final Int32List subjectOf;
  final Float64List y;
  final bool jacobian;
  final List<bool>? includePrior;

  /// Whether the caller needs the full `f64` accuracy of the CPU engine:
  /// the pass's values are differenced (finite-difference gradients and
  /// Hessians) or reported. A runner that evaluates in lower precision (a
  /// GPU in `f32`) answers these passes on the CPU.
  final bool precise;

  /// Filled by the runner.
  final Float64List out;
  final Float64List? negLogLikelihood;
  final Float64List? gradient;

  /// Number of elements.
  int get length => subjectOf.length;

  /// Evaluates the pass on this isolate.
  void run() => model.negLogPosterior(
    subjects,
    subjectOf,
    y,
    out,
    jacobian: jacobian,
    includePrior: includePrior,
    negLogLikelihood: negLogLikelihood,
    gradient: gradient,
  );

  /// Elements `[start, end)` as a pass of their own (over the same
  /// subjects).
  Pass slice(int start, int end) {
    final k = model.dimension;
    return Pass(
      model,
      subjects,
      Int32List.sublistView(subjectOf, start, end),
      Float64List.sublistView(y, start * k, end * k),
      jacobian: jacobian,
      includePrior: includePrior,
      precise: precise,
      negLogLikelihood: negLogLikelihood != null,
      gradient: gradient != null,
    );
  }

  /// Copies the outputs of elements `[start, end)` (from a [slice], or
  /// results of the same shape).
  void absorb(
    int start,
    int end,
    Float64List out,
    Float64List? negLogLikelihood,
    Float64List? gradient,
  ) {
    final k = model.dimension;
    this.out.setRange(start, end, out);
    if (negLogLikelihood != null) {
      this.negLogLikelihood!.setRange(start, end, negLogLikelihood);
    }
    if (gradient != null) {
      this.gradient!.setRange(start * k, end * k, gradient);
    }
  }

  /// Splits the elements into at most [parts] contiguous ranges of about
  /// equal cost (trials filtered), none cheaper than [minCost] trials
  /// unless there is only one. Returns the range boundaries, starting at 0
  /// and ending at [length].
  List<int> partition(int parts, int minCost) {
    final n = length;
    if (n == 0) return const [0];
    final nIn = model.network.nInputColumns;
    final cost = Float64List(n + 1);
    for (var e = 0; e < n; e++) {
      cost[e + 1] = cost[e] + subjects[subjectOf[e]].inputs.length / nIn;
    }
    final total = cost[n];
    final p = parts.clamp(1, n);
    final wanted = minCost <= 0 ? p : (total / minCost).floor().clamp(1, p);
    final bounds = [0];
    var e = 0;
    for (var i = 1; i < wanted; i++) {
      final target = total * i / wanted;
      while (e < n && cost[e] < target) {
        e++;
      }
      if (e > bounds.last && e < n) bounds.add(e);
    }
    bounds.add(n);
    return bounds;
  }
}

/// Answers the passes of [Program]s: `FitWorkers` (isolates, Web Workers
/// or the calling isolate), or another evaluator of [HgfModel]s, like a GPU.
/// The fitting functions run on any runner as extension methods
/// (`runner.fitMap(...)`, `runner.sampleNuts(...)`, ...).
abstract interface class PassRunner {
  /// Runs [program], answering each of its passes.
  Future<T> run<T>(Program<T> program);

  /// How many [Concurrent] streams a driver should split independent tasks
  /// into (1: every task in each pass, in lock step).
  int get streams;
}

/// Holds a program's result.
final class Out<T> {
  late T value;
}

/// A driver: yields the passes it needs answered, in order, and puts its
/// result in the [Out] when it ends.
typedef Program<T> = Iterable<Step> Function(Out<T> result);

/// Runs [program] on this isolate ([Concurrent] streams one after the
/// other).
T runInline<T>(Program<T> program) {
  final result = Out<T>();
  runStepsInline(program(result));
  return result.value;
}

/// Answers [steps] on this isolate.
void runStepsInline(Iterable<Step> steps) {
  for (final step in steps) {
    switch (step) {
      case Pass():
        step.run();
      case Concurrent(:final streams):
        for (final s in streams) {
          runStepsInline(s);
        }
    }
  }
}

/// The indices of [tasks] tasks dealt into at most [streams] groups,
/// strided (group `g` gets tasks `g, g + streams, …`): neighbouring tasks
/// tend to be alike (the same subject's chains, subjects sorted by a
/// covariate), so striding spreads the slow ones over the groups.
List<List<int>> splitTasks(int tasks, int streams) {
  final g = streams.clamp(1, tasks < 1 ? 1 : tasks);
  return [
    for (var i = 0; i < g; i++) [for (var t = i; t < tasks; t += g) t],
  ];
}
