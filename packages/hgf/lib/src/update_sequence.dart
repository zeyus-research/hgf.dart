part of 'network.dart';

/// Port of rshgf's `set_update_sequence` (`src/utils/set_sequence.rs`), which
/// mirrors pyhgf's `get_update_sequence`; the ef-state and categorical
/// steps follow pyhgf's JAX version (rshgf only has the fixed-rate ef one,
/// and no categorical node).
///
/// One difference from JAX: its posterior pass resets `update_fn` once per
/// pass, so a node that should get no posterior step (a binary node with a
/// categorical child) inherits the step of the continuous node before it in
/// the same pass, and crashes ([P7](../../../../docs/upstream/README.md#p7)).
/// Here only continuous nodes get posterior steps.
extension on Network {
  UpdateSequence _buildUpdateSequence() {
    return UpdateSequence(
      List.unmodifiable(_predictionSequence()),
      List.unmodifiable(_updateSweep()),
    );
  }

  /// Predictions run from the roots down: repeatedly take the first remaining
  /// node none of whose parents is still pending.
  List<(int, UpdateStep)> _predictionSequence() {
    final out = <(int, UpdateStep)>[];
    final remaining = [for (var i = 0; i < _n; i++) i];
    while (remaining.isNotEmpty) {
      var found = -1;
      for (final idx in remaining) {
        final d = _defs[idx];
        final pending =
            d.valueParents.any(remaining.contains) ||
            d.volatilityParents.any(remaining.contains);
        if (!pending) {
          found = idx;
          break;
        }
      }
      if (found < 0) break; // cycle
      switch (_defs[found].kind) {
        case NodeKind.continuousState:
          out.add((
            found,
            meanFieldUpdates
                ? UpdateStep.predictionContinuousMeanField
                : UpdateStep.predictionContinuous,
          ));
        case NodeKind.binaryState:
          out.add((found, UpdateStep.predictionBinary));
        case NodeKind.constantState:
        case NodeKind.efState:
        case NodeKind.categoricalState:
          break; // no prediction step
      }
      remaining.remove(found);
    }
    return out;
  }

  /// Prediction errors and posteriors run from the leaves up, alternating
  /// batches: posteriors of nodes whose children have all sent their
  /// prediction errors, then prediction errors of nodes whose posterior is
  /// done.
  List<(int, UpdateStep)> _updateSweep() {
    final out = <(int, UpdateStep)>[];
    final pe = [for (var i = 0; i < _n; i++) i];
    // No posterior for constant nodes, nor for the continuous nodes that
    // receive an ef-state node's sufficient statistics (as pyhgf).
    final po = [
      for (var i = 0; i < _n; i++)
        if (_defs[i].hasChildren &&
            _defs[i].kind != NodeKind.constantState &&
            !_defs[i].valueChildren.any(
              (c) => _defs[c].kind == NodeKind.efState,
            ))
          i,
    ];

    while (true) {
      var progressed = false;

      final eligiblePo = [
        for (final idx in po)
          if (!_defs[idx].valueChildren.any(pe.contains) &&
              !_defs[idx].volatilityChildren.any(pe.contains))
            idx,
      ];
      for (final idx in eligiblePo) {
        final d = _defs[idx];
        if (d.kind == NodeKind.continuousState) {
          out.add((idx, _posteriorStep(d)));
        }
        progressed = true;
      }
      po.removeWhere(eligiblePo.contains);

      final eligiblePe = [
        for (final idx in pe)
          if (!po.contains(idx)) idx,
      ];
      for (final idx in eligiblePe) {
        final d = _defs[idx];
        final hasParents =
            d.valueParents.isNotEmpty || d.volatilityParents.isNotEmpty;
        if (d.kind == NodeKind.efState) {
          // pyhgf queues the posterior first; it moves to the end below.
          if (hasParents) {
            out.add((idx, UpdateStep.posteriorEfDynamic));
            out.add((idx, UpdateStep.predictionErrorEfDynamic));
          } else {
            out.add((idx, UpdateStep.predictionErrorEfFixed));
          }
          progressed = true;
          continue;
        }
        if (d.kind == NodeKind.categoricalState) {
          // Always has its binary parents (addCategoricalState); the
          // posterior is queued first and moves to the end, as for ef.
          out.add((idx, UpdateStep.posteriorCategorical));
          out.add((idx, UpdateStep.predictionErrorCategorical));
          progressed = true;
          continue;
        }
        if (!hasParents) continue;
        switch (d.kind) {
          case NodeKind.continuousState:
            out.add((idx, UpdateStep.predictionErrorContinuous));
          case NodeKind.binaryState:
            out.add((idx, UpdateStep.predictionErrorBinary));
          case NodeKind.constantState:
          case NodeKind.efState:
          case NodeKind.categoricalState:
            break; // no parents / handled above; unreachable
        }
        progressed = true;
      }
      pe.removeWhere(eligiblePe.contains);

      if (pe.isEmpty && po.isEmpty) break;
      if (!progressed) break;
    }
    _moveDeferredSteps(out);
    return out;
  }

  /// pyhgf moves the steps that must wait for the whole sweep (the ef-state
  /// and categorical posteriors) to the end with `remove` + `append` while iterating over the
  /// same list. Python's list iterator then skips the element after each
  /// move, so with several such steps the final order isn't simply "in
  /// order of appearance". Replayed literally, to keep the same order.
  static void _moveDeferredSteps(List<(int, UpdateStep)> steps) {
    for (var k = 0; k < steps.length; k++) {
      final step = steps[k];
      if (step.$2 == UpdateStep.posteriorEfDynamic ||
          step.$2 == UpdateStep.posteriorCategorical) {
        steps
          ..removeAt(k)
          ..add(step);
      }
    }
  }

  UpdateStep _posteriorStep(_NodeDef d) {
    final mf = meanFieldUpdates;
    if (d.volatilityChildren.isNotEmpty) {
      switch (volatilityUpdates) {
        case VolatilityUpdates.eHGF:
          return mf
              ? UpdateStep.posteriorContinuousEhgfMeanField
              : UpdateStep.posteriorContinuousEhgf;
        case VolatilityUpdates.unbounded:
          return UpdateStep.posteriorContinuousUnbounded;
        case VolatilityUpdates.standard:
          break;
      }
    }
    return mf
        ? UpdateStep.posteriorContinuousMeanField
        : UpdateStep.posteriorContinuous;
  }
}
