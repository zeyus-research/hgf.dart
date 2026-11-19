import 'package:hgf/hgf.dart'
    show
        CategoricalSurprise,
        DpSurprise,
        FirstLevelBinarySurprise,
        FirstLevelGaussianSurprise,
        TotalGaussianSurprise;

import '../expr_readout.dart';
import '../response_model.dart';
import 'trial.dart';

/// The IR response for [response] (the GPU kernels' and the CPU reverse
/// mode's), or throws [UnsupportedError].
GpuResponse irResponseOf(ResponseModel response) => switch (response) {
  GaussianReadout(:final node, :final field, :final precision) =>
    GpuGaussianReadout(node: node, field: field, precision: precision),
  SoftmaxReadout(
    :final options,
    :final field,
    :final inverseTemperature,
    :final stickiness,
    :final lapse,
    :final bias,
  ) =>
    GpuSoftmax(
      options: options,
      field: field,
      inverseTemperature: inverseTemperature,
      stickiness: stickiness,
      lapse: lapse,
      bias: bias,
    ),
  ExprReadout(
    :final kind,
    :final expressions,
    :final beliefs,
    :final parameters,
  ) =>
    GpuExprReadout(
      kind: kind.name,
      expressions: expressions,
      beliefs: beliefs,
      parameters: parameters,
    ),
  BinarySoftmaxReadout(:final node, :final inverseTemperature) =>
    GpuBinarySoftmax(node: node, inverseTemperature: inverseTemperature),
  CategoricalSoftmaxReadout(:final node, :final inverseTemperature) =>
    GpuCategoricalSoftmax(node: node, inverseTemperature: inverseTemperature),
  ObservationSurprise(:final function) => switch (function) {
    TotalGaussianSurprise() => GpuResponse.totalGaussianSurprise,
    FirstLevelGaussianSurprise() => GpuResponse.firstLevelGaussianSurprise,
    FirstLevelBinarySurprise() => GpuResponse.firstLevelBinarySurprise,
    CategoricalSurprise() => GpuResponse.categoricalSurprise,
    DpSurprise() => GpuResponse.dpSurprise,
    _ => throw UnsupportedError('${function.runtimeType} has no GPU kernel'),
  },
  _ => throw UnsupportedError('${response.runtimeType} has no GPU kernel'),
};
