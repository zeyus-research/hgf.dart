import 'package:hgf/hgf.dart'
    show
        FirstLevelBinarySurprise,
        FirstLevelGaussianSurprise,
        TotalGaussianSurprise;

import '../response_model.dart';
import 'trial.dart';

/// The IR response for [response] (the GPU kernels' and the CPU reverse
/// mode's), or throws [UnsupportedError].
GpuResponse irResponseOf(ResponseModel response) => switch (response) {
  GaussianReadout(:final node, :final field, :final precision) =>
    GpuGaussianReadout(node: node, field: field, precision: precision),
  SoftmaxReadout(:final options, :final field, :final inverseTemperature) =>
    GpuSoftmax(
      options: options,
      field: field,
      inverseTemperature: inverseTemperature,
    ),
  BinarySoftmaxReadout(:final node, :final inverseTemperature) =>
    GpuBinarySoftmax(node: node, inverseTemperature: inverseTemperature),
  ObservationSurprise(:final function) => switch (function) {
    TotalGaussianSurprise() => GpuResponse.totalGaussianSurprise,
    FirstLevelGaussianSurprise() => GpuResponse.firstLevelGaussianSurprise,
    FirstLevelBinarySurprise() => GpuResponse.firstLevelBinarySurprise,
    _ => throw UnsupportedError('${function.runtimeType} has no GPU kernel'),
  },
  _ => throw UnsupportedError('${response.runtimeType} has no GPU kernel'),
};
