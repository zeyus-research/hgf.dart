import 'dart:typed_data';

import 'package:hgf/hgf.dart';

import 'wgsl.dart';

/// A GPU buffer owned by a [GpuBackend].
abstract interface class GpuBuffer {
  int get bytes;
}

/// A compiled compute pipeline owned by a [GpuBackend].
abstract interface class GpuPipeline {}

/// The few WebGPU operations [GpuBatch] needs, so the same host code runs on
/// the browser's WebGPU and on a native implementation.
///
/// Writes, copies and dispatches run in the order they are queued.
abstract interface class GpuBackend {
  /// Adapter / driver description, for reports.
  String get description;

  /// Largest storage buffer binding, in bytes.
  int get maxStorageBufferBytes;

  /// Compiles [wgsl] (entry point `main`). Throws with the compiler's
  /// messages if it doesn't compile.
  Future<GpuPipeline> pipeline(String wgsl);

  /// A storage buffer (also a copy source and destination).
  GpuBuffer storage(int bytes);

  /// A uniform buffer.
  GpuBuffer uniform(int bytes);

  /// A host array that [write] uploads without an extra copy (JS-backed on
  /// the web, where copying Dart typed data to JS goes element by element).
  Float32List hostFloats(int length);

  /// Queues a write of `data[start, start + length)` to [buffer] at [offset]
  /// bytes.
  void write(
    GpuBuffer buffer,
    Float32List data, {
    int offset = 0,
    int start = 0,
    int? length,
  });

  /// Queues a write of [data] (bit patterns, e.g. indices, or floats
  /// through a `Float32List` view) to [buffer] at [offset] bytes.
  void writeUints(GpuBuffer buffer, Uint32List data, {int offset = 0});

  /// Queues a buffer-to-buffer copy.
  void copy(
    GpuBuffer source,
    int sourceOffset,
    GpuBuffer destination,
    int destinationOffset,
    int bytes,
  );

  /// Queues one dispatch of [pipeline] over [groups] workgroups, with
  /// [bindings] at bindings 0, 1, 2... of group 0.
  void dispatch(GpuPipeline pipeline, List<GpuBuffer> bindings, int groups);

  /// Reads [bytes] bytes of [buffer] from [offset], after all queued work.
  Future<Float32List> read(
    GpuBuffer buffer, {
    int offset = 0,
    required int bytes,
  });

  /// Waits until all queued work has finished.
  Future<void> idle();

  void release(GpuBuffer buffer);
}

/// At or above: the kernel's non-finite marker (3.4e38 in `f32`).
const double _f32Max = 3.3e38;

/// Wall-clock breakdown of one [GpuBatch.surprise] call, in milliseconds.
/// `upload` is host-side preparation and queueing; `gpu` runs from there
/// until the results are mapped (transfers, compute and the round trip);
/// `readback` copies them out.
typedef GpuTimings = ({
  double upload,
  double gpu,
  double readback,
  double total,
  int dispatches,
});

/// [batchSize] copies of a network on the GPU, like `BatchNetwork`: every
/// element has its own parameters and initial beliefs, and one call filters
/// them all, returning each element's total response (e.g. the negative log
/// likelihood).
///
/// The initial state lives on the GPU next to the working state; each call
/// resets the working state with a GPU-side copy, so between calls only the
/// slots changed with [setInitial] and friends cross the bus, and the data
/// only when it is a different list.
final class GpuBatch {
  GpuBatch._(this.backend, this.kernel, this.batchSize, this._pipeline)
    : _initial = backend.hostFloats(kernel.nSlots * batchSize),
      _initialBuffer = backend.storage(kernel.nSlots * batchSize * 4),
      _stateBuffer = backend.storage(kernel.nSlots * batchSize * 4),
      _paramsBuffer = backend.uniform(16) {
    for (var s = 0; s < kernel.nSlots; s++) {
      _initial.fillRange(s * batchSize, (s + 1) * batchSize, kernel.initial[s]);
    }
    backend.write(_initialBuffer, _initial);
  }

  /// Compiles [kernel] on [backend] for [batchSize] elements.
  static Future<GpuBatch> create(
    GpuBackend backend,
    HgfKernel kernel,
    int batchSize,
  ) async {
    final bytes = kernel.nSlots * batchSize * 4;
    if (bytes > backend.maxStorageBufferBytes) {
      throw ArgumentError(
        'State for $batchSize elements is $bytes bytes; the device allows '
        '${backend.maxStorageBufferBytes}',
      );
    }
    final pipeline = await backend.pipeline(kernel.source);
    return GpuBatch._(backend, kernel, batchSize, pipeline);
  }

  final GpuBackend backend;
  final HgfKernel kernel;
  final int batchSize;
  final GpuPipeline _pipeline;

  /// Host copy of the initial state, `[slot][element]`.
  final Float32List _initial;
  final GpuBuffer _initialBuffer, _stateBuffer, _paramsBuffer;
  final _dirty = <int>{};

  GpuBuffer? _dataBuffer, _maskBuffer, _stepsBuffer, _respBuffer;
  Object? _lastData, _lastMask, _lastSteps, _lastResp;

  /// Timings of the last [surprise] call.
  GpuTimings? lastTimings;

  void _setSlot(int slot, List<double> values) {
    if (values.length != batchSize) {
      throw ArgumentError('Need one value per element ($batchSize)');
    }
    final base = slot * batchSize;
    for (var e = 0; e < batchSize; e++) {
      _initial[base + e] = values[e];
    }
    _dirty.add(slot);
  }

  /// Initial value of [field] of [node], per element.
  void setInitial(int node, NodeField field, List<double> values) =>
      _setSlot(kernel.slot(node, field), values);

  void setValueCoupling(int parent, int child, List<double> values) =>
      _setSlot(kernel.valueCouplingSlot(parent, child), values);

  void setVolatilityCoupling(int parent, int child, List<double> values) =>
      _setSlot(kernel.volatilityCouplingSlot(parent, child), values);

  /// The response's parameter [j] ([GpuResponse.defaults] order), per
  /// element.
  void setResponseParameter(int j, List<double> values) =>
      _setSlot(kernel.responseParameterSlot(j), values);

  /// Filters [data] (one row of [HgfKernel.nInputs] values per trial,
  /// shared by every element, or `[trial][input][element]` with
  /// [HgfKernel.perElementData]) from the initial state and returns each
  /// element's total response; infinite where the element's state stopped
  /// being finite.
  ///
  /// [observed] (0/1, the layout of [data]) is required by kernels compiled
  /// with [HgfKernel.observedMask]; [timeSteps] (one per trial) by those
  /// compiled with [HgfKernel.timeSteps], which ignore [timeStep];
  /// [responses] (one per trial, or `[trial][element]` with
  /// [HgfKernel.perElementData]; NaN for a missed response) by those whose
  /// response scores responses.
  ///
  /// The lists are uploaded again only if they are not those of the
  /// previous call (don't modify them in place between calls).
  ///
  /// [trialsPerDispatch] bounds the work per dispatch (browsers reset the
  /// GPU on long-running ones); 0 runs the whole series in one.
  Future<Float64List> surprise(
    List<double> data, {
    List<double>? observed,
    List<double>? timeSteps,
    List<double>? responses,
    double timeStep = 1.0,
    int trialsPerDispatch = 0,
  }) async {
    if (kernel.response.usesResponses != (responses != null)) {
      throw ArgumentError(
        kernel.response.usesResponses
            ? 'This kernel needs responses'
            : 'This kernel scores no responses',
      );
    }
    if (kernel.observedMask != (observed != null)) {
      throw ArgumentError(
        kernel.observedMask
            ? 'This kernel needs observed'
            : 'This kernel was compiled without observedMask',
      );
    }
    if (kernel.timeSteps != (timeSteps != null)) {
      throw ArgumentError(
        kernel.timeSteps
            ? 'This kernel needs timeSteps'
            : 'This kernel was compiled without timeSteps',
      );
    }
    final b = batchSize;
    final width = kernel.nInputs * (kernel.perElementData ? b : 1);
    if (data.length % width != 0) {
      throw ArgumentError('data length must be a multiple of $width');
    }
    final nTime = data.length ~/ width;
    final sw = Stopwatch()..start();

    if (observed != null && observed.length != data.length) {
      throw ArgumentError('observed must have the shape of data');
    }
    if (timeSteps != null && timeSteps.length != nTime) {
      throw ArgumentError('timeSteps needs one entry per trial ($nTime)');
    }
    final respWidth = kernel.perElementData ? b : 1;
    if (responses != null && responses.length < nTime * respWidth) {
      throw ArgumentError('responses needs ${nTime * respWidth} entries');
    }
    if (!identical(data, _lastData)) {
      _dataBuffer = _upload(_dataBuffer, data);
      _lastData = data;
    }
    if (observed != null && !identical(observed, _lastMask)) {
      _maskBuffer = _upload(_maskBuffer, observed);
      _lastMask = observed;
    }
    if (timeSteps != null && !identical(timeSteps, _lastSteps)) {
      _stepsBuffer = _upload(_stepsBuffer, timeSteps);
      _lastSteps = timeSteps;
    }
    if (responses != null && !identical(responses, _lastResp)) {
      _respBuffer = _upload(_respBuffer, responses);
      _lastResp = responses;
    }
    for (final s in _dirty) {
      backend.write(
        _initialBuffer,
        _initial,
        offset: s * b * 4,
        start: s * b,
        length: b,
      );
    }
    _dirty.clear();
    backend.copy(_initialBuffer, 0, _stateBuffer, 0, kernel.nSlots * b * 4);
    final upload = sw.elapsedMicroseconds / 1e3;

    final groups = (b + kernel.workgroupSize - 1) ~/ kernel.workgroupSize;
    final chunk = trialsPerDispatch <= 0 ? nTime : trialsPerDispatch;
    var dispatches = 0;
    final params = Uint32List(4);
    final paramsF = Float32List.view(params.buffer);
    for (var t0 = 0; t0 < nTime; t0 += chunk) {
      final nT = t0 + chunk < nTime ? chunk : nTime - t0;
      params
        ..[0] = b
        ..[1] = t0
        ..[2] = nT;
      paramsF[3] = timeStep;
      backend
        ..write(_paramsBuffer, paramsF)
        ..dispatch(_pipeline, [
          _stateBuffer,
          _dataBuffer!,
          _paramsBuffer,
          ?_maskBuffer,
          ?_stepsBuffer,
          ?_respBuffer,
        ], groups);
      dispatches++;
    }
    // No idle(): mapping the results waits for the work, and each wait is a
    // round trip to the GPU process (~2.5 ms in Chromium).
    final raw = await backend.read(
      _stateBuffer,
      offset: kernel.resultSlot * b * 4,
      bytes: b * 4,
    );
    final mapped = sw.elapsedMicroseconds / 1e3;
    final out = Float64List(b);
    for (var e = 0; e < b; e++) {
      final x = raw[e];
      out[e] = x >= _f32Max ? double.infinity : x;
    }
    final total = sw.elapsedMicroseconds / 1e3;
    lastTimings = (
      upload: upload,
      gpu: mapped - upload,
      readback: total - mapped,
      total: total,
      dispatches: dispatches,
    );
    return out;
  }

  /// [values] in [buffer] (reallocated if it is too small).
  GpuBuffer _upload(GpuBuffer? buffer, List<double> values) {
    final bytes = values.length * 4;
    if (buffer == null || buffer.bytes < bytes) {
      if (buffer != null) backend.release(buffer);
      buffer = backend.storage(bytes < 16 ? 16 : bytes);
    }
    final host = backend.hostFloats(values.length);
    for (var k = 0; k < values.length; k++) {
      host[k] = values[k];
    }
    backend.write(buffer, host);
    return buffer;
  }

  /// The final value of [field] of [node] per element, after [surprise].
  /// Only the fields a dispatch stores are current: the means and
  /// precisions (expected and posterior) of every node.
  Future<Float64List> read(int node, NodeField field) async {
    final raw = await backend.read(
      _stateBuffer,
      offset: kernel.slot(node, field) * batchSize * 4,
      bytes: batchSize * 4,
    );
    return Float64List.fromList(raw);
  }

  void dispose() {
    backend
      ..release(_initialBuffer)
      ..release(_stateBuffer)
      ..release(_paramsBuffer);
    for (final b in [_dataBuffer, _maskBuffer, _stepsBuffer, _respBuffer]) {
      if (b != null) backend.release(b);
    }
  }
}
