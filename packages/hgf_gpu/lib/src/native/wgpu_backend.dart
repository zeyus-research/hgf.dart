import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../batch.dart';
import 'wgpu_bindings.g.dart';

const _allowProcessEvents = 2; // WGPUCallbackMode_AllowProcessEvents
const _sTypeShaderSourceWgsl = 2; // WGPUSType_ShaderSourceWGSL
const _highPerformance = 2; // WGPUPowerPreference_HighPerformance
const _errorTypeNoError = 1;
const _success = 1; // *Status_Success

final class _NativeBuffer implements GpuBuffer {
  _NativeBuffer(this.handle, this.bytes);
  final WGPUBuffer handle;
  @override
  final int bytes;
}

final class _NativePipeline implements GpuPipeline {
  _NativePipeline(this.pipeline, this.layout);
  final WGPUComputePipeline pipeline;
  final WGPUBindGroupLayout layout;
}

String _str(WGPUStringView v) => v.data == nullptr || v.length == 0
    ? ''
    : utf8.decode(
        v.data.cast<Uint8>().asTypedList(
          // WGPU_STRLEN: null-terminated.
          v.length == -1 ? v.data.cast<Utf8>().length : v.length,
        ),
        allowMalformed: true,
      );

/// [GpuBackend] on wgpu-native (Vulkan, Metal, D3D12) through `dart:ffi`:
/// the same WGSL kernels as in the browser, on desktop.
///
/// Every call completes before it returns (the futures are already
/// complete): callbacks fire inside `wgpuInstanceProcessEvents` on this
/// thread, and reads block in `wgpuDevicePoll`.
final class WgpuBackend implements GpuBackend {
  WgpuBackend._(
    this._w,
    this._instance,
    this._device,
    this._queue,
    this.description,
    this.maxStorageBufferBytes,
  );

  /// Opens [library] (default: `$WGPU_LIB`, else `libwgpu_native.so`
  /// / `.dylib` / `wgpu_native.dll` on the library path) and its
  /// high-performance adapter. Returns null if there is no adapter.
  static WgpuBackend? create({String? library}) {
    final path =
        library ??
        Platform.environment['WGPU_LIB'] ??
        (Platform.isMacOS
            ? 'libwgpu_native.dylib'
            : Platform.isWindows
            ? 'wgpu_native.dll'
            : 'libwgpu_native.so');
    final w = WgpuBindings(DynamicLibrary.open(path));
    final instance = w.wgpuCreateInstance(nullptr);

    WGPUAdapter adapter = nullptr;
    String? failure;
    final onAdapter =
        NativeCallable<WGPURequestAdapterCallbackFunction>.isolateLocal((
          int status,
          WGPUAdapter a,
          WGPUStringView message,
          Pointer<Void> u1,
          Pointer<Void> u2,
        ) {
          if (status == _success) {
            adapter = a;
          } else {
            failure = _str(message);
          }
        });
    using((arena) {
      final options = arena<WGPURequestAdapterOptions>()
        ..ref.powerPreferenceAsInt = _highPerformance;
      final info = arena<WGPURequestAdapterCallbackInfo>()
        ..ref.modeAsInt = _allowProcessEvents
        ..ref.callback = onAdapter.nativeFunction;
      w.wgpuInstanceRequestAdapter(instance, options, info.ref);
      w.wgpuInstanceProcessEvents(instance);
    });
    onAdapter.close();
    if (adapter == nullptr) {
      if (failure != null) stderr.writeln('wgpu: no adapter: $failure');
      return null;
    }

    final description = using((arena) {
      final info = arena<WGPUAdapterInfo>();
      w.wgpuAdapterGetInfo(adapter, info);
      final d = [
        _str(info.ref.vendor),
        _str(info.ref.architecture),
        _str(info.ref.device),
        _str(info.ref.description),
      ].where((s) => s.isNotEmpty).join(' / ');
      w.wgpuAdapterInfoFreeMembers(info.ref);
      return d;
    });

    WGPUDevice device = nullptr;
    final onDevice =
        NativeCallable<WGPURequestDeviceCallbackFunction>.isolateLocal((
          int status,
          WGPUDevice d,
          WGPUStringView message,
          Pointer<Void> u1,
          Pointer<Void> u2,
        ) {
          if (status == _success) {
            device = d;
          } else {
            failure = _str(message);
          }
        });
    var maxBytes = 0;
    using((arena) {
      // Ask for the adapter's own limits (the defaults cap storage
      // bindings at 128 MiB).
      final limits = arena<WGPULimits>();
      w.wgpuAdapterGetLimits(adapter, limits);
      final m = limits.ref.maxStorageBufferBindingSize;
      final b = limits.ref.maxBufferSize;
      maxBytes = m < b ? m : b;
      final descriptor = arena<WGPUDeviceDescriptor>()
        ..ref.requiredLimits = limits;
      final info = arena<WGPURequestDeviceCallbackInfo>()
        ..ref.modeAsInt = _allowProcessEvents
        ..ref.callback = onDevice.nativeFunction;
      w.wgpuAdapterRequestDevice(adapter, descriptor, info.ref);
      w.wgpuInstanceProcessEvents(instance);
    });
    onDevice.close();
    if (device == nullptr) {
      stderr.writeln('wgpu: no device: $failure');
      return null;
    }
    return WgpuBackend._(
      w,
      instance,
      device,
      w.wgpuDeviceGetQueue(device),
      'wgpu-native: $description',
      maxBytes,
    );
  }

  final WgpuBindings _w;
  final WGPUInstance _instance;
  final WGPUDevice _device;
  final WGPUQueue _queue;

  @override
  final String description;

  @override
  final int maxStorageBufferBytes;

  void _setString(Pointer<WGPUStringView> view, String s, Arena arena) {
    final bytes = utf8.encode(s);
    final p = arena<Uint8>(bytes.length);
    p.asTypedList(bytes.length).setAll(0, bytes);
    view.ref
      ..data = p.cast()
      ..length = bytes.length;
  }

  @override
  Future<GpuPipeline> pipeline(String wgsl) async {
    _w.wgpuDevicePushErrorScope(
      _device,
      WGPUErrorFilter.WGPUErrorFilter_Validation,
    );
    final (module, pipeline) = using((arena) {
      final source = arena<WGPUShaderSourceWGSL>();
      source.ref.chain.sTypeAsInt = _sTypeShaderSourceWgsl;
      _setString(
        Pointer<WGPUStringView>.fromAddress(
          source.address + sizeOf<WGPUChainedStruct>(),
        ),
        wgsl,
        arena,
      );
      final md = arena<WGPUShaderModuleDescriptor>()
        ..ref.nextInChain = source.cast();
      final module = _w.wgpuDeviceCreateShaderModule(_device, md);
      final pd = arena<WGPUComputePipelineDescriptor>();
      pd.ref.compute.module = module;
      final entry = arena<WGPUStringView>();
      _setString(entry, 'main', arena);
      pd.ref.compute.entryPoint = entry.ref;
      return (module, _w.wgpuDeviceCreateComputePipeline(_device, pd));
    });
    String? error;
    final onPop =
        NativeCallable<WGPUPopErrorScopeCallbackFunction>.isolateLocal((
          int status,
          int type,
          WGPUStringView message,
          Pointer<Void> u1,
          Pointer<Void> u2,
        ) {
          if (status == _success && type != _errorTypeNoError) {
            error = _str(message);
          }
        });
    using((arena) {
      final info = arena<WGPUPopErrorScopeCallbackInfo>()
        ..ref.modeAsInt = _allowProcessEvents
        ..ref.callback = onPop.nativeFunction;
      _w.wgpuDevicePopErrorScope(_device, info.ref);
      _w.wgpuInstanceProcessEvents(_instance);
    });
    onPop.close();
    _w.wgpuShaderModuleRelease(module);
    if (error != null) throw StateError('WGSL compilation failed:\n$error');
    return _NativePipeline(
      pipeline,
      _w.wgpuComputePipelineGetBindGroupLayout(pipeline, 0),
    );
  }

  _NativeBuffer _create(int bytes, int usage) => using((arena) {
    final d = arena<WGPUBufferDescriptor>()
      ..ref.usage = usage
      ..ref.size = bytes;
    return _NativeBuffer(_w.wgpuDeviceCreateBuffer(_device, d), bytes);
  });

  static int _align4(int n) => (n + 3) & ~3;

  @override
  GpuBuffer storage(int bytes) => _create(
    _align4(bytes),
    WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
  );

  @override
  GpuBuffer uniform(int bytes) => _create(
    _align4(bytes),
    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
  );

  @override
  Float32List hostFloats(int length) => Float32List(length);

  /// Staging memory for [write]: wgpuQueueWriteBuffer copies out of it
  /// before returning.
  Pointer<Float> _scratch = nullptr;
  int _scratchLength = 0;

  @override
  void write(
    GpuBuffer buffer,
    Float32List data, {
    int offset = 0,
    int start = 0,
    int? length,
  }) {
    final n = length ?? data.length - start;
    if (n > _scratchLength) {
      if (_scratch != nullptr) calloc.free(_scratch);
      _scratch = calloc<Float>(n);
      _scratchLength = n;
    }
    _scratch.asTypedList(n).setRange(0, n, data, start);
    _w.wgpuQueueWriteBuffer(
      _queue,
      (buffer as _NativeBuffer).handle,
      offset,
      _scratch.cast(),
      n * 4,
    );
  }

  void _submit(WGPUCommandEncoder encoder) {
    final commands = _w.wgpuCommandEncoderFinish(encoder, nullptr);
    using((arena) {
      final list = arena<WGPUCommandBuffer>()..value = commands;
      _w.wgpuQueueSubmit(_queue, 1, list);
    });
    _w.wgpuCommandBufferRelease(commands);
    _w.wgpuCommandEncoderRelease(encoder);
  }

  @override
  void copy(
    GpuBuffer source,
    int sourceOffset,
    GpuBuffer destination,
    int destinationOffset,
    int bytes,
  ) {
    final encoder = _w.wgpuDeviceCreateCommandEncoder(_device, nullptr);
    _w.wgpuCommandEncoderCopyBufferToBuffer(
      encoder,
      (source as _NativeBuffer).handle,
      sourceOffset,
      (destination as _NativeBuffer).handle,
      destinationOffset,
      bytes,
    );
    _submit(encoder);
  }

  @override
  void dispatch(GpuPipeline pipeline, List<GpuBuffer> bindings, int groups) {
    final p = pipeline as _NativePipeline;
    final group = using((arena) {
      final entries = arena<WGPUBindGroupEntry>(bindings.length);
      for (var i = 0; i < bindings.length; i++) {
        final b = bindings[i] as _NativeBuffer;
        entries[i]
          ..binding = i
          ..buffer = b.handle
          ..offset = 0
          ..size = b.bytes;
      }
      final d = arena<WGPUBindGroupDescriptor>()
        ..ref.layout = p.layout
        ..ref.entryCount = bindings.length
        ..ref.entries = entries;
      return _w.wgpuDeviceCreateBindGroup(_device, d);
    });
    final encoder = _w.wgpuDeviceCreateCommandEncoder(_device, nullptr);
    final pass = _w.wgpuCommandEncoderBeginComputePass(encoder, nullptr);
    _w
      ..wgpuComputePassEncoderSetPipeline(pass, p.pipeline)
      ..wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr)
      ..wgpuComputePassEncoderDispatchWorkgroups(pass, groups, 1, 1)
      ..wgpuComputePassEncoderEnd(pass)
      ..wgpuComputePassEncoderRelease(pass);
    _submit(encoder);
    _w.wgpuBindGroupRelease(group);
  }

  @override
  Future<Float32List> read(
    GpuBuffer buffer, {
    int offset = 0,
    required int bytes,
  }) async {
    final size = _align4(bytes);
    final staging = _create(
      size,
      WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
    );
    copy(buffer, offset, staging, 0, size);
    var status = 0;
    final onMap = NativeCallable<WGPUBufferMapCallbackFunction>.isolateLocal((
      int s,
      WGPUStringView message,
      Pointer<Void> u1,
      Pointer<Void> u2,
    ) {
      status = s;
    });
    using((arena) {
      final info = arena<WGPUBufferMapCallbackInfo>()
        ..ref.modeAsInt = _allowProcessEvents
        ..ref.callback = onMap.nativeFunction;
      _w.wgpuBufferMapAsync(
        staging.handle,
        WGPUMapMode_Read,
        0,
        size,
        info.ref,
      );
    });
    _w.wgpuDevicePoll(_device, 1, nullptr);
    _w.wgpuInstanceProcessEvents(_instance);
    onMap.close();
    if (status != _success) {
      throw StateError('wgpu: mapping failed (status $status)');
    }
    final mapped = _w.wgpuBufferGetConstMappedRange(staging.handle, 0, size);
    final out = Float32List.fromList(
      mapped.cast<Float>().asTypedList(bytes ~/ 4),
    );
    _w.wgpuBufferUnmap(staging.handle);
    release(staging);
    return out;
  }

  @override
  Future<void> idle() async {
    _w.wgpuDevicePoll(_device, 1, nullptr);
  }

  @override
  void release(GpuBuffer buffer) {
    final h = (buffer as _NativeBuffer).handle;
    _w
      ..wgpuBufferDestroy(h)
      ..wgpuBufferRelease(h);
  }
}
