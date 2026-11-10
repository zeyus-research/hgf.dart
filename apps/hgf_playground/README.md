# HGF playground

An interactive playground for Hierarchical Gaussian Filters, built on the
[`hgf`](../../packages/hgf) engine and [`hgf_widgets`](../../packages/hgf_widgets).

- Pick a network (binary 3-level, continuous 2- or 3-level HGF), tap a node
  and drag its parameters: the whole series is filtered again every frame.
- Simulated data (reversal learning, or a random walk with calm and volatile
  phases) with the hidden truth overlaid, or load your own CSV (one column;
  `NaN` marks a missing observation).
- Play the network trial by trial (`predict` / `step`, as in a real-time
  experiment) and watch the beliefs and the graph's captions update.
- Fit the parameters ([`hgf_fit`](../../packages/hgf_fit)): to the
  observations (least surprise), or to responses, simulated from an agent
  with the current sliders (parameter recovery: the truth is marked) or
  loaded from a CSV. MAP with Laplace intervals in tens of milliseconds, or
  NUTS with progress and cancel; each parameter's prior and
  posterior are drawn, and *Apply* moves the sliders to the estimates.
  Fits run off the UI thread: in a host isolate on desktop, and on the web
  in a host Web Worker (`fit_host`) whose pool is nested Web Workers
  (`fit_worker`). Without `fit_host` they run from the page (passes still
  on the workers); without `fit_worker` on the main thread between frames.
- Fits can run on the GPU ([`hgf_gpu`](../../packages/hgf_gpu)): WebGPU in
  the browser (Chrome, Edge; Firefox with WebGPU enabled, where the adapter
  shows without a name), wgpu-native on desktop and mobile (bundled into the app
  by `hgf_gpu`'s build hook, or set `WGPU_LIB` to override it; it runs in a background
  isolate). The GPU
  pays off for NUTS with many chains; *Auto* uses it from 32 chains.
- Settings (the *tune* button, saved between sessions): theme, device
  (Auto / CPU f64 / GPU f32), the number of workers (default: one per
  logical processor), MAP restarts, and NUTS chains, warm-up, draws,
  target acceptance and maximum tree depth.
- The *reset* button puts the current preset back as it ships:
  parameters, couplings, update scheme and data settings.
- Light and dark themes, following the system unless you pick one.

```sh
tool/build_worker.sh                          # the fit host and worker, into web/ (once)
flutter run -d chrome --wasm                  # the primary target
flutter build web --wasm                      # or: melos run build:playground
flutter run -d linux                          # desktop / mobile work too
```

Frame-time benchmark: `melos run bench:playground` from the repository root
(open `?bench=10000` on a web build, or pass `--dart-define=HGF_BENCH=10000`
on other platforms; it prints an `HGF_BENCH {json}` line).

Fitting benchmark: `melos run bench:playground-fit` (opens `?fitbench`,
`?fitbench&fithost=0` and `?fitbench&workers=0`; prints each fit's time, the frame intervals
while it ran, and the estimates against the simulated agent's truth).
`&device=gpu|cpu|auto&chains=N` pick the device and chains (with `GPU=1`
in the environment Chromium uses the real GPU), `&idle` stops forcing a
frame every vsync, `&report=URL` also POSTs the result there (for
browsers whose console a harness can't read). On desktop: `--dart-define=HGF_FITBENCH=1` (or `idle`),
`HGF_FIT_DEVICE`, `HGF_FIT_CHAINS`.
