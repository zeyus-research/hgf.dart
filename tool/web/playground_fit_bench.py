"""Fitting in the playground's web build (Phase 6 fitting view).

Serves apps/hgf_playground/build/web like playground_bench.py, opens
`?fitbench` (fits on the Web Workers) and `?fitbench&workers=0` (on the
main thread, a slice per frame) in headless Chromium, and prints each fit's
wall time, the frame intervals while it ran, and the estimates against the
simulated agent's truth.

    melos run build:playground
    uv run --with playwright python -I tool/web/playground_fit_bench.py

With GPU=1, Chromium gets WebGPU on the machine's GPU (Vulkan), for
queries like `fitbench&device=gpu&chains=64`; without it, WebGPU falls back
to SwiftShader.
"""

import json
import os
import sys

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playground_bench import ROOT, Handler  # noqa: E402

import functools  # noqa: E402
import http.server  # noqa: E402
import threading  # noqa: E402


def main(queries: list[str]) -> None:
    if not (ROOT / "index.html").exists():
        sys.exit(f"No build at {ROOT}: run `melos run build:playground` first")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(ROOT))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"),
            args=[
                "--enable-unsafe-webgpu",
                "--enable-features=Vulkan,WebGPUService",
                "--use-angle=vulkan",
                "--ignore-gpu-blocklist",
                "--enable-gpu",
            ]
            if os.environ.get("GPU")
            else ["--enable-unsafe-swiftshader"],
        )
        for q in queries:
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            got: list[str] = []
            logs: list[str] = []
            page.on(
                "console",
                lambda m: (got if "HGF_FITBENCH" in m.text else logs).append(m.text),
            )
            page.goto(f"{url}?{q}")
            for _ in range(600):
                if got:
                    break
                page.wait_for_timeout(1000)  # keeps Playwright's event loop running
            if not got:
                print(f"?{q}: no result")
                for line in logs[-10:]:
                    print("   ", line)
                continue
            r = json.loads(got[0].split(" ", 1)[1])
            print(f"?{q} ({r['target']})")
            for run in r["runs"]:
                if run["error"]:
                    print(f"  {run['preset']:12} {run['method']:5} error: {run['error']}")
                    continue
                print(
                    f"  {run['preset']:12} {run['method']:5} {run['ms']:>6} ms on {run['where']}; "
                    f"frames during the fit: {run['frames']}, median "
                    f"{run['frame_median_ms']:.1f} ms, max {run['frame_max_ms']:.1f} ms"
                )
                for k, (est, lo, hi, truth) in run["estimates"].items():
                    t = "" if truth is None else f"  (true {truth:.3g})"
                    print(f"      {k:20} {est:9.3g}  [{lo:.3g}, {hi:.3g}]{t}")
            page.close()
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:] or ["fitbench", "fitbench&workers=0"])
