"""Frame-time benchmark of the playground's web build (Phase 5).

Serves apps/hgf_playground/build/web (cross-origin isolated, as a Wasm
deployment should be), opens `?bench=N` in headless Chromium for each N, and
prints the app's `HGF_BENCH {json}` line: a slider dragged for 300 frames,
each re-filtering all N trials and redrawing every plot.

    melos run build:playground
    uv run --with playwright python -I tool/web/playground_bench.py 300 1000 10000

Headless Chromium rasterises in software (SwiftShader), so frame intervals
are an upper bound; the filter times are what the engine costs. Uses the
system Chromium (CHROMIUM, default /usr/bin/chromium).
"""

import functools
import http.server
import json
import os
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2] / "apps" / "hgf_playground" / "build" / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main(trials: list[str]) -> None:
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
            args=["--enable-unsafe-swiftshader"],
        )
        for n in trials:
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            got: list[str] = []
            page.on("console", lambda m: got.append(m.text) if "HGF_BENCH" in m.text else None)
            page.goto(f"{url}?bench={n}")
            for _ in range(180):
                if got:
                    break
                page.wait_for_timeout(1000)  # keeps Playwright's event loop running
            if not got:
                print(f"{n} trials: no result")
                continue
            r = json.loads(got[0].split(" ", 1)[1])
            print(
                f"{r['target']} {r['mode']}, {r['trials']:>6} trials: "
                f"filter median {r['filter_ms']['median']:.2f} ms (p95 {r['filter_ms']['p95']:.2f}), "
                f"frame interval median {r['frame_interval_ms']['median']:.1f} ms "
                f"(p95 {r['frame_interval_ms']['p95']:.1f})"
            )
            page.close()
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:] or ["300", "1000", "10000"])
