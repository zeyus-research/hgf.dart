"""Phase 7 WebGPU probe in Chromium: serves DIR (the compiled
benchmark/web/gpu_bench.dart page plus inputs.json) and prints the page's
`HGF_GPU` result lines as JSON.

    melos run bench:gpu-web
    uv run --with playwright python -I tool/gpu_bench.py DIR [query]

Uses the system Chromium (CHROMIUM, default /usr/bin/chromium) with WebGPU
on Vulkan; HEADED=1 runs it in a window.
"""

import functools
import http.server
import json
import mimetypes
import os
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main(argv: list[str]) -> None:
    root = Path(argv[0])
    query = argv[1] if len(argv) > 1 else ""
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/index.html?{query}"
    args = [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPUService",
        "--use-angle=vulkan",
        "--ignore-gpu-blocklist",
        "--enable-gpu",
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"),
            headless=not os.environ.get("HEADED"),
            args=args,
        )
        page = browser.new_page()
        done = threading.Event()

        def on_console(m):
            t = m.text
            if t.startswith("HGF_GPU "):
                if '"done"' in t:
                    done.set()
            else:
                print(f"  [page] {t}", file=sys.stderr, flush=True)

        page.on("console", on_console)
        page.on("pageerror", lambda e: print(f"  [error] {e}", file=sys.stderr, flush=True))
        page.goto(url)
        for _ in range(int(os.environ.get("TIMEOUT_S", "900")) * 4):
            if done.is_set():
                break
            page.wait_for_timeout(250)
        # The page's own record; console events can be dropped.
        for line in page.evaluate("globalThis.hgfResults"):
            print(json.dumps(json.loads(line)), flush=True)
        browser.close()


if __name__ == "__main__":
    main(sys.argv[1:])
