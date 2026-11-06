"""FitWorkers on Web Workers against the page's main thread, in headless
Chromium (Phase 6): bench:parallel's jobs, with the page and its worker
compiled alike (dart2wasm -O4, dart2js -O2).

    melos run bench:web-parallel
    uv run --with playwright python -I tool/web/fit_workers_bench.py DIR [wasm js] [--draws N] [--workers 2,4,8]

DIR holds the compiled page (bench.{wasm,mjs,js}), the worker
(fit_worker.{wasm,mjs,js}), index.html and binary_input.txt. Uses the
system Chromium (CHROMIUM, default /usr/bin/chromium). Prints a Markdown
table per target: main-thread time, then the speed-up per pool size.
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
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def main(argv: list[str]) -> None:
    root = Path(argv[0])
    rest = argv[1:]
    query = {}
    targets = []
    i = 0
    while i < len(rest):
        if rest[i] == "--draws":
            query["draws"] = rest[i + 1]
            i += 2
        elif rest[i] == "--workers":
            query["workers"] = rest[i + 1]
            i += 2
        else:
            targets.append(rest[i])
            i += 1
    targets = targets or ["wasm", "js"]
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/index.html"
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium")
        )
        for target in targets:
            page = browser.new_page()
            got: list[str] = []
            errors: list[str] = []

            def on_console(m, got=got):
                if "HGF_BENCH" in m.text:
                    got.append(m.text)
                else:
                    print(f"  [{target}] {m.text}", flush=True)

            page.on("console", on_console)
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            params = "&".join(f"{k}={v}" for k, v in {"target": target, **query}.items())
            page.goto(f"{base}?{params}")
            for _ in range(3600):
                if got or errors:
                    break
                page.wait_for_timeout(1000)
            if errors or not got:
                print(f"{target}: failed {errors}")
                continue
            r = json.loads(got[0].split(" ", 1)[1])
            counts = r["workers"]
            print(f"\n{r['target']} (page and workers), draws {r['draws']}, "
                  f"bit-identical: {r['identical']}\n")
            print("| job | main thread | " + " | ".join(f"{c} workers" for c in counts[1:]) + " |")
            print("|---|---|" + "---|" * (len(counts) - 1))
            for row in r["rows"]:
                ms = row["ms"]
                main_ms = ms[0]
                unit = f"{main_ms / 1000:.2f} s" if main_ms >= 1000 else f"{main_ms:.1f} ms"
                print(f"| {row['job']} | {unit} | " +
                      " | ".join(f"{main_ms / t:.2f}×" for t in ms[1:]) + " |")
            page.close()
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main(sys.argv[1:])
