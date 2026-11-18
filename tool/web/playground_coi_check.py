"""Cross-origin isolation of the playground without server headers.

GitHub Pages cannot send COOP/COEP, so the playground's coi_sw.js (registered
by its custom flutter_bootstrap.js) adds them. This serves
apps/hgf_playground/build/web with a plain static server, like Pages, and
checks in headless Chromium that:

- the first visit reloads once and comes back `crossOriginIsolated` (skwasm
  renders on its worker);
- a second visit does not reload;
- a fit (`?fitbench`) runs on the fit host and its nested Web Workers;
- `?coi=0` in a fresh profile loads without isolation, as before.

    melos run build:playground
    uv run --with playwright python -I tool/web/playground_coi_check.py

Uses the system Chromium (CHROMIUM, default /usr/bin/chromium).
"""

import functools
import http.server
import json
import os
import sys
import threading

from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from playground_bench import ROOT  # noqa: E402


class PlainHandler(http.server.SimpleHTTPRequestHandler):
    """No COOP/COEP: what GitHub Pages serves."""

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):
        pass


def visit(context, url: str, wait_for: str | None = None) -> dict:
    page = context.new_page()
    navigations, found = [], []
    page.on(
        "framenavigated",
        lambda f: navigations.append(f.url) if f == page.main_frame else None,
    )
    if wait_for:
        page.on("console", lambda m: found.append(m.text) if wait_for in m.text else None)
    page.goto(url)
    page.wait_for_selector("flutter-view", state="attached", timeout=60_000)
    for _ in range(240 if wait_for else 4):
        if wait_for and found:
            break
        page.wait_for_timeout(500)
    state = page.evaluate(
        """() => ({
            isolated: self.crossOriginIsolated,
            controller: navigator.serviceWorker.controller?.scriptURL ?? null,
        })"""
    )
    # Same-document navigations (the app's routes) count too; compare visits.
    state["navigations"] = len([u for u in navigations if "#" not in u])
    state["workers"] = len(page.workers)
    state["found"] = found[0] if found else None
    page.close()
    return state


def main() -> None:
    if not (ROOT / "index.html").exists():
        sys.exit(f"No build at {ROOT}: run `melos run build:playground` first")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(PlainHandler, directory=str(ROOT))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    failures = []

    def check(name: str, ok: bool, detail: object) -> None:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if not ok:
            failures.append(name)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=os.environ.get("CHROMIUM", "/usr/bin/chromium"),
            args=["--enable-unsafe-swiftshader"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        first = visit(context, url)
        check("first visit isolated", first["isolated"], first)
        check("skwasm render worker", first["workers"] >= 1, first["workers"])
        second = visit(context, url)
        check("second visit isolated", second["isolated"], second)
        check(
            "second visit does not reload",
            second["navigations"] < first["navigations"],
            (first["navigations"], second["navigations"]),
        )
        fit = visit(context, url + "?fitbench", wait_for="HGF_FITBENCH")
        runs = json.loads(fit["found"].split(" ", 1)[1])["runs"] if fit["found"] else []
        check(
            "fit on Web Workers",
            bool(runs) and all(r["error"] is None and "workers" in r["where"] for r in runs),
            [(r["method"], r["where"], r["error"]) for r in runs],
        )
        fresh = browser.new_context()
        off = visit(fresh, url + "?coi=0")
        check("?coi=0 loads unisolated", not off["isolated"] and not off["controller"], off)
        browser.close()
    server.shutdown()
    if failures:
        sys.exit(f"{len(failures)} check(s) failed")


if __name__ == "__main__":
    main()
