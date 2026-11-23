{{flutter_js}}
{{flutter_build_config}}
_flutter.loader.load({
  config: {
    wasmAllowList: {
      blink: true,
      webkit: true,
      gecko: true,
    },
  },
});
// Not Flutter's default bootstrap: no `serviceWorkerSettings`. Flutter's own
// service worker is deprecated (it only unregisters itself) and would take
// the place of coi_sw.js, which makes the page cross-origin isolated where
// the host cannot send COOP/COEP (GitHub Pages; see coi_sw.js).
//
// First visit: register coi_sw.js, wait until it is active, reload once
// (before the engine loads, so nothing is downloaded twice). If the page is
// still not isolated after that reload (a browser that ignores headers from
// a service worker), it gives up for the session and loads as before.
// `?coi=0` skips all of this.
(async () => {
  const flag = 'hgf-coi-reloaded';
  const storage = {
    get: () => { try { return sessionStorage.getItem(flag); } catch (_) { return null; } },
    set: () => { try { sessionStorage.setItem(flag, '1'); } catch (_) {} },
    clear: () => { try { sessionStorage.removeItem(flag); } catch (_) {} },
  };

  async function reloadIsolated() {
    if (window.crossOriginIsolated) {
      storage.clear();
      return false;
    }
    if (new URLSearchParams(location.search).get('coi') === '0') return false;
    if (!window.isSecureContext || !('serviceWorker' in navigator)) return false;
    let registration;
    try {
      registration = await navigator.serviceWorker.register('coi_sw.js');
    } catch (e) {
      console.warn('Cross-origin isolation: no service worker:', e);
      return false;
    }
    if (storage.get()) {
      console.warn('Cross-origin isolation: not isolated after a reload; going on without it.');
      return false;
    }
    const worker = registration.installing || registration.waiting || registration.active;
    if (!worker) return false;
    if (worker.state !== 'activated') {
      const activated = new Promise((resolve) => {
        worker.addEventListener('statechange', () => {
          if (worker.state === 'activated') resolve(true);
          if (worker.state === 'redundant') resolve(false);
        });
      });
      const timeout = new Promise((resolve) => setTimeout(() => resolve(false), 5000));
      if (!(await Promise.race([activated, timeout]))) return false;
    }
    storage.set();
    location.reload();
    return true;
  }

  let reloading = false;
  try {
    reloading = await reloadIsolated();
  } catch (e) {
    console.warn('Cross-origin isolation failed:', e);
  }
  if (!reloading) _flutter.loader.load();
})();
