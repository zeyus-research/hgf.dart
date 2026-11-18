// Cross-origin isolation for hosts that cannot send headers (GitHub Pages).
//
// Adds COOP/COEP to every same-origin response in its scope, so the page is
// `crossOriginIsolated`: skwasm then renders on a worker thread (about 4x
// less main-thread time while dragging on the builder's canvases) and the
// fit workers can share memory. Cross-origin requests (the engine and fonts
// from gstatic, which send CORS headers) are left to the browser.
//
// Registered by flutter_bootstrap.js, which reloads the page once on the
// first visit so that the reloaded page comes through this worker. No
// caching: responses come from the network (and the HTTP cache) as before.

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (new URL(request.url).origin !== self.location.origin) return;
  // A Chrome quirk: devtools issues these and fetch() rejects them.
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') {
    return;
  }
  event.respondWith(fetch(request).then(isolate));
});

function isolate(response) {
  // Opaque redirects (navigations) cannot be rebuilt; pass them through.
  if (response.status === 0) return response;
  const headers = new Headers(response.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
  headers.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}
