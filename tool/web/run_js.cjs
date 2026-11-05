// Run a `dart compile js` output under Node: node run_js.cjs app.js
// dart2js targets browsers; give it the globals it expects.
globalThis.self = globalThis;
require(require('node:path').resolve(process.argv[2]));
