// Run a `dart compile wasm` output under Node:
//   node run_wasm.mjs app.wasm [args...]
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const wasmPath = process.argv[2];
const loader = await import(pathToFileURL(wasmPath.replace(/\.wasm$/, '.mjs')).href);
const compiled = await loader.compile(await readFile(wasmPath));
const app = await compiled.instantiate({});
app.invokeMain(...process.argv.slice(3));
