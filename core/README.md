# SLM Wasm core

This dependency-free Rust crate implements the packed numerical kernels used by
the TypeScript API. It targets `wasm32-unknown-unknown`, owns no dynamic heap,
and exports one ABI version function plus FFT and Hungarian assignment entry
points. JavaScript grows linear memory once and passes non-overlapping packed
buffers to each call, so hot loops do not cross the Wasm boundary.

The module imports `env.sin` and `env.cos` as `(f64) -> f64`. Power-of-two FFTs
invoke them once per stage; the reference DFT invokes them once per output.

Run `npm run build:wasm` from the repository root to compile the crate, validate
the imports and exports, generate `wasm/slm_core.wasm`, and refresh the embedded
TypeScript byte module.

For Node-only deployment environments, `npm run build:wasm:prebuilt` validates
the checked-in Wasm artifact and refreshes the embedded module without invoking
Cargo.
