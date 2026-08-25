# SLM Sequence Compiler

This repository contains the TypeScript orchestration layer and a dependency-free
Rust/WebAssembly numerical core for the pipeline in `design.md`. Assignment and
one- and two-dimensional FFTs execute over packed arrays in Wasm linear memory;
validation, planning orchestration, storage, and the public API remain in
TypeScript.

```ts
import { SlmSequenceCompiler } from "slm-sequence-compiler";

const compiler = await SlmSequenceCompiler.create({ calibration });
const sequence = await compiler.compileRearrangement({
  initialAtoms: [{ xUm: 0, yUm: 0 }],
  targetSites: [{ xUm: 10, yUm: 0 }],
  calibrationId: calibration.manifest.calibrationId,
});
```

The compiler keeps frame data in `MemoryFrameStore` by default. A caller can
provide another store implementing the exported `FrameStore` interface.

`getWasmCoreInfo()` reports the loaded core ABI and build identifier. The Wasm
bytes are embedded in the JavaScript output so the synchronous numerical API
works in Node and browsers without a separate fetch. Package builds also emit
`wasm/slm_core.wasm`. Building from source requires Rust with the
`wasm32-unknown-unknown` target:

```sh
rustup target add wasm32-unknown-unknown
npm run build
```

Node-only deployment builders automatically validate and reuse the checked-in
`wasm/slm_core.wasm` artifact when Cargo is unavailable. Run
`npm run build:wasm:prebuilt` to exercise that path locally. After changing the
Rust crate, run `npm run build:wasm` in a Rust-capable environment and commit
both the refreshed Wasm artifact and generated TypeScript byte module.

The browser workspace includes a snapping visual coordinate editor for adding,
moving, inspecting, and removing initial atoms and target sites. Every visual
change is reflected in the JSON inputs, which remain available for direct edits
and imports.

The `/tweezers` workspace generates a single SLM frame directly from optical
tweezer coordinates, requested phases, and relative intensities. Inputs can be
edited in the table or as JSON, and the resulting 8-bit frame can be downloaded
as raw pixels, a PNG preview, or with its JSON metadata.

SLM output defaults to 1272×1024 pixels. Width and height can be changed from
the compiler controls before a run; exported frame dimensions follow those
values. Non-power-of-two dimensions are centered on a zero-padded power-of-two
FFT grid and cropped back to the selected active SLM area for export.

The workspace is a Vue 3 single-page application built with Vite. Run it locally
with `npm run dev`, create the production site in `web-dist/` with
`npm run build:web`, or inspect that production build with `npm run preview`.
Cloudflare Workers static-assets deployment uses the same `web-dist/` output via
`npm run deploy`.

Measured runs must provide a calibration package with a phase-response or
inverse phase LUT. Set `simulationMode: true` only for the synthetic identity
calibration used by reference simulations. Set `hologram.requireConvergence`
to `true` when a run must reject frames that exhaust their WGS budget.
