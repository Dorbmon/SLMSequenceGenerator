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

Target sites can also be extracted from an uploaded target-field image. The
in-browser importer detects bright or dark connected spots in a dedicated Web
Worker, previews their weighted centroids, and maps them into a centered
micrometer field using adjustable physical width and height. Threshold, minimum
blob area, polarity, and point limit remain adjustable before the detected sites
replace the target JSON.

The `/tweezers` workspace generates a single SLM frame directly from optical
tweezer coordinates, requested phases, and relative intensities. Inputs can be
edited in the table or as JSON. It also accepts target-field images in two
modes: isolated-spot mode converts connected components to centroids, while
image-pattern mode samples text and other connected shapes into a spatially
uniform point cloud. Integrated pixel signal seeds relative intensity, and
imported phases start at zero for subsequent editing. The resulting 8-bit frame
can be downloaded as raw pixels, a standards-compatible grayscale BMP, or with
its JSON metadata. A forward-simulation panel accepts the exported indexed BMP
or a raw U8 phase-code frame, applies the same uniform active aperture and
centered power-of-two FFT model in a dedicated worker, and renders the
FFT-shifted focal-plane intensity on a linear or adjustable decibel scale. The
normalized intensity field can also be exported as a display BMP or Float32
raw array. Forward propagation supports both Wasm and a GPU-resident WebGPU
path.

Browser calculations run in a dedicated Web Worker. The interface, progress
animations, navigation, and elapsed-time display remain responsive while Wasm
is solving, and either sequence compilation or single-frame generation can be
cancelled immediately.

Both workspaces expose a compute-backend selector. The compatibility path uses
the Rust/WebAssembly FFT core. On supported browsers, the WebGPU path keeps the
phase field, radix-2 FFT intermediates, WGS weights, phase constraints,
quantization, and accepted sequential state in GPU buffers across iterations;
only the final cropped frame and compact metrics are read back. WebGPU support
is checked inside the same dedicated worker that performs the computation.
Wasm and WebGPU use the same range-reduced, seeded initialization when target
phasors destructively cancel, preventing precision-dependent solver branches.
Frames that exhaust their WGS budget remain exportable but are shown as a
non-convergence warning instead of a verified acceptance.

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
