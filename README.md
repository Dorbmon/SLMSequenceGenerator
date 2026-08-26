# SLM Sequence Compiler

This repository contains the TypeScript orchestration layer and a dependency-free
Rust/WebAssembly numerical core for the pipeline in `design.md`. Assignment,
one- and two-dimensional FFTs, exact arbitrary-frequency trap sampling, and its
adjoint trap synthesis execute over packed arrays in Wasm linear memory;
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
edited in the table or as JSON. A freehand focal-plane editor can draw or erase
trap paths, undo gestures, and load the current coordinates for revision. It
samples pointer paths by physical arc length rather than browser event rate and
rejects crossings or retraced samples inside the calibrated optical-resolution
limit. Applied drawings continue through the existing table and JSON model.
The workspace also accepts target-field images in two
modes: isolated-spot mode converts connected components to centroids, while
image-pattern mode samples text and other connected shapes into a spatially
uniform point cloud. Optical-tweezer imports discard sparse edge bins, fit the
detected foreground, auto-size the physical field from the calibrated aperture
resolution, and consolidate samples that would address nearly identical Fourier
modes. Uniform trap intensity and free output phase are the safe defaults;
average image brightness and phase-locked JSON/table inputs remain available.
The resulting 8-bit frame
can be downloaded as raw pixels, a standards-compatible grayscale BMP, or with
its JSON metadata. A forward-simulation panel accepts the exported indexed BMP
or a raw U8 phase-code frame, applies the same calibrated Gaussian incident
field and centered power-of-two FFT model in a dedicated worker, and renders the
FFT-shifted focal-plane intensity on a linear or adjustable decibel scale. When
current target coordinates are available, the preview opens on a calibrated
target-region crop, marks the requested positions, and uses physical focal-plane
aspect instead of the generally non-square FFT pixel aspect. The full optical
field remains selectable. Measured display-code phase LUTs and the configured
beam diameter/centre are honored by both the Wasm and WebGPU forward paths. The
normalized intensity field can also be exported as a display BMP or Float32
raw array. Forward propagation supports both Wasm and a GPU-resident WebGPU
path.

Manually entered focal-plane coordinates are never auto-fitted. Both workspaces expose the laser
wavelength, Fourier-lens focal length, and SLM pixel pitch and use the physical
Fraunhofer relation `u = x N p / (lambda f)` (with the image-row sign applied to
`y`). The displayed field of view is the corresponding Nyquist interval. The
single-frame workspace defaults to the current Hamamatsu X15213-05 experiment:
407 nm, 12.5 µm pixels, a 100 mm physical Fourier lens, and an 8 mm 1/e²
Gaussian beam diameter. All values remain editable. Its default SLMControl3
pipeline exports logical `0–255 = 0–2π` phase codes and deliberately leaves the
407 nm LUT to SLMControl3. A mutually exclusive device-ready mode can bake in
the inspection-sheet 2π signal level (217 by default) or a monotonic measured
display-code-to-phase LUT from JSON, CSV, or plain text; its output must be
loaded with the SLMControl3 LUT disabled to prevent double correction.
The single-frame workspace reports a conservative effective focal-spot scale
that combines the finite active aperture with Gaussian-beam underfill,
rejects unresolved trap pairs before computation, and exposes the maximum
relative-amplitude error used by its certificate. Image patterns select a 10%
limit by default; manually entered complex-field targets retain the strict
0.01% default, and either value can be changed explicitly.

Browser calculations run in a dedicated Web Worker. The interface, progress
animations, navigation, and elapsed-time display remain responsive while Wasm
is solving, and either sequence compilation or single-frame generation can be
cancelled immediately.

Both workspaces expose a compute-backend selector. Wasm and WebGPU use the same
trap-domain WGS model. It evaluates the unnormalised discrete Fourier sum
exactly at each requested, potentially fractional trap coordinate; a measured
phase-precompensation stage separates requested output phases from the internal
synthesis phasors, and damped feedback then balances amplitudes before exact
adjoint synthesis. Candidate
quality is measured after display-code quantization and LUT decoding, and the
best certified candidate is retained so a larger iteration budget cannot return
a worse frame. No bilinear FFT-bin sampling or scattering is used. The WebGPU
path keeps that entire iterative process, best-candidate state, quantization,
and accepted sequential state in GPU buffers. A full radix-2 FFT is run only
after quantization for full-plane power and ghost diagnostics, followed by one
cropped-frame/metrics readback.
WebGPU support is checked inside the same dedicated worker that performs the
computation. Both backends use the same range-reduced, seeded initialization
when target phasors destructively cancel. Frames that exhaust their WGS budget
remain exportable but are shown as a non-convergence warning instead of a
verified acceptance.

SLM output defaults to 1272×1024 pixels. Width and height can be changed from
the compiler controls before a run; exported frame dimensions follow those
values. Non-power-of-two dimensions are centered on a zero-padded power-of-two
FFT grid and cropped back to the selected active SLM area for export.

The workspace is a Vue 3 single-page application built with Vite. Run it locally
with `npm run dev`, create the production site in `web-dist/` with
`npm run build:web`, or inspect that production build with `npm run preview`.
Cloudflare Workers static-assets deployment uses the same `web-dist/` output via
`npm run deploy`.

The regression suite includes artifact-boundary oracles for an off-grid trap
and the default four-trap phase-locked target. They export and decode the indexed
grayscale BMP, then independently evaluate that decoded phase frame with a
direct complex Fourier sum. This guards against a backend appearing converged
under its own sampling approximation while emitting an optically different
frame.

Measured runs must provide a calibration package with a phase-response or
inverse phase LUT. Set `simulationMode: true` only for the synthetic identity
calibration used by reference simulations. Set `hologram.requireConvergence`
to `true` when a run must reject frames that exhaust their WGS budget.
