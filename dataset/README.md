# Node/Dawn WebGPU WGS HDF5 dataset generator

This directory generates training records of

```text
(trap positions, measured target-plane phases) -> SLMControl3 logical SLM frame
```

The compute process is headless Node.js using Dawn's `dawn.node` WebGPU
binding. It instantiates the existing `WebGpuSequentialWgsSolver`; there is no
browser, CPU solver, or Wasm solver fallback. Python owns the HDF5 collector,
validation, sharding, checksums, and crash-safe resume.

Every sample uses `REFERENCE_WGS`, unit target intensity, and a zero placeholder
input phase. `measured_phases` are the phases sampled from the final quantized
logical frame under the ideal SLMControl3 compensation model, in the same stable
order as `positions` and `trap_ids`. This is the solver variable named
`measuredPhases`; it is not a camera or interferometer measurement.
Non-converged or numerically invalid candidates are counted but never written.

## Install

From the repository root:

```powershell
uv venv dataset/.venv
uv pip install --python dataset/.venv/Scripts/python.exe -r dataset/requirements.txt
npm install --prefix dataset
```

The dataset-local Node package pins the Dawn-backed
[`webgpu`](https://www.npmjs.com/package/webgpu) runtime and keeps its native
dependencies isolated under `dataset/node_modules/`.

## Generate

Only the accepted sample count is required. For the normal SLMControl3 display
path, do not pass an external LUT:

```powershell
dataset/.venv/Scripts/python.exe dataset/generate.py `
  --samples 10000 `
  --dawn-backend d3d12
```

The default output is `<repo>/dataset/data`. The Python process starts a
loopback collector and then starts the Node/Dawn runner automatically. No UI or
browser interaction is required. In an interactive terminal, generation uses a
single-line progress bar showing accepted samples, the current trap count,
rejections, throughput, and ETA. When output is redirected to a file or CI log,
progress is automatically reduced to roughly one update per percentage point.

Important options:

```text
--samples N                         accepted samples to produce (required)
--lut PATH                          optional; bake a measured LUT into raw codes
--output-dir PATH                   default: <repo>/dataset/data
--shard-size N                      default: 256
--min-traps N / --max-traps N       default: 1 / 2000
--count-distribution log-uniform    default; uniform is also available
--iterations N                      encoded WGS budget, default: 12
--max-iterations N                  hard cap only; no adaptive extension
--convergence-tolerance VALUE       default: 0.001
--dataset-seed UINT32               count/coordinate reproducibility
--solver-seed UINT32                fixed free-phase WGS seed
--phase-convention VALUE            override ambiguous LUT absolute range
--min-separation-um VALUE           override optical-resolution-safe spacing
--max-retries-per-sample N          per-run retry window; restart continues
--dawn-backend NAME                 e.g. d3d12 or vulkan; Dawn auto if omitted
--dawn-adapter NAME                 optional native adapter selector
--dawn-option OPTION                repeatable raw Dawn create option
--no-runner                         collector-only/manual integration mode
```

Run `dataset/generate.py --help` for all optical, sampling, and solver options.

`--iterations` is the work encoded for every solve. Early convergence does not
shorten the already-recorded WebGPU command buffer, and `--max-iterations` is
only a hard cap. If one invocation exhausts its retry window, rerunning the
same command continues with the next deterministic positions. A Dawn/device
exception stops immediately and does not consume a data-rejection attempt.

Ctrl+C leaves a flushed `.partial` shard. The identical command resumes it. A
changed output/LUT mode, LUT, Dawn selection, numerical configuration, or
sampling configuration requires a new output directory because the
configuration hash intentionally changes.

## Frame and LUT modes

The default mode is `SLMCONTROL3_LOGICAL`. It exports logical UINT8 phase codes
covering one wrapped 2π cycle. Display these frames through SLMControl3 with the
matching wavelength LUT enabled. For the current experiment that means the
407 nm SLMControl3 calibration. Do not bake the inspection-sheet value 217 into
these frames.

Passing `--lut PATH` explicitly selects the compatibility mode
`DEVICE_READY_LUT_BAKED`. It maps phase to raw device codes using that measured
response. When displaying those frames through SLMControl3, its LUT must be
disabled or the correction will be applied twice.

For the optional baked mode, accepted LUT inputs are:

Accepted inputs are:

- a JSON number array or recognized `phaseResponseLut` object;
- one-column CSV or whitespace-separated values;
- a `code,phase_rad` CSV table.

JSON/one-column values are treated as uniformly sampled over display codes
0–255. An explicit code column means literal UINT8 codes, must include both 0
and 255, and is linearly expanded to all 256 codes. All phases must be finite,
monotonic, and span a non-zero range. Source and normalized-value SHA-256 hashes
are stored in the manifest and every shard.

The inverse mapper accepts LUTs represented on `-pi..pi` or `0..2pi`. If a
response is stored as `0..-2pi`, add `2*pi` to every value first and pass
`--phase-convention zero-to-two-pi`; the generator rejects the unsupported
representation instead of silently clipping half the phase range.

## HDF5 layout

Each `shard-xxxxx.h5` contains:

```text
frames                 uint8   [S, active_height, active_width]
positions              float32 [S, 2000, 2]
measured_phases        float32 [S, 2000]
trap_ids               uint32  [S, 2000]
trap_count             uint16  [S]
sample_id              uint64  [S]
sampling_seed          uint32  [S]
frame_crc32            uint32  [S]
metrics/*                       convergence and optical diagnostics
calibration/phase_response_lut  float64 [optional; baked-LUT mode only]
```

Entries after `trap_count[i]` are zero padding. Frames are row-major active-SLM
codes, not the padded FFT grid. Attributes include Dawn/backend selection,
solver and sampling configuration, dimensions, units, `frame_mode`, LUT
application, and controller requirements. In the default mode,
`calibration/phase_response_lut` is deliberately absent and
`lut_application=SLMCONTROL3`. `manifest.json` records ranges and shard SHA-256
sums.

For training, mask by `trap_count`. Since phase wraps at `-pi/pi`, representing
it as `sin(phase)` and `cos(phase)` is generally safer than using a raw scalar.

At 1272×1024, one frame is about 1.30 MB; 10,000 uncompressed frames are about
12.1 GiB before HDF5 overhead.

## Why 8192×8192 is not the default

The default FFT grid is the smallest power-of-two grid containing the active
SLM: 2048×1024 for a 1272×1024 device. Trap sampling is an exact NUDFT at
arbitrary physical coordinates, so it does not round targets to FFT bins.
Increasing the padded grid therefore does not increase trap-coordinate
precision or add physical SLM degrees of freedom.

For the current solver, 8192×8192 has 32 times more pixels than the default. A
single complex field buffer is 512 MiB, persistent buffers total several GiB,
and a one-dimensional pixel dispatch needs 262,144 workgroups, exceeding the
65,535 limit of the tested Dawn/D3D12 device. Keep 2048×1024 for production and
benchmark representative trap counts before a large run. In particular,
support for 2000 traps is an interface limit, not a throughput guarantee: exact
target sampling scales with `pixels × traps × iterations`.

## Verify

```powershell
npm run typecheck --prefix dataset
npm test --prefix dataset
dataset/.venv/Scripts/python.exe -m pytest dataset/tests -q
npm run typecheck
```

The end-to-end smoke test uses Node/Dawn D3D12 to generate a real HDF5 shard and
verifies frame type/shape, CRC, measured phase, convergence flags, zero padding,
SLMControl3 frame metadata, and manifest checksum.
