<script setup lang="ts">
import { computed, nextTick, ref, shallowRef, watch } from "vue";
import {
  SequentialWgsSolver,
  crc32,
  type CalibrationPackage,
  type FrameMetrics,
} from "../../../src/index.js";
import SlmFramePreview from "../components/SlmFramePreview.vue";
import {
  DEFAULT_SLM_HEIGHT,
  DEFAULT_SLM_WIDTH,
  MAX_SLM_DIMENSION,
  MIN_SLM_DIMENSION,
  fftDimensionFor,
  normalizeSlmDimension,
} from "../lib/resolution.js";
import {
  DEFAULT_OPTICAL_TWEEZERS,
  cloneOpticalTweezers,
  nextTweezerId,
  opticalTweezersToFrame,
  parseOpticalTweezers,
  serializeOpticalTweezers,
  type OpticalTweezerInput,
} from "../lib/tweezers.js";

type JsonStatus = "synced" | "dirty" | "invalid";
type OutputState = "idle" | "running" | "accepted" | "rejected";

const upload = ref<HTMLInputElement | null>(null);
const tweezers = ref<OpticalTweezerInput[]>(cloneOpticalTweezers(DEFAULT_OPTICAL_TWEEZERS));
const jsonDraft = ref(serializeOpticalTweezers(tweezers.value));
const jsonStatus = ref<JsonStatus>("synced");
const slmWidth = ref(DEFAULT_SLM_WIDTH);
const slmHeight = ref(DEFAULT_SLM_HEIGHT);
const iterations = ref(4);
const running = ref(false);
const outputState = ref<OutputState>("idle");
const outputPixels = shallowRef<Uint8Array | Uint16Array | null>(null);
const metrics = shallowRef<FrameMetrics | null>(null);
const elapsedMs = ref<number | null>(null);
const errorMessage = ref("");
const checksum = ref<number | null>(null);
let suppressTableSync = false;
let generation = 0;

const fftWidth = computed(() => fftDimensionFor(slmWidth.value));
const fftHeight = computed(() => fftDimensionFor(slmHeight.value));
const phaseStatus = computed(() => {
  if (running.value) return "SOLVING / WASM WGS";
  if (outputState.value === "accepted") return "FRAME ACCEPTED";
  if (outputState.value === "rejected") return "GENERATION REJECTED";
  return "AWAITING TWEEZER INPUT";
});
const jsonStatusLabel = computed(() => {
  if (jsonStatus.value === "dirty") return "UNAPPLIED CHANGES";
  if (jsonStatus.value === "invalid") return "INVALID JSON";
  return "SYNCHRONIZED";
});
const maximumPlotX = computed(() => Math.max(5, ...tweezers.value.map((tweezer) => Math.abs(finiteOrZero(tweezer.xUm)))) * 1.18);
const maximumPlotY = computed(() => Math.max(5, ...tweezers.value.map((tweezer) => Math.abs(finiteOrZero(tweezer.yUm)))) * 1.18);

watch(tweezers, () => {
  if (suppressTableSync) return;
  jsonDraft.value = serializeOpticalTweezers(tweezers.value);
  jsonStatus.value = "synced";
  invalidateOutput();
}, { deep: true });

function finiteOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function invalidateOutput(): void {
  generation += 1;
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = null;
  checksum.value = null;
  outputState.value = "idle";
  errorMessage.value = "";
}

function markJsonDirty(): void {
  jsonStatus.value = "dirty";
  invalidateOutput();
}

function applyJson(): boolean {
  try {
    const parsed = parseOpticalTweezers(jsonDraft.value);
    suppressTableSync = true;
    tweezers.value = parsed;
    jsonDraft.value = serializeOpticalTweezers(parsed);
    jsonStatus.value = "synced";
    errorMessage.value = "";
    invalidateOutput();
    nextTick(() => { suppressTableSync = false; });
    return true;
  } catch (error) {
    jsonStatus.value = "invalid";
    errorMessage.value = error instanceof Error ? error.message : "Invalid optical tweezer input";
    return false;
  }
}

function validatedTweezers(): OpticalTweezerInput[] | null {
  if (jsonStatus.value !== "synced" && !applyJson()) return null;
  try {
    const parsed = parseOpticalTweezers(serializeOpticalTweezers(tweezers.value));
    errorMessage.value = "";
    return parsed;
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Invalid optical tweezer input";
    outputState.value = "rejected";
    return null;
  }
}

function addTweezer(): void {
  tweezers.value.push({
    trapId: nextTweezerId(tweezers.value),
    xUm: 0,
    yUm: 0,
    phaseRad: 0,
    intensity: 1,
  });
}

function removeTweezer(index: number): void {
  tweezers.value.splice(index, 1);
}

function reset(): void {
  generation += 1;
  suppressTableSync = true;
  tweezers.value = cloneOpticalTweezers(DEFAULT_OPTICAL_TWEEZERS);
  jsonDraft.value = serializeOpticalTweezers(tweezers.value);
  jsonStatus.value = "synced";
  slmWidth.value = DEFAULT_SLM_WIDTH;
  slmHeight.value = DEFAULT_SLM_HEIGHT;
  iterations.value = 4;
  running.value = false;
  outputState.value = "idle";
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = null;
  checksum.value = null;
  errorMessage.value = "";
  nextTick(() => { suppressTableSync = false; });
}

function updateDimension(event: Event, dimension: "width" | "height"): void {
  const input = event.target as HTMLInputElement;
  const current = dimension === "width" ? slmWidth.value : slmHeight.value;
  if (!Number.isFinite(input.valueAsNumber)) {
    input.value = String(current);
    return;
  }
  try {
    const normalized = normalizeSlmDimension(Math.round(input.valueAsNumber), `SLM ${dimension}`);
    if (dimension === "width") slmWidth.value = normalized;
    else slmHeight.value = normalized;
    input.value = String(normalized);
    invalidateOutput();
  } catch (error) {
    input.value = String(current);
    errorMessage.value = error instanceof Error ? error.message : `Invalid SLM ${dimension}`;
  }
}

function updateIterations(event: Event): void {
  iterations.value = Number((event.target as HTMLInputElement).value);
  invalidateOutput();
}

function calibrationFor(input: readonly OpticalTweezerInput[]): CalibrationPackage {
  const maximumX = Math.max(1, ...input.map((tweezer) => Math.abs(tweezer.xUm)));
  const maximumY = Math.max(1, ...input.map((tweezer) => Math.abs(tweezer.yUm)));
  return {
    manifest: {
      calibrationId: "browser-single-frame",
      wavelengthNm: 1,
      activeWidth: slmWidth.value,
      activeHeight: slmHeight.value,
      fftWidth: fftWidth.value,
      fftHeight: fftHeight.value,
      coordinateConvention: "+x right, +y up",
    },
    coordinateTransform: {
      originXUm: fftWidth.value / 2,
      originYUm: fftHeight.value / 2,
      scaleX: fftWidth.value * 0.4 / maximumX,
      scaleY: fftHeight.value * 0.4 / maximumY,
    },
  };
}

async function generateFrame(): Promise<void> {
  if (running.value) return;
  const input = validatedTweezers();
  if (!input) return;
  const activeGeneration = ++generation;
  running.value = true;
  outputState.value = "running";
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = null;
  checksum.value = null;
  errorMessage.value = "";

  await nextTick();
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  const started = performance.now();
  try {
    const solver = new SequentialWgsSolver(calibrationFor(input), {
      width: fftWidth.value,
      height: fftHeight.value,
      format: "UINT8",
      firstFrameIterations: iterations.value,
      subsequentFrameIterations: iterations.value,
      maxIterations: iterations.value,
      targetPhaseMode: "PHASE_LOCKED_WGS",
      backgroundPolicy: "ZERO",
      requireConvergence: false,
      measureSolveTime: true,
    });
    const result = solver.solveSequentialFrame(opticalTweezersToFrame(input));
    if (activeGeneration !== generation) return;
    outputPixels.value = result.pixels;
    metrics.value = result.metrics;
    elapsedMs.value = performance.now() - started;
    checksum.value = crc32(bytesFor(result.pixels));
    outputState.value = result.metrics.accepted ? "accepted" : "rejected";
    if (!result.metrics.accepted) errorMessage.value = "The generated frame did not pass the numerical quality checks";
  } catch (error) {
    if (activeGeneration !== generation) return;
    outputState.value = "rejected";
    errorMessage.value = error instanceof Error ? error.message : "Unable to generate the SLM frame";
  } finally {
    if (activeGeneration === generation) running.value = false;
  }
}

function bytesFor(pixels: Uint8Array | Uint16Array): Uint8Array {
  return new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportRawFrame(): void {
  if (!outputPixels.value) return;
  const copy = new Uint8Array(bytesFor(outputPixels.value));
  download(
    new Blob([copy.buffer], { type: "application/octet-stream" }),
    `slm-frame-${slmWidth.value}x${slmHeight.value}-u8.bin`,
  );
}

function frameImageData(): ImageData | null {
  const pixels = outputPixels.value;
  if (!pixels) return null;
  const image = new ImageData(slmWidth.value, slmHeight.value);
  const divisor = pixels instanceof Uint16Array ? 257 : 1;
  for (let index = 0; index < pixels.length; index += 1) {
    const value = Math.round(pixels[index]! / divisor);
    const offset = index * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }
  return image;
}

function exportPng(): void {
  const image = frameImageData();
  if (!image) return;
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  canvas.getContext("2d")?.putImageData(image, 0, 0);
  canvas.toBlob((blob) => {
    if (blob) download(blob, `slm-frame-${slmWidth.value}x${slmHeight.value}.png`);
  }, "image/png");
}

function exportMetadata(): void {
  if (!outputPixels.value || !metrics.value) return;
  const input = validatedTweezers();
  if (!input) return;
  const calibration = calibrationFor(input);
  const payload = {
    formatVersion: "1.0",
    output: {
      width: slmWidth.value,
      height: slmHeight.value,
      pixelFormat: "UINT8",
      byteLength: outputPixels.value.byteLength,
      crc32: checksum.value,
    },
    fftGrid: { width: fftWidth.value, height: fftHeight.value },
    solver: {
      backend: "wasm-fft-phase-locked-wgs",
      requestedIterations: iterations.value,
      backgroundPolicy: "ZERO",
      elapsedMs: elapsedMs.value,
    },
    calibration: {
      manifest: calibration.manifest,
      coordinateTransform: calibration.coordinateTransform,
    },
    tweezers: input,
    metrics: metrics.value,
  };
  download(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), "slm-frame-metadata.json");
}

async function loadJson(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  try {
    jsonDraft.value = await file.text();
    jsonStatus.value = "dirty";
    applyJson();
  } catch (error) {
    jsonStatus.value = "invalid";
    errorMessage.value = error instanceof Error ? error.message : "Unable to read the JSON file";
  }
}

function plotX(value: number): number {
  return 320 + finiteOrZero(value) / maximumPlotX.value * 255;
}

function plotY(value: number): number {
  return 160 - finiteOrZero(value) / maximumPlotY.value * 122;
}

function phaseColor(phase: number): string {
  const turn = ((finiteOrZero(phase) / (Math.PI * 2)) % 1 + 1) % 1;
  return `hsl(${Math.round(turn * 300 + 75)} 78% 68%)`;
}

function phaseDegrees(phase: number): string {
  return `${(finiteOrZero(phase) * 180 / Math.PI).toFixed(1)}°`;
}

defineExpose({ reset });
</script>

<template>
  <section class="workspace tweezer-workspace">
    <div class="workspace-heading">
      <div>
        <p class="eyebrow">Direct hologram workspace</p>
        <h1>Optical tweezer frame</h1>
      </div>
        <p class="heading-note">Enter tweezer coordinates and phases<br>to synthesize one exportable 8-bit SLM frame.</p>
    </div>

    <div class="tweezer-page-grid">
      <section class="panel tweezer-editor" aria-labelledby="tweezer-input-title">
        <div class="panel-bar">
          <span id="tweezer-input-title" class="panel-kicker">TWEEZERS / COORDINATES + PHASE</span>
          <span>{{ String(tweezers.length).padStart(2, "0") }} TRAPS</span>
        </div>
        <div class="tweezer-editor-intro">
          <h2>Target field</h2>
          <p>Coordinates are in micrometres and auto-fit to the simulated Fourier plane. Phase is the desired complex-field phase in radians; intensity is relative power.</p>
        </div>

        <div class="tweezer-plane" aria-label="Optical tweezer coordinate and phase preview">
          <svg viewBox="0 0 640 320" role="img">
            <title>Optical tweezer coordinate and phase preview</title>
            <line x1="40" y1="160" x2="600" y2="160"></line>
            <line x1="320" y1="24" x2="320" y2="296"></line>
            <g v-for="(tweezer, index) in tweezers" :key="`${tweezer.trapId}:${index}`">
              <circle class="tweezer-phase-glow" :cx="plotX(tweezer.xUm)" :cy="plotY(tweezer.yUm)" r="15" :fill="phaseColor(tweezer.phaseRad)"></circle>
              <circle class="tweezer-phase-point" :cx="plotX(tweezer.xUm)" :cy="plotY(tweezer.yUm)" r="5" :fill="phaseColor(tweezer.phaseRad)"></circle>
              <text :x="plotX(tweezer.xUm) + 10" :y="plotY(tweezer.yUm) - 9">{{ tweezer.trapId }} / {{ phaseDegrees(tweezer.phaseRad) }}</text>
            </g>
          </svg>
          <div class="tweezer-plane-caption"><span>PHASE / HUE</span><span>UM PLANE / TOP VIEW</span></div>
        </div>

        <div class="tweezer-table-wrap">
          <table class="tweezer-table">
            <thead><tr><th>ID</th><th>X / UM</th><th>Y / UM</th><th>PHASE / RAD</th><th>INTENSITY</th><th></th></tr></thead>
            <tbody>
              <tr v-for="(tweezer, index) in tweezers" :key="index">
                <td><input v-model.number="tweezer.trapId" type="number" min="0" step="1" :disabled="running" :aria-label="`Tweezer ${index + 1} ID`"></td>
                <td><input v-model.number="tweezer.xUm" type="number" step="any" :disabled="running" :aria-label="`Tweezer ${index + 1} X coordinate`"></td>
                <td><input v-model.number="tweezer.yUm" type="number" step="any" :disabled="running" :aria-label="`Tweezer ${index + 1} Y coordinate`"></td>
                <td><input v-model.number="tweezer.phaseRad" type="number" step="any" :disabled="running" :aria-label="`Tweezer ${index + 1} phase in radians`"></td>
                <td><input v-model.number="tweezer.intensity" type="number" min="0.001" step="any" :disabled="running" :aria-label="`Tweezer ${index + 1} intensity`"></td>
                <td><button class="tweezer-remove" type="button" :disabled="running" :aria-label="`Remove tweezer ${index + 1}`" @click="removeTweezer(index)">&times;</button></td>
              </tr>
            </tbody>
          </table>
        </div>
        <button class="tweezer-add" type="button" :disabled="running" @click="addTweezer">+ Add optical tweezer</button>

        <div class="tweezer-json">
          <div class="tweezer-json-heading">
            <label for="tweezer-json-input">JSON INPUT / ADVANCED</label>
            <span :class="`is-${jsonStatus}`">{{ jsonStatusLabel }}</span>
          </div>
          <textarea id="tweezer-json-input" v-model="jsonDraft" spellcheck="false" :disabled="running" @input="markJsonDirty"></textarea>
          <div class="tweezer-json-actions">
            <button type="button" :disabled="running" @click="applyJson">Apply JSON</button>
            <button type="button" :disabled="running" @click="upload?.click()">Load JSON</button>
            <input ref="upload" class="sr-only" type="file" accept="application/json,.json" @change="loadJson">
          </div>
        </div>
      </section>

      <div class="tweezer-output-column">
        <SlmFramePreview
          :pixels="outputPixels"
          :width="slmWidth"
          :height="slmHeight"
          :status="phaseStatus"
        />

        <section class="panel tweezer-generator" aria-labelledby="frame-generator-title">
          <div class="panel-bar">
            <span id="frame-generator-title" class="panel-kicker">FRAME GENERATOR</span>
            <span class="valid-badge">{{ outputState.toUpperCase() }}</span>
          </div>
          <div class="tweezer-generator-body">
            <div class="resolution-block">
              <div class="control-label"><span>SLM RESOLUTION</span><output>{{ slmWidth }} &times; {{ slmHeight }} px</output></div>
              <div class="resolution-fields">
                <label>WIDTH / PX
                  <input type="number" :min="MIN_SLM_DIMENSION" :max="MAX_SLM_DIMENSION" step="1" :value="slmWidth" :disabled="running" @change="updateDimension($event, 'width')">
                </label>
                <span aria-hidden="true">&times;</span>
                <label>HEIGHT / PX
                  <input type="number" :min="MIN_SLM_DIMENSION" :max="MAX_SLM_DIMENSION" step="1" :value="slmHeight" :disabled="running" @change="updateDimension($event, 'height')">
                </label>
              </div>
              <p class="resolution-note">FFT COMPUTE GRID {{ fftWidth }} &times; {{ fftHeight }} / POWER-OF-TWO PADDED</p>
              <p class="tweezer-calibration-note">SIMULATION CALIBRATION / LINEAR 0–2π PHASE RESPONSE</p>
            </div>
            <div class="control-block">
              <div class="control-label"><span>WGS ITERATIONS</span><output>{{ String(iterations).padStart(2, "0") }} / FRAME</output></div>
              <input type="range" min="1" max="12" step="1" :value="iterations" :disabled="running" @input="updateIterations">
            </div>
            <p v-if="errorMessage" class="tweezer-error" role="alert">{{ errorMessage }}</p>
            <button class="compile-button tweezer-generate-button" type="button" :disabled="running" @click="generateFrame">
              <span></span>{{ running ? "Generating SLM frame..." : "Generate SLM frame" }}
            </button>
            <div class="tweezer-export-actions">
              <button type="button" :disabled="!outputPixels || running" @click="exportRawFrame">Raw frame <b>&darr;</b></button>
              <button type="button" :disabled="!outputPixels || running" @click="exportPng">PNG preview <b>&darr;</b></button>
              <button type="button" :disabled="!outputPixels || running" @click="exportMetadata">Metadata <b>&darr;</b></button>
            </div>
          </div>
        </section>
      </div>
    </div>

    <div class="metrics-row tweezer-metrics" aria-label="SLM frame metrics">
      <div><small>TWEEZERS</small><strong>{{ tweezers.length }}</strong></div>
      <div><small>CALCULATION TIME</small><strong>{{ elapsedMs === null ? "--" : `${(elapsedMs / 1000).toFixed(2)}s` }}</strong></div>
      <div><small>DIFFRACTION EFFICIENCY</small><strong>{{ metrics ? `${(metrics.diffractionEfficiency * 100).toFixed(1)}%` : "--" }}</strong></div>
      <div><small>MAX PHASE ERROR</small><strong>{{ metrics ? `${metrics.maximumTargetPhaseErrorRad.toFixed(2)} rad` : "--" }}</strong></div>
    </div>
  </section>
</template>
