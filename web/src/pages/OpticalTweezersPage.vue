<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import {
  type CalibrationPackage,
  type FrameMetrics,
} from "../../../src/index.js";
import ComputationActivity from "../components/ComputationActivity.vue";
import SlmFramePreview from "../components/SlmFramePreview.vue";
import TargetImageImporter from "../components/TargetImageImporter.vue";
import { encodeGrayscaleBmp } from "../lib/bmp.js";
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
  parseOpticalTweezers,
  serializeOpticalTweezers,
  type OpticalTweezerInput,
} from "../lib/tweezers.js";
import type {
  CompilerWorkerRequest,
  CompilerWorkerResponse,
  ComputeBackend,
} from "../workers/compiler-messages.js";

type JsonStatus = "synced" | "dirty" | "invalid";
type OutputState = "idle" | "running" | "accepted" | "warning" | "rejected" | "cancelled";

interface TargetImageImporterHandle {
  reset(): void;
}

const props = defineProps<{
  webgpuAvailable: boolean;
  webgpuStatus: string;
}>();

const upload = ref<HTMLInputElement | null>(null);
const targetImageImporter = ref<TargetImageImporterHandle | null>(null);
const tweezers = ref<OpticalTweezerInput[]>(cloneOpticalTweezers(DEFAULT_OPTICAL_TWEEZERS));
const jsonDraft = ref(serializeOpticalTweezers(tweezers.value));
const jsonStatus = ref<JsonStatus>("synced");
const slmWidth = ref(DEFAULT_SLM_WIDTH);
const slmHeight = ref(DEFAULT_SLM_HEIGHT);
const iterations = ref(4);
const computeBackend = ref<ComputeBackend>("wasm");
const running = ref(false);
const outputState = ref<OutputState>("idle");
const outputPixels = shallowRef<Uint8Array | Uint16Array | null>(null);
const metrics = shallowRef<FrameMetrics | null>(null);
const elapsedMs = ref<number | null>(null);
const errorMessage = ref("");
const qualityWarning = ref("");
const checksum = ref<number | null>(null);
const resultBackendId = ref("wasm-fft-phase-locked-wgs");
let suppressTableSync = false;
let generation = 0;
let frameWorker: Worker | null = null;
let elapsedTimer = 0;
let calculationStarted = 0;

const fftWidth = computed(() => fftDimensionFor(slmWidth.value));
const fftHeight = computed(() => fftDimensionFor(slmHeight.value));
const phaseStatus = computed(() => {
  if (running.value) return computeBackend.value === "webgpu" ? "SOLVING / WEBGPU WGS" : "SOLVING / WASM WGS";
  if (outputState.value === "accepted") return "FRAME ACCEPTED";
  if (outputState.value === "warning") return "FRAME READY / NOT CONVERGED";
  if (outputState.value === "rejected") return "GENERATION REJECTED";
  if (outputState.value === "cancelled") return "GENERATION CANCELLED";
  return "AWAITING TWEEZER INPUT";
});
const jsonStatusLabel = computed(() => {
  if (jsonStatus.value === "dirty") return "UNAPPLIED CHANGES";
  if (jsonStatus.value === "invalid") return "INVALID JSON";
  return "SYNCHRONIZED";
});
const maximumPlotX = computed(() => Math.max(5, ...tweezers.value.map((tweezer) => Math.abs(finiteOrZero(tweezer.xUm)))) * 1.18);
const maximumPlotY = computed(() => Math.max(5, ...tweezers.value.map((tweezer) => Math.abs(finiteOrZero(tweezer.yUm)))) * 1.18);
const densePlotGeometry = computed(() => {
  if (tweezers.value.length <= 48) return null;
  const xValues = tweezers.value.map((tweezer) => finiteOrZero(tweezer.xUm));
  const yValues = tweezers.value.map((tweezer) => finiteOrZero(tweezer.yUm));
  const minimumX = Math.min(...xValues);
  const maximumX = Math.max(...xValues);
  const minimumY = Math.min(...yValues);
  const maximumY = Math.max(...yValues);
  const spanX = Math.max(0.1, maximumX - minimumX);
  const spanY = Math.max(0.1, maximumY - minimumY);
  return {
    centerX: (minimumX + maximumX) / 2,
    centerY: (minimumY + maximumY) / 2,
    scale: Math.min(520 / (spanX * 1.08), 240 / (spanY * 1.08)),
  };
});

watch(tweezers, () => {
  if (suppressTableSync) return;
  jsonDraft.value = serializeOpticalTweezers(tweezers.value);
  jsonStatus.value = "synced";
  invalidateOutput();
}, { deep: true });

watch(() => props.webgpuAvailable, (available) => {
  if (!available && computeBackend.value === "webgpu") {
    computeBackend.value = "wasm";
    invalidateOutput();
  }
});

function finiteOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function invalidateOutput(): void {
  generation += 1;
  terminateFrameWorker();
  stopElapsedClock();
  running.value = false;
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = null;
  checksum.value = null;
  outputState.value = "idle";
  errorMessage.value = "";
  qualityWarning.value = "";
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

function applyImageTweezers(
  points: readonly { xUm: number; yUm: number; relativeIntensity: number }[],
  complete: (accepted: boolean) => void,
): void {
  if (running.value || points.length === 0) {
    complete(false);
    return;
  }
  suppressTableSync = true;
  tweezers.value = points.map((point, index) => ({
    trapId: index + 1,
    xUm: point.xUm,
    yUm: point.yUm,
    phaseRad: 0,
    intensity: Math.max(0.001, point.relativeIntensity),
  }));
  jsonDraft.value = serializeOpticalTweezers(tweezers.value);
  jsonStatus.value = "synced";
  invalidateOutput();
  nextTick(() => { suppressTableSync = false; });
  complete(true);
}

function reset(): void {
  generation += 1;
  terminateFrameWorker();
  stopElapsedClock();
  suppressTableSync = true;
  tweezers.value = cloneOpticalTweezers(DEFAULT_OPTICAL_TWEEZERS);
  jsonDraft.value = serializeOpticalTweezers(tweezers.value);
  jsonStatus.value = "synced";
  slmWidth.value = DEFAULT_SLM_WIDTH;
  slmHeight.value = DEFAULT_SLM_HEIGHT;
  iterations.value = 4;
  computeBackend.value = "wasm";
  running.value = false;
  outputState.value = "idle";
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = null;
  checksum.value = null;
  resultBackendId.value = "wasm-fft-phase-locked-wgs";
  errorMessage.value = "";
  qualityWarning.value = "";
  nextTick(() => {
    suppressTableSync = false;
    targetImageImporter.value?.reset();
  });
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

function updateComputeBackend(event: Event): void {
  const value = (event.target as HTMLSelectElement).value as ComputeBackend;
  if (value === "webgpu" && !props.webgpuAvailable) return;
  computeBackend.value = value;
  invalidateOutput();
}

function calibrationFor(input: readonly OpticalTweezerInput[]): CalibrationPackage {
  const maximumX = Math.max(1, ...input.map((tweezer) => Math.abs(tweezer.xUm)));
  const maximumY = Math.max(1, ...input.map((tweezer) => Math.abs(tweezer.yUm)));
  const uniformScale = Math.min(
    fftWidth.value * 0.4 / maximumX,
    fftHeight.value * 0.4 / maximumY,
  );
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
      scaleX: uniformScale,
      scaleY: uniformScale,
    },
  };
}

function generateFrame(): void {
  if (running.value) return;
  const input = validatedTweezers();
  if (!input) return;
  terminateFrameWorker();
  const jobId = ++generation;
  running.value = true;
  outputState.value = "running";
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = 0;
  checksum.value = null;
  errorMessage.value = "";
  qualityWarning.value = "";
  startElapsedClock();

  const worker = new Worker(new URL("../workers/compiler.worker.ts", import.meta.url), { type: "module" });
  frameWorker = worker;
  worker.onmessage = (event: MessageEvent<CompilerWorkerResponse>) => {
    const response = event.data;
    if (response.jobId !== jobId || jobId !== generation) return;
    if (response.kind === "WORKER_ERROR") {
      rejectFrame(response.message, jobId, worker);
      return;
    }
    if (response.kind !== "TWEEZER_FRAME_RESULT") return;
    outputPixels.value = response.format === "UINT16" ? new Uint16Array(response.buffer) : new Uint8Array(response.buffer);
    metrics.value = response.metrics;
    elapsedMs.value = response.elapsedMs;
    checksum.value = response.checksum;
    resultBackendId.value = response.backendId;
    outputState.value = !response.metrics.accepted
      ? "rejected"
      : response.metrics.converged ? "accepted" : "warning";
    if (!response.metrics.accepted) {
      errorMessage.value = "The generated frame did not pass the numerical quality checks";
    } else if (!response.metrics.converged) {
      qualityWarning.value = "The WGS iteration budget ended before convergence. The frame is exportable, but its optical quality is not certified; review the metrics before use.";
    }
    running.value = false;
    stopElapsedClock(response.elapsedMs);
    disposeFrameWorker(worker);
  };
  worker.onerror = (event: ErrorEvent) => {
    if (jobId !== generation) return;
    rejectFrame(event.message || "SLM frame worker failed", jobId, worker);
  };
  const request: CompilerWorkerRequest = {
    kind: "GENERATE_TWEEZER_FRAME",
    jobId,
    input: {
      tweezers: input,
      slmWidth: slmWidth.value,
      slmHeight: slmHeight.value,
      fftWidth: fftWidth.value,
      fftHeight: fftHeight.value,
      iterations: iterations.value,
      backend: computeBackend.value,
    },
  };
  worker.postMessage(request);
}

function rejectFrame(message: string, jobId: number, worker: Worker): void {
  if (jobId !== generation) return;
  running.value = false;
  outputState.value = "rejected";
  errorMessage.value = message;
  stopElapsedClock();
  disposeFrameWorker(worker);
}

function cancelGeneration(): void {
  if (!running.value) return;
  generation += 1;
  terminateFrameWorker();
  stopElapsedClock();
  running.value = false;
  outputState.value = "cancelled";
  outputPixels.value = null;
  metrics.value = null;
  checksum.value = null;
  errorMessage.value = "";
  qualityWarning.value = "";
}

function startElapsedClock(): void {
  stopElapsedClock();
  calculationStarted = performance.now();
  elapsedMs.value = 0;
  elapsedTimer = window.setInterval(() => {
    elapsedMs.value = performance.now() - calculationStarted;
  }, 100);
}

function stopElapsedClock(finalElapsedMs?: number): void {
  window.clearInterval(elapsedTimer);
  elapsedTimer = 0;
  if (finalElapsedMs !== undefined) elapsedMs.value = finalElapsedMs;
}

function disposeFrameWorker(worker: Worker): void {
  worker.terminate();
  if (frameWorker === worker) frameWorker = null;
}

function terminateFrameWorker(): void {
  frameWorker?.terminate();
  frameWorker = null;
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

function exportBmp(): void {
  if (!outputPixels.value) return;
  const bmp = encodeGrayscaleBmp(outputPixels.value, slmWidth.value, slmHeight.value);
  download(
    new Blob([bmp.buffer as ArrayBuffer], { type: "image/bmp" }),
    `slm-frame-${slmWidth.value}x${slmHeight.value}.bmp`,
  );
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
      backend: resultBackendId.value,
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
  if (densePlotGeometry.value) {
    return 320 + (finiteOrZero(value) - densePlotGeometry.value.centerX) * densePlotGeometry.value.scale;
  }
  return 320 + finiteOrZero(value) / maximumPlotX.value * 255;
}

function plotY(value: number): number {
  if (densePlotGeometry.value) {
    return 160 - (finiteOrZero(value) - densePlotGeometry.value.centerY) * densePlotGeometry.value.scale;
  }
  return 160 - finiteOrZero(value) / maximumPlotY.value * 122;
}

function phaseColor(phase: number): string {
  const turn = ((finiteOrZero(phase) / (Math.PI * 2)) % 1 + 1) % 1;
  return `hsl(${Math.round(turn * 300 + 75)} 78% 68%)`;
}

function phaseDegrees(phase: number): string {
  return `${(finiteOrZero(phase) * 180 / Math.PI).toFixed(1)}°`;
}

onBeforeUnmount(() => {
  generation += 1;
  terminateFrameWorker();
  stopElapsedClock();
});

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
            <line x1="40" :y1="plotY(0)" x2="600" :y2="plotY(0)"></line>
            <line :x1="plotX(0)" y1="24" :x2="plotX(0)" y2="296"></line>
            <g v-for="(tweezer, index) in tweezers" :key="`${tweezer.trapId}:${index}`">
              <circle class="tweezer-phase-glow" :cx="plotX(tweezer.xUm)" :cy="plotY(tweezer.yUm)" :r="densePlotGeometry ? 4 : 15" :fill="phaseColor(tweezer.phaseRad)"></circle>
              <circle class="tweezer-phase-point" :cx="plotX(tweezer.xUm)" :cy="plotY(tweezer.yUm)" :r="densePlotGeometry ? 2 : 5" :fill="phaseColor(tweezer.phaseRad)"></circle>
              <text v-if="!densePlotGeometry" :x="plotX(tweezer.xUm) + 10" :y="plotY(tweezer.yUm) - 9">{{ tweezer.trapId }} / {{ phaseDegrees(tweezer.phaseRad) }}</text>
            </g>
          </svg>
          <div class="tweezer-plane-caption"><span>PHASE / HUE</span><span>UM PLANE / TOP VIEW</span></div>
        </div>

        <div class="data-subbar target-image-subbar tweezer-image-subbar">
          <span>IMAGE INPUT / OPTICAL TWEEZERS</span>
          <span>SPOT POWER → RELATIVE INTENSITY / PHASE 0</span>
        </div>
        <TargetImageImporter
          ref="targetImageImporter"
          destination="tweezers"
          :disabled="running"
          @apply="applyImageTweezers"
        />

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
          :running="running"
        />

        <section class="panel tweezer-generator" aria-labelledby="frame-generator-title">
          <div class="panel-bar">
            <span id="frame-generator-title" class="panel-kicker">FRAME GENERATOR</span>
            <span class="valid-badge" :class="{ 'is-warning': outputState === 'warning', 'is-rejected': outputState === 'rejected' }">{{ outputState.toUpperCase() }}</span>
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
            <label class="tweezer-backend-choice">COMPUTE BACKEND
              <select :value="computeBackend" :disabled="running" @change="updateComputeBackend">
                <option value="wasm">WebAssembly / CPU FFT</option>
                <option value="webgpu" :disabled="!webgpuAvailable">WebGPU / GPU-resident WGS</option>
              </select>
              <small :class="{ 'is-available': webgpuAvailable }">{{ webgpuStatus }}</small>
            </label>
            <div class="control-block">
              <div class="control-label"><span>WGS ITERATIONS</span><output>{{ String(iterations).padStart(2, "0") }} / FRAME</output></div>
              <input type="range" min="1" max="12" step="1" :value="iterations" :disabled="running" @input="updateIterations">
            </div>
            <ComputationActivity
              v-if="running"
              label="SYNTHESIZING PHASE FIELD"
              :detail="`${fftWidth} × ${fftHeight} FFT / DEDICATED WORKER`"
              :elapsed-ms="elapsedMs ?? 0"
              :progress="null"
            />
            <p v-if="errorMessage" class="tweezer-error" role="alert">{{ errorMessage }}</p>
            <p v-else-if="qualityWarning" class="tweezer-warning" role="status">{{ qualityWarning }}</p>
            <button class="compile-button tweezer-generate-button" :class="{ 'is-running': running }" type="button" @click="running ? cancelGeneration() : generateFrame()">
              <span></span>{{ running ? "Cancel generation" : "Generate SLM frame" }}
            </button>
            <div class="tweezer-export-actions">
              <button type="button" :disabled="!outputPixels || running" @click="exportRawFrame">Raw frame <b>&darr;</b></button>
              <button type="button" :disabled="!outputPixels || running" @click="exportBmp">BMP frame <b>&darr;</b></button>
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
