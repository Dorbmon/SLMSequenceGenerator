<script setup lang="ts">
import { computed, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import { mapPhysicalPointToDftFrequency } from "../../../src/index.js";
import ComputationActivity from "./ComputationActivity.vue";
import IntensityMapPreview from "./IntensityMapPreview.vue";
import { decodeGrayscaleBmp, encodeGrayscaleBmp } from "../lib/bmp.js";
import {
  forwardSimulationRegionAspect,
  shiftedForwardCoordinate,
  targetForwardSimulationRegion,
  type ForwardSimulationMetrics,
  type ForwardSimulationRegion,
} from "../lib/forward-simulation.js";
import type { OpticalCalibrationInput } from "../lib/optical-calibration.js";
import {
  createOpticalCalibration,
  opticalFieldOfViewUm,
} from "../lib/optical-calibration.js";
import {
  MAX_SLM_DIMENSION,
  MIN_SLM_DIMENSION,
  fftDimensionFor,
  normalizeSlmDimension,
} from "../lib/resolution.js";
import type { OpticalTweezerInput } from "../lib/tweezers.js";
import type { ComputeBackend } from "../workers/compiler-messages.js";
import type {
  ForwardSimulationWorkerRequest,
  ForwardSimulationWorkerResponse,
} from "../workers/forward-simulation-messages.js";

interface LoadedFrame {
  pixels: Uint8Array;
  width: number;
  height: number;
  name: string;
  source: "BMP" | "RAW U8" | "GENERATED";
}

const props = defineProps<{
  generatedPixels: Uint8Array | Uint16Array | null;
  generatedWidth: number;
  generatedHeight: number;
  webgpuAvailable: boolean;
  webgpuStatus: string;
  targets: readonly OpticalTweezerInput[];
  opticalCalibration: OpticalCalibrationInput;
}>();

const upload = ref<HTMLInputElement | null>(null);
const loadedFrame = shallowRef<LoadedFrame | null>(null);
const intensity = shallowRef<Float32Array | null>(null);
const metrics = shallowRef<ForwardSimulationMetrics | null>(null);
const backend = ref<ComputeBackend>("wasm");
const resultBackendId = ref("");
const logarithmic = ref(false);
const floorDb = ref(-60);
const viewMode = ref<"TARGETS" | "FULL">("TARGETS");
const running = ref(false);
const dragging = ref(false);
const elapsedMs = ref<number | null>(null);
const errorMessage = ref("");
let worker: Worker | null = null;
let jobGeneration = 0;
let elapsedTimer = 0;
let calculationStarted = 0;

const fftWidth = computed(() => loadedFrame.value ? fftDimensionFor(loadedFrame.value.width) : 0);
const fftHeight = computed(() => loadedFrame.value ? fftDimensionFor(loadedFrame.value.height) : 0);
const status = computed(() => {
  if (running.value) return backend.value === "webgpu" ? "PROPAGATING / WEBGPU" : "PROPAGATING / WASM";
  if (intensity.value) return "SIMULATION COMPLETE";
  if (loadedFrame.value) return "FRAME READY";
  return "AWAITING SLM FRAME";
});
const canUseGenerated = computed(() => (
  props.generatedPixels !== null && props.generatedPixels.length === props.generatedWidth * props.generatedHeight
));
const targetFrequencies = computed(() => {
  const frame = loadedFrame.value;
  if (!frame || props.targets.length === 0) return [];
  try {
    const calibration = createOpticalCalibration({
      activeWidth: frame.width,
      activeHeight: frame.height,
      fftWidth: fftWidth.value,
      fftHeight: fftHeight.value,
    }, props.opticalCalibration, "browser-forward-simulation-calibration", {
      includeIncidentAmplitude: false,
    });
    return props.targets.map((target) => mapPhysicalPointToDftFrequency(
      target,
      calibration,
      fftWidth.value,
      fftHeight.value,
    ));
  } catch {
    return [];
  }
});
const targetRegion = computed(() => {
  const frame = loadedFrame.value;
  if (!frame) return null;
  const beam = props.opticalCalibration.incidentBeam;
  const effectiveWidth = beam
    ? Math.max(1, Math.min(frame.width, Math.round(beam.diameterXMm * 1000 / props.opticalCalibration.pixelPitchUm)))
    : frame.width;
  const effectiveHeight = beam
    ? Math.max(1, Math.min(frame.height, Math.round(beam.diameterYMm * 1000 / props.opticalCalibration.pixelPitchUm)))
    : frame.height;
  return targetForwardSimulationRegion(
    targetFrequencies.value,
    fftWidth.value,
    fftHeight.value,
    effectiveWidth,
    effectiveHeight,
  );
});
const fullRegion = computed<ForwardSimulationRegion | null>(() => (
  fftWidth.value > 0 && fftHeight.value > 0
    ? { x: 0, y: 0, width: fftWidth.value, height: fftHeight.value }
    : null
));
const displayRegion = computed(() => (
  viewMode.value === "TARGETS" && targetRegion.value ? targetRegion.value : fullRegion.value
));
const displayAspectRatio = computed(() => {
  const region = displayRegion.value;
  return region ? forwardSimulationRegionAspect(region, fftWidth.value, fftHeight.value) : 16 / 9;
});
const targetMarkers = computed(() => targetFrequencies.value.map((frequency, index) => ({
  x: shiftedForwardCoordinate(frequency.x, fftWidth.value),
  y: shiftedForwardCoordinate(frequency.y, fftHeight.value),
  label: String(props.targets[index]?.trapId ?? index + 1),
})));
const displayFieldSummary = computed(() => {
  const region = displayRegion.value;
  if (!region) return "--";
  try {
    const fieldOfView = opticalFieldOfViewUm(props.opticalCalibration);
    const widthUm = region.width / fftWidth.value * fieldOfView;
    const heightUm = region.height / fftHeight.value * fieldOfView;
    return `${widthUm.toFixed(2)} × ${heightUm.toFixed(2)} µm`;
  } catch {
    return `${region.width} × ${region.height} bins`;
  }
});

watch(() => props.webgpuAvailable, (available) => {
  if (!available && backend.value === "webgpu") backend.value = "wasm";
});
watch(() => props.generatedPixels, (generated) => {
  if (generated !== null || loadedFrame.value?.source !== "GENERATED") return;
  cancelSimulation(false);
  loadedFrame.value = null;
  intensity.value = null;
  metrics.value = null;
  resultBackendId.value = "";
  elapsedMs.value = null;
});

async function loadFrame(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (file) await loadFile(file);
}

async function loadFile(file: File): Promise<void> {
  if (running.value) return;
  try {
    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    const isBmp = bytes.length >= 2 && bytes[0] === 0x42 && bytes[1] === 0x4d;
    if (isBmp) {
      const decoded = decodeGrayscaleBmp(bytes);
      validateFrameDimensions(decoded.width, decoded.height);
      setLoadedFrame({ ...decoded, name: file.name, source: "BMP" });
      return;
    }
    const expected = props.generatedWidth * props.generatedHeight;
    if (bytes.length !== expected) {
      throw new Error(`Raw U8 frames must contain exactly ${expected.toLocaleString()} bytes for the current ${props.generatedWidth} × ${props.generatedHeight} SLM resolution`);
    }
    validateFrameDimensions(props.generatedWidth, props.generatedHeight);
    setLoadedFrame({ pixels: new Uint8Array(bytes), width: props.generatedWidth, height: props.generatedHeight, name: file.name, source: "RAW U8" });
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Unable to load the SLM frame";
  }
}

function useGeneratedFrame(): void {
  const source = props.generatedPixels;
  if (!source || !canUseGenerated.value || running.value) return;
  const pixels = new Uint8Array(source.length);
  if (source instanceof Uint16Array) {
    for (let index = 0; index < source.length; index += 1) pixels[index] = Math.round(source[index]! / 257);
  } else {
    pixels.set(source);
  }
  setLoadedFrame({
    pixels,
    width: props.generatedWidth,
    height: props.generatedHeight,
    name: `generated-${props.generatedWidth}x${props.generatedHeight}.bmp`,
    source: "GENERATED",
  });
}

function setLoadedFrame(frame: LoadedFrame): void {
  cancelSimulation(false);
  loadedFrame.value = frame;
  intensity.value = null;
  metrics.value = null;
  resultBackendId.value = "";
  elapsedMs.value = null;
  errorMessage.value = "";
  viewMode.value = props.targets.length > 0 ? "TARGETS" : "FULL";
}

function validateFrameDimensions(width: number, height: number): void {
  normalizeSlmDimension(width, "SLM frame width");
  normalizeSlmDimension(height, "SLM frame height");
}

function simulate(): void {
  const frame = loadedFrame.value;
  if (!frame || running.value) return;
  terminateWorker();
  const jobId = ++jobGeneration;
  const copiedPixels = new Uint8Array(frame.pixels);
  intensity.value = null;
  metrics.value = null;
  resultBackendId.value = "";
  errorMessage.value = "";
  running.value = true;
  startElapsedClock();

  const nextWorker = new Worker(new URL("../workers/forward-simulation.worker.ts", import.meta.url), { type: "module" });
  worker = nextWorker;
  nextWorker.onmessage = (event: MessageEvent<ForwardSimulationWorkerResponse>) => {
    const response = event.data;
    if (response.jobId !== jobId || jobId !== jobGeneration) return;
    if (response.kind === "FORWARD_SIMULATION_ERROR") {
      rejectSimulation(response.message, nextWorker, jobId);
      return;
    }
    intensity.value = new Float32Array(response.intensity);
    metrics.value = response.metrics;
    resultBackendId.value = response.backendId;
    running.value = false;
    stopElapsedClock(response.elapsedMs);
    disposeWorker(nextWorker);
  };
  nextWorker.onerror = (event: ErrorEvent) => {
    if (jobId === jobGeneration) rejectSimulation(event.message || "Forward-simulation worker failed", nextWorker, jobId);
  };
  const request: ForwardSimulationWorkerRequest = {
    kind: "SIMULATE_SLM_FRAME",
    jobId,
    input: {
      pixels: copiedPixels.buffer,
      width: frame.width,
      height: frame.height,
      fftWidth: fftWidth.value,
      fftHeight: fftHeight.value,
      backend: backend.value,
      pixelPitchUm: props.opticalCalibration.pixelPitchUm,
      ...(props.opticalCalibration.incidentBeam
        ? { incidentBeam: { ...props.opticalCalibration.incidentBeam } }
        : {}),
      ...(props.opticalCalibration.phaseResponseLut
        ? { phaseResponseLut: [...props.opticalCalibration.phaseResponseLut] }
        : {}),
    },
  };
  nextWorker.postMessage(request, [request.input.pixels]);
}

function rejectSimulation(message: string, target: Worker, jobId: number): void {
  if (jobId !== jobGeneration) return;
  running.value = false;
  errorMessage.value = message;
  stopElapsedClock();
  disposeWorker(target);
}

function cancelSimulation(markCancelled = true): void {
  if (!running.value && !worker) return;
  jobGeneration += 1;
  terminateWorker();
  stopElapsedClock();
  running.value = false;
  if (markCancelled) errorMessage.value = "Forward simulation cancelled";
}

function updateBackend(event: Event): void {
  const value = (event.target as HTMLSelectElement).value as ComputeBackend;
  if (value === "webgpu" && !props.webgpuAvailable) return;
  backend.value = value;
}

function exportIntensityBmp(): void {
  const values = intensity.value;
  const result = metrics.value;
  if (!values || !result) return;
  const pixels = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    const display = logarithmic.value
      ? clamp((10 * Math.log10(Math.max(values[index]!, 1e-12)) - floorDb.value) / -floorDb.value, 0, 1)
      : clamp(values[index]!, 0, 1);
    pixels[index] = Math.round(display * 255);
  }
  const bmp = encodeGrayscaleBmp(pixels, result.fftWidth, result.fftHeight);
  download(new Blob([bmp.buffer as ArrayBuffer], { type: "image/bmp" }), `simulated-intensity-${result.fftWidth}x${result.fftHeight}.bmp`);
}

function exportIntensityRaw(): void {
  const values = intensity.value;
  const result = metrics.value;
  if (!values || !result) return;
  const copy = new Float32Array(values);
  download(new Blob([copy.buffer], { type: "application/octet-stream" }), `simulated-intensity-${result.fftWidth}x${result.fftHeight}-f32.raw`);
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function handleDragOver(event: DragEvent): void {
  event.preventDefault();
  if (!running.value) dragging.value = true;
}

function handleDragLeave(event: DragEvent): void {
  if (!(event.currentTarget as HTMLElement).contains(event.relatedTarget as Node | null)) dragging.value = false;
}

function handleDrop(event: DragEvent): void {
  event.preventDefault();
  dragging.value = false;
  const file = event.dataTransfer?.files[0];
  if (file) void loadFile(file);
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

function disposeWorker(target: Worker): void {
  target.terminate();
  if (worker === target) worker = null;
}

function terminateWorker(): void {
  worker?.terminate();
  worker = null;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function reset(): void {
  cancelSimulation(false);
  loadedFrame.value = null;
  intensity.value = null;
  metrics.value = null;
  backend.value = "wasm";
  resultBackendId.value = "";
  logarithmic.value = false;
  floorDb.value = -60;
  viewMode.value = "TARGETS";
  elapsedMs.value = null;
  errorMessage.value = "";
  dragging.value = false;
}

onBeforeUnmount(() => {
  jobGeneration += 1;
  terminateWorker();
  stopElapsedClock();
});

defineExpose({ reset });
</script>

<template>
  <section class="panel forward-simulator" aria-labelledby="forward-simulator-title">
    <div class="panel-bar">
      <span id="forward-simulator-title" class="panel-kicker">FORWARD SIMULATOR / SLM → FOCAL PLANE</span>
      <span>{{ status }}</span>
    </div>
    <div class="forward-simulator-grid">
      <div class="forward-simulator-controls">
        <div class="forward-simulator-intro">
          <p class="eyebrow">Independent propagation</p>
          <h2>Simulate an SLM frame</h2>
          <p>Upload an 8-bit BMP or raw U8 phase-code frame. The simulator applies a uniform rectangular aperture, performs a centered forward FFT, and displays normalized focal-plane intensity.</p>
        </div>

        <div
          class="forward-upload"
          :class="{ 'is-dragging': dragging, 'has-frame': loadedFrame }"
          @dragover="handleDragOver"
          @dragleave="handleDragLeave"
          @drop="handleDrop"
        >
          <div class="forward-upload-mark" aria-hidden="true"><i></i><i></i><b>→</b></div>
          <div v-if="loadedFrame" class="forward-file-summary">
            <small>{{ loadedFrame.source }} / PHASE CODES</small>
            <strong>{{ loadedFrame.name }}</strong>
            <span>{{ loadedFrame.width }} × {{ loadedFrame.height }} px → FFT {{ fftWidth }} × {{ fftHeight }}</span>
          </div>
          <div v-else class="forward-file-summary">
            <small>SLM FRAME INPUT</small>
            <strong>Drop a BMP or raw U8 frame</strong>
            <span>Raw files use the current {{ generatedWidth }} × {{ generatedHeight }} SLM resolution.</span>
          </div>
          <button type="button" :disabled="running" @click="upload?.click()">{{ loadedFrame ? "Replace" : "Choose frame" }}</button>
          <input ref="upload" class="sr-only" type="file" accept="image/bmp,.bmp,.bin,.raw,application/octet-stream" @change="loadFrame">
        </div>

        <button class="forward-use-generated" type="button" :disabled="running || !canUseGenerated" @click="useGeneratedFrame">
          Use generated SLM frame
          <span>{{ canUseGenerated ? `${generatedWidth} × ${generatedHeight}` : "Generate a frame first" }}</span>
        </button>

        <label class="tweezer-backend-choice">SIMULATION BACKEND
          <select :value="backend" :disabled="running" @change="updateBackend">
            <option value="wasm">WebAssembly / CPU FFT</option>
            <option value="webgpu" :disabled="!webgpuAvailable">WebGPU / GPU-resident propagation</option>
          </select>
          <small :class="{ 'is-available': webgpuAvailable }">{{ webgpuStatus }}</small>
        </label>

        <div class="forward-display-controls">
          <label>FOCAL-PLANE VIEW
            <select v-model="viewMode">
              <option value="TARGETS" :disabled="!targetRegion">Current target region</option>
              <option value="FULL">Full optical field</option>
            </select>
          </label>
          <p class="forward-view-summary">{{ viewMode === "TARGETS" && targetRegion ? `${targetMarkers.length} TARGETS / ${displayFieldSummary}` : `FULL FIELD / ${displayFieldSummary}` }}</p>
          <label>INTENSITY DISPLAY
            <select v-model="logarithmic">
              <option :value="true">Logarithmic / dB</option>
              <option :value="false">Linear</option>
            </select>
          </label>
          <div v-if="logarithmic" class="control-block">
            <div class="control-label"><span>DISPLAY FLOOR</span><output>{{ floorDb }} dB</output></div>
            <input v-model.number="floorDb" type="range" min="-100" max="-10" step="1">
          </div>
        </div>

        <ComputationActivity
          v-if="running"
          label="PROPAGATING SLM FIELD"
          :detail="`${fftWidth} × ${fftHeight} FFT / DEDICATED WORKER`"
          :elapsed-ms="elapsedMs ?? 0"
          :progress="null"
        />
        <p v-if="errorMessage" class="tweezer-error" role="alert">{{ errorMessage }}</p>
        <button class="compile-button forward-simulate-button" :class="{ 'is-running': running }" type="button" :disabled="!loadedFrame && !running" @click="running ? cancelSimulation() : simulate()">
          <span></span>{{ running ? "Cancel simulation" : "Simulate intensity map" }}
        </button>
        <div class="forward-export-actions">
          <button type="button" :disabled="!intensity || running" @click="exportIntensityBmp">Display BMP <b>&darr;</b></button>
          <button type="button" :disabled="!intensity || running" @click="exportIntensityRaw">Normalized F32 <b>&darr;</b></button>
        </div>
        <p class="forward-model-note">MODEL / {{ opticalCalibration.phaseResponseLut ? "DEVICE-READY PHASE RESPONSE" : "SLMCONTROL3-CORRECTED 0–2π" }} · {{ opticalCalibration.incidentBeam ? `${opticalCalibration.incidentBeam.diameterXMm}×${opticalCalibration.incidentBeam.diameterYMm} MM GAUSSIAN BEAM` : "UNIFORM ILLUMINATION" }} · NO MEASURED ABERRATION</p>
      </div>

      <IntensityMapPreview
        :intensity="intensity"
        :width="metrics?.fftWidth ?? fftWidth"
        :height="metrics?.fftHeight ?? fftHeight"
        :running="running"
        :status="status"
        :logarithmic="logarithmic"
        :floor-db="floorDb"
        :region="displayRegion"
        :target-markers="viewMode === 'TARGETS' ? targetMarkers : []"
        :physical-aspect-ratio="displayAspectRatio"
        :view-label="viewMode === 'TARGETS' && targetRegion ? 'CALIBRATED TARGET REGION' : 'FULL OPTICAL FIELD'"
      />
    </div>

    <div class="forward-metrics" aria-label="Forward simulation metrics">
      <div><small>PROPAGATION TIME</small><strong>{{ elapsedMs === null ? "--" : `${(elapsedMs / 1000).toFixed(2)}s` }}</strong></div>
      <div><small>PEAK / FFT OFFSET</small><strong>{{ metrics ? `${metrics.peakOffsetX}, ${metrics.peakOffsetY}` : "--" }}</strong></div>
      <div><small>ZERO ORDER / PEAK</small><strong>{{ metrics ? `${(metrics.zeroOrderRelativeIntensity * 100).toFixed(2)}%` : "--" }}</strong></div>
      <div><small>BACKEND</small><strong>{{ resultBackendId ? (resultBackendId.startsWith("webgpu") ? "WEBGPU" : "WASM") : "--" }}</strong></div>
    </div>
  </section>
</template>
