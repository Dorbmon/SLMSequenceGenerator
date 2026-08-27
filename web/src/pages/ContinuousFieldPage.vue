<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import ComputationActivity from "../components/ComputationActivity.vue";
import ContinuousTargetEditor, {
  type ContinuousTargetChange,
} from "../components/ContinuousTargetEditor.vue";
import IntensityMapPreview from "../components/IntensityMapPreview.vue";
import SlmFramePreview from "../components/SlmFramePreview.vue";
import { encodeGrayscaleBmp } from "../lib/bmp.js";
import {
  DEFAULT_CONTINUOUS_FIELD_ITERATIONS,
  DEFAULT_CONTINUOUS_FIELD_MIXING,
  type ContinuousFieldMetrics,
  type ContinuousFieldRegion,
} from "../lib/continuous-field.js";
import {
  DEFAULT_FOCAL_LENGTH_MM,
  DEFAULT_INCIDENT_BEAM_DIAMETER_MM,
  DEFAULT_PIXEL_PITCH_UM,
  DEFAULT_TWO_PI_SIGNAL_LEVEL,
  DEFAULT_WAVELENGTH_NM,
  opticalFieldOfViewUm,
  parsePhaseResponseLut,
  phaseResponseForTwoPiSignalLevel,
  type OpticalCalibrationInput,
} from "../lib/optical-calibration.js";
import {
  DEFAULT_SLM_HEIGHT,
  DEFAULT_SLM_WIDTH,
  MAX_SLM_DIMENSION,
  MIN_SLM_DIMENSION,
  fftDimensionFor,
  normalizeSlmDimension,
} from "../lib/resolution.js";
import type { ComputeBackend } from "../workers/compiler-messages.js";
import type {
  ContinuousFieldWorkerRequest,
  ContinuousFieldWorkerResponse,
} from "../workers/continuous-field-messages.js";

type OutputState = "idle" | "running" | "ready" | "warning" | "rejected" | "cancelled";
type LutApplicationMode = "SLMCONTROL3" | "BROWSER";

interface ContinuousTargetEditorHandle {
  reset(): void;
}

const props = defineProps<{
  webgpuAvailable: boolean;
  webgpuStatus: string;
}>();

const editor = ref<ContinuousTargetEditorHandle | null>(null);
const calibrationUpload = ref<HTMLInputElement | null>(null);
const target = shallowRef<Float32Array>(new Float32Array());
const targetWidth = ref(0);
const targetHeight = ref(0);
const slmWidth = ref(DEFAULT_SLM_WIDTH);
const slmHeight = ref(DEFAULT_SLM_HEIGHT);
const iterations = ref(DEFAULT_CONTINUOUS_FIELD_ITERATIONS);
const mixingFactor = ref(DEFAULT_CONTINUOUS_FIELD_MIXING);
const deterministicSeed = ref(1);
const fieldWidthUm = ref(600);
const fieldHeightUm = ref(375);
const fieldCenterXUm = ref(0);
const fieldCenterYUm = ref(0);
const wavelengthNm = ref(DEFAULT_WAVELENGTH_NM);
const focalLengthMm = ref(DEFAULT_FOCAL_LENGTH_MM);
const pixelPitchUm = ref(DEFAULT_PIXEL_PITCH_UM);
const beamDiameterXMm = ref(DEFAULT_INCIDENT_BEAM_DIAMETER_MM);
const beamDiameterYMm = ref(DEFAULT_INCIDENT_BEAM_DIAMETER_MM);
const beamCenterXMm = ref(0);
const beamCenterYMm = ref(0);
const lutApplicationMode = ref<LutApplicationMode>("SLMCONTROL3");
const twoPiSignalLevel = ref(DEFAULT_TWO_PI_SIGNAL_LEVEL);
const phaseResponseLut = shallowRef<number[] | undefined>();
const phaseResponseFilename = ref("");
const computeBackend = ref<ComputeBackend>("wasm");
const backendWasChosen = ref(false);
const running = ref(false);
const outputState = ref<OutputState>("idle");
const outputPixels = shallowRef<Uint8Array | null>(null);
const outputIntensity = shallowRef<Float32Array | null>(null);
const outputRegion = shallowRef<ContinuousFieldRegion | null>(null);
const metrics = shallowRef<ContinuousFieldMetrics | null>(null);
const resultBackendId = ref("");
const checksum = ref<number | null>(null);
const elapsedMs = ref<number | null>(null);
const errorMessage = ref("");
const warningMessage = ref("");
let generation = 0;
let worker: Worker | null = null;
let timer = 0;
let started = 0;

const fftWidth = computed(() => fftDimensionFor(slmWidth.value));
const fftHeight = computed(() => fftDimensionFor(slmHeight.value));
const effectivePhaseResponseLut = computed(() => {
  if (lutApplicationMode.value === "SLMCONTROL3") return undefined;
  if (phaseResponseLut.value) return phaseResponseLut.value;
  try {
    return phaseResponseForTwoPiSignalLevel(twoPiSignalLevel.value);
  } catch {
    return undefined;
  }
});
const opticalCalibration = computed<OpticalCalibrationInput>(() => ({
  wavelengthNm: wavelengthNm.value,
  focalLengthMm: focalLengthMm.value,
  pixelPitchUm: pixelPitchUm.value,
  incidentBeam: {
    profile: "GAUSSIAN",
    diameterXMm: beamDiameterXMm.value,
    diameterYMm: beamDiameterYMm.value,
    centerXMm: beamCenterXMm.value,
    centerYMm: beamCenterYMm.value,
  },
  ...(effectivePhaseResponseLut.value ? { phaseResponseLut: [...effectivePhaseResponseLut.value] } : {}),
}));
const fieldOfViewUm = computed(() => {
  try {
    return opticalFieldOfViewUm(opticalCalibration.value);
  } catch {
    return null;
  }
});
const targetAspectRatio = computed(() => (
  Number.isFinite(fieldWidthUm.value) && Number.isFinite(fieldHeightUm.value) && fieldWidthUm.value > 0 && fieldHeightUm.value > 0
    ? fieldWidthUm.value / fieldHeightUm.value
    : 1
));
const phaseStatus = computed(() => {
  if (running.value) return computeBackend.value === "webgpu" ? "Computing on WebGPU" : "Computing in Wasm worker";
  if (outputState.value === "ready") return "Frame ready";
  if (outputState.value === "warning") return "Frame ready / review fit";
  if (outputState.value === "rejected") return "Generation failed";
  if (outputState.value === "cancelled") return "Generation cancelled";
  return "Ready to generate";
});
const reconstructionStatus = computed(() => {
  if (running.value) return "Propagating quantized frame";
  if (metrics.value) return `Correlation ${metrics.value.intensityCorrelation.toFixed(4)}`;
  return "Forward validation appears here";
});

watch(() => props.webgpuAvailable, (available) => {
  if (available && !backendWasChosen.value && !outputPixels.value) computeBackend.value = "webgpu";
  if (!available && computeBackend.value === "webgpu") computeBackend.value = "wasm";
});

function updateTarget(change: ContinuousTargetChange): void {
  target.value = change.intensity;
  targetWidth.value = change.width;
  targetHeight.value = change.height;
  invalidate();
}

function updateDimension(event: Event, dimension: "width" | "height"): void {
  const element = event.target as HTMLInputElement;
  const current = dimension === "width" ? slmWidth.value : slmHeight.value;
  try {
    const value = normalizeSlmDimension(element.valueAsNumber, dimension === "width" ? "SLM width" : "SLM height");
    if (dimension === "width") slmWidth.value = value;
    else slmHeight.value = value;
    invalidate();
  } catch (error) {
    element.value = String(current);
    errorMessage.value = error instanceof Error ? error.message : "Invalid SLM resolution";
  }
}

function chooseBackend(event: Event): void {
  const value = (event.target as HTMLSelectElement).value as ComputeBackend;
  if (value === "webgpu" && !props.webgpuAvailable) return;
  computeBackend.value = value;
  backendWasChosen.value = true;
  invalidate();
}

function settingsChanged(): void {
  invalidate();
}

function invalidate(): void {
  generation += 1;
  terminateWorker();
  stopClock();
  running.value = false;
  outputState.value = "idle";
  outputPixels.value = null;
  outputIntensity.value = null;
  outputRegion.value = null;
  metrics.value = null;
  checksum.value = null;
  elapsedMs.value = null;
  errorMessage.value = "";
  warningMessage.value = "";
}

function generate(): void {
  if (running.value) {
    cancel();
    return;
  }
  if (target.value.length !== targetWidth.value * targetHeight.value) {
    fail("Draw or upload a target before generating");
    return;
  }
  if (lutApplicationMode.value === "BROWSER" && !phaseResponseLut.value) {
    try {
      phaseResponseForTwoPiSignalLevel(twoPiSignalLevel.value);
    } catch (error) {
      fail(error instanceof Error ? error.message : "Invalid 2π signal level");
      return;
    }
  }
  terminateWorker();
  const jobId = ++generation;
  running.value = true;
  outputState.value = "running";
  outputPixels.value = null;
  outputIntensity.value = null;
  outputRegion.value = null;
  metrics.value = null;
  checksum.value = null;
  errorMessage.value = "";
  warningMessage.value = "";
  startClock();
  const activeWorker = new Worker(new URL("../workers/continuous-field.worker.ts", import.meta.url), { type: "module" });
  worker = activeWorker;
  activeWorker.onmessage = (event: MessageEvent<ContinuousFieldWorkerResponse>) => {
    const response = event.data;
    if (activeWorker !== worker || response.jobId !== jobId || jobId !== generation) return;
    if (response.kind === "WORKER_ERROR") {
      fail(response.message, activeWorker);
      return;
    }
    outputPixels.value = new Uint8Array(response.pixels);
    outputIntensity.value = new Float32Array(response.intensity);
    outputRegion.value = response.targetRegion;
    metrics.value = response.metrics;
    resultBackendId.value = response.backendId;
    checksum.value = response.checksum;
    running.value = false;
    stopClock(response.metrics.solveTimeMs);
    if (!response.metrics.numericalValid) {
      outputState.value = "rejected";
      errorMessage.value = "The generated frame contains invalid numerical output and cannot be exported.";
    } else if (response.metrics.accepted) {
      outputState.value = "ready";
    } else {
      outputState.value = "warning";
      warningMessage.value = `The frame is exportable. Its normalized target-fit error is ${(response.metrics.normalizedIntensityRmse * 100).toFixed(2)}%; compare the reconstructed map before use.`;
    }
    disposeWorker(activeWorker);
  };
  activeWorker.onerror = (event: ErrorEvent) => {
    if (activeWorker !== worker || jobId !== generation) return;
    fail(event.message || "Continuous-field worker failed", activeWorker);
  };
  const targetCopy = new Float32Array(target.value).buffer;
  const request: ContinuousFieldWorkerRequest = {
    kind: "GENERATE_CONTINUOUS_FIELD",
    jobId,
    input: {
      targetBuffer: targetCopy,
      targetWidth: targetWidth.value,
      targetHeight: targetHeight.value,
      slmWidth: slmWidth.value,
      slmHeight: slmHeight.value,
      fftWidth: fftWidth.value,
      fftHeight: fftHeight.value,
      fieldWidthUm: fieldWidthUm.value,
      fieldHeightUm: fieldHeightUm.value,
      fieldCenterXUm: fieldCenterXUm.value,
      fieldCenterYUm: fieldCenterYUm.value,
      iterations: iterations.value,
      mixingFactor: mixingFactor.value,
      deterministicSeed: deterministicSeed.value,
      backend: computeBackend.value,
      opticalCalibration: opticalCalibration.value,
    },
  };
  activeWorker.postMessage(request, [targetCopy]);
}

function fail(message: string, sourceWorker?: Worker): void {
  running.value = false;
  outputState.value = "rejected";
  outputPixels.value = null;
  outputIntensity.value = null;
  outputRegion.value = null;
  metrics.value = null;
  errorMessage.value = message;
  warningMessage.value = "";
  stopClock();
  if (sourceWorker) disposeWorker(sourceWorker);
}

function cancel(): void {
  if (!running.value) return;
  generation += 1;
  terminateWorker();
  stopClock();
  running.value = false;
  outputState.value = "cancelled";
  errorMessage.value = "";
  warningMessage.value = "";
}

function startClock(): void {
  stopClock();
  started = performance.now();
  elapsedMs.value = 0;
  timer = window.setInterval(() => { elapsedMs.value = performance.now() - started; }, 100);
}

function stopClock(finalValue?: number): void {
  window.clearInterval(timer);
  timer = 0;
  if (finalValue !== undefined) elapsedMs.value = finalValue;
}

function disposeWorker(source: Worker): void {
  source.terminate();
  if (worker === source) worker = null;
}

function terminateWorker(): void {
  worker?.terminate();
  worker = null;
}

async function loadPhaseResponse(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  try {
    phaseResponseLut.value = parsePhaseResponseLut(await file.text());
    phaseResponseFilename.value = file.name;
    errorMessage.value = "";
    invalidate();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Unable to load the phase-response LUT";
  }
}

function clearPhaseResponse(): void {
  phaseResponseLut.value = undefined;
  phaseResponseFilename.value = "";
  invalidate();
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportBmp(): void {
  if (!outputPixels.value || !metrics.value?.numericalValid) return;
  const bmp = encodeGrayscaleBmp(outputPixels.value, slmWidth.value, slmHeight.value);
  download(new Blob([bmp.buffer as ArrayBuffer], { type: "image/bmp" }), `continuous-field-${slmWidth.value}x${slmHeight.value}.bmp`);
}

function exportMetadata(): void {
  if (!outputPixels.value || !metrics.value) return;
  const payload = {
    formatVersion: "1.0",
    output: {
      width: slmWidth.value,
      height: slmHeight.value,
      pixelFormat: "UINT8",
      crc32: checksum.value,
    },
    target: {
      sourceRaster: { width: targetWidth.value, height: targetHeight.value },
      focalPlane: {
        widthUm: fieldWidthUm.value,
        heightUm: fieldHeightUm.value,
        centerXUm: fieldCenterXUm.value,
        centerYUm: fieldCenterYUm.value,
      },
      fftRegion: outputRegion.value,
    },
    solver: {
      algorithm: "mixed-region amplitude freedom (MRAF)",
      backend: resultBackendId.value,
      iterations: iterations.value,
      mixingFactor: mixingFactor.value,
      deterministicSeed: deterministicSeed.value,
      elapsedMs: elapsedMs.value,
    },
    calibration: {
      ...opticalCalibration.value,
      phaseCodePipeline: lutApplicationMode.value === "SLMCONTROL3"
        ? "logical 0-255; LUT applied by SLMControl3"
        : "device-ready phase response baked into BMP",
      measuredPhaseResponseFilename: phaseResponseFilename.value || null,
    },
    metrics: metrics.value,
  };
  download(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }), "continuous-field-metadata.json");
}

function reset(): void {
  generation += 1;
  terminateWorker();
  stopClock();
  slmWidth.value = DEFAULT_SLM_WIDTH;
  slmHeight.value = DEFAULT_SLM_HEIGHT;
  iterations.value = DEFAULT_CONTINUOUS_FIELD_ITERATIONS;
  mixingFactor.value = DEFAULT_CONTINUOUS_FIELD_MIXING;
  deterministicSeed.value = 1;
  fieldWidthUm.value = 600;
  fieldHeightUm.value = 375;
  fieldCenterXUm.value = 0;
  fieldCenterYUm.value = 0;
  wavelengthNm.value = DEFAULT_WAVELENGTH_NM;
  focalLengthMm.value = DEFAULT_FOCAL_LENGTH_MM;
  pixelPitchUm.value = DEFAULT_PIXEL_PITCH_UM;
  beamDiameterXMm.value = DEFAULT_INCIDENT_BEAM_DIAMETER_MM;
  beamDiameterYMm.value = DEFAULT_INCIDENT_BEAM_DIAMETER_MM;
  beamCenterXMm.value = 0;
  beamCenterYMm.value = 0;
  lutApplicationMode.value = "SLMCONTROL3";
  twoPiSignalLevel.value = DEFAULT_TWO_PI_SIGNAL_LEVEL;
  phaseResponseLut.value = undefined;
  phaseResponseFilename.value = "";
  computeBackend.value = props.webgpuAvailable ? "webgpu" : "wasm";
  backendWasChosen.value = false;
  running.value = false;
  outputState.value = "idle";
  outputPixels.value = null;
  outputIntensity.value = null;
  outputRegion.value = null;
  metrics.value = null;
  checksum.value = null;
  elapsedMs.value = null;
  errorMessage.value = "";
  warningMessage.value = "";
  nextTick(() => editor.value?.reset());
}

onBeforeUnmount(() => {
  generation += 1;
  terminateWorker();
  stopClock();
});

defineExpose({ reset });
</script>

<template>
  <section class="workspace continuous-workspace">
    <div class="workspace-heading">
      <div>
        <p class="eyebrow">Continuous field</p>
        <h1>Draw an intensity target</h1>
      </div>
      <p class="heading-note">Image or brush input<br>phase-only MRAF output.</p>
    </div>

    <div class="continuous-input-grid">
      <ContinuousTargetEditor ref="editor" :disabled="running" @change="updateTarget" />

      <section class="panel continuous-settings" aria-labelledby="continuous-settings-title">
        <div class="panel-bar">
          <span id="continuous-settings-title">Generation settings</span>
          <span :class="['valid-badge', { 'is-warning': outputState === 'warning', 'is-rejected': outputState === 'rejected' }]">{{ outputState }}</span>
        </div>
        <div class="continuous-settings-body">
          <div class="resolution-block compact-form-block">
            <div class="control-label"><span>SLM resolution</span><output>{{ slmWidth }} × {{ slmHeight }}</output></div>
            <div class="resolution-fields">
              <label>Width / px
                <input type="number" :min="MIN_SLM_DIMENSION" :max="MAX_SLM_DIMENSION" :value="slmWidth" :disabled="running" @change="updateDimension($event, 'width')">
              </label>
              <span>×</span>
              <label>Height / px
                <input type="number" :min="MIN_SLM_DIMENSION" :max="MAX_SLM_DIMENSION" :value="slmHeight" :disabled="running" @change="updateDimension($event, 'height')">
              </label>
            </div>
            <p class="resolution-note">FFT {{ fftWidth }} × {{ fftHeight }}</p>
          </div>

          <div class="continuous-number-grid">
            <label>Field width / µm<input v-model.number="fieldWidthUm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
            <label>Field height / µm<input v-model.number="fieldHeightUm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
            <label>Centre X / µm<input v-model.number="fieldCenterXUm" type="number" step="any" :disabled="running" @change="settingsChanged"></label>
            <label>Centre Y / µm<input v-model.number="fieldCenterYUm" type="number" step="any" :disabled="running" @change="settingsChanged"></label>
          </div>
          <p class="resolution-note">Calibrated focal-plane FOV: {{ fieldOfViewUm === null ? "invalid" : `${fieldOfViewUm.toFixed(1)} × ${fieldOfViewUm.toFixed(1)} µm` }}</p>

          <label class="tweezer-backend-choice">Compute backend
            <select :value="computeBackend" :disabled="running" @change="chooseBackend">
              <option value="wasm">WebAssembly / worker</option>
              <option value="webgpu" :disabled="!webgpuAvailable">WebGPU / GPU-resident</option>
            </select>
            <small :class="{ 'is-available': webgpuAvailable }">{{ webgpuStatus }}</small>
          </label>

          <div class="continuous-slider-grid">
            <label>Iterations <output>{{ iterations }}</output><input v-model.number="iterations" type="range" min="1" max="64" step="1" :disabled="running" @change="settingsChanged"></label>
            <label>MRAF mixing <output>{{ mixingFactor.toFixed(2) }}</output><input v-model.number="mixingFactor" type="range" min="0.05" max="1" step="0.05" :disabled="running" @change="settingsChanged"></label>
          </div>
          <label class="continuous-seed">Deterministic seed<input v-model.number="deterministicSeed" type="number" min="0" max="4294967295" step="1" :disabled="running" @change="settingsChanged"></label>

          <details class="continuous-calibration">
            <summary>Optical calibration and LUT</summary>
            <div class="continuous-number-grid">
              <label>Wavelength / nm<input v-model.number="wavelengthNm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Focal length / mm<input v-model.number="focalLengthMm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Pixel pitch / µm<input v-model.number="pixelPitchUm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Beam diameter X / mm<input v-model.number="beamDiameterXMm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Beam diameter Y / mm<input v-model.number="beamDiameterYMm" type="number" min="0.001" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Beam centre X / mm<input v-model.number="beamCenterXMm" type="number" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Beam centre Y / mm<input v-model.number="beamCenterYMm" type="number" step="any" :disabled="running" @change="settingsChanged"></label>
              <label>Phase-code pipeline
                <select v-model="lutApplicationMode" :disabled="running" @change="settingsChanged">
                  <option value="SLMCONTROL3">SLMControl3 applies LUT</option>
                  <option value="BROWSER">Bake LUT into BMP</option>
                </select>
              </label>
              <label v-if="lutApplicationMode === 'BROWSER'">2π signal level<input v-model.number="twoPiSignalLevel" type="number" min="1" max="255" step="1" :disabled="running" @change="settingsChanged"></label>
            </div>
            <div v-if="lutApplicationMode === 'BROWSER'" class="continuous-lut-actions">
              <button type="button" :disabled="running" @click="calibrationUpload?.click()">{{ phaseResponseLut ? "Replace LUT" : "Upload measured LUT" }}</button>
              <button v-if="phaseResponseLut" type="button" :disabled="running" @click="clearPhaseResponse">Use 2π level</button>
              <span>{{ phaseResponseFilename }}</span>
            </div>
            <input ref="calibrationUpload" type="file" accept="application/json,.json,.csv,.txt,text/plain,text/csv" hidden @change="loadPhaseResponse">
          </details>

          <ComputationActivity
            v-if="running"
            label="Synthesizing continuous field"
            :detail="`${fftWidth} × ${fftHeight} MRAF / dedicated worker`"
            :elapsed-ms="elapsedMs ?? 0"
            :progress="null"
          />
          <p v-if="errorMessage" class="tweezer-error" role="alert">{{ errorMessage }}</p>
          <p v-else-if="warningMessage" class="tweezer-warning">{{ warningMessage }}</p>
          <button class="compile-button continuous-generate" :class="{ 'is-running': running }" type="button" @click="generate">
            {{ running ? "Cancel" : "Generate SLM frame" }}
          </button>
          <div class="tweezer-export-actions">
            <button type="button" :disabled="!outputPixels || !metrics?.numericalValid || running" @click="exportBmp">BMP frame <b>↓</b></button>
            <button type="button" :disabled="!outputPixels || running" @click="exportMetadata">Metadata <b>↓</b></button>
          </div>
        </div>
      </section>
    </div>

    <div class="continuous-output-grid">
      <SlmFramePreview :pixels="outputPixels" :width="slmWidth" :height="slmHeight" :status="phaseStatus" :running="running" />
      <IntensityMapPreview
        :intensity="outputIntensity"
        :width="fftWidth"
        :height="fftHeight"
        :running="running"
        :status="reconstructionStatus"
        :logarithmic="false"
        :floor-db="-40"
        :region="outputRegion"
        :target-markers="[]"
        :physical-aspect-ratio="targetAspectRatio"
        view-label="Quantized-frame reconstruction"
        empty-message="Generate a frame to preview its focal intensity"
      />
    </div>

    <div class="metrics-row continuous-metrics" aria-label="Continuous-field metrics">
      <div><small>Calculation time</small><strong>{{ elapsedMs === null ? "--" : `${(elapsedMs / 1000).toFixed(2)}s` }}</strong></div>
      <div><small>Target-fit RMSE</small><strong>{{ metrics ? `${(metrics.normalizedIntensityRmse * 100).toFixed(2)}%` : "--" }}</strong></div>
      <div><small>Correlation</small><strong>{{ metrics ? metrics.intensityCorrelation.toFixed(4) : "--" }}</strong></div>
      <div><small>Signal efficiency</small><strong>{{ metrics ? `${(metrics.diffractionEfficiency * 100).toFixed(1)}%` : "--" }}</strong></div>
      <div><small>Bright-region speckle</small><strong>{{ metrics ? metrics.brightRegionSpeckleContrast.toFixed(3) : "--" }}</strong></div>
    </div>
  </section>
</template>
