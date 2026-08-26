<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, shallowRef, watch } from "vue";
import {
  type CalibrationPackage,
  type FrameMetrics,
  WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN,
  WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN,
  WGS_REFERENCE_TRAP_AMPLITUDE_GAIN,
} from "../../../src/index.js";
import ComputationActivity from "../components/ComputationActivity.vue";
import ForwardSimulator from "../components/ForwardSimulator.vue";
import SlmFramePreview from "../components/SlmFramePreview.vue";
import TargetImageImporter from "../components/TargetImageImporter.vue";
import { encodeGrayscaleBmp } from "../lib/bmp.js";
import {
  createOpticalCalibration,
  analyzeOpticalTrapResolution,
  DEFAULT_FOCAL_LENGTH_MM,
  DEFAULT_PIXEL_PITCH_UM,
  DEFAULT_WAVELENGTH_NM,
  opticalFieldOfViewUm,
  opticalFirstNullResolutionUm,
  parsePhaseResponseLut,
  validateOpticalCalibration,
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
type TweezerTargetPhaseMode = "REFERENCE_WGS" | "PHASE_LOCKED_WGS";

interface TargetImageImporterHandle {
  reset(): void;
}

interface ForwardSimulatorHandle {
  reset(): void;
}

const props = defineProps<{
  webgpuAvailable: boolean;
  webgpuStatus: string;
}>();

const upload = ref<HTMLInputElement | null>(null);
const calibrationUpload = ref<HTMLInputElement | null>(null);
const targetImageImporter = ref<TargetImageImporterHandle | null>(null);
const forwardSimulator = ref<ForwardSimulatorHandle | null>(null);
const tweezers = ref<OpticalTweezerInput[]>(cloneOpticalTweezers(DEFAULT_OPTICAL_TWEEZERS));
const jsonDraft = ref(serializeOpticalTweezers(tweezers.value));
const jsonStatus = ref<JsonStatus>("synced");
const slmWidth = ref(DEFAULT_SLM_WIDTH);
const slmHeight = ref(DEFAULT_SLM_HEIGHT);
const iterations = ref(4);
const computeBackend = ref<ComputeBackend>("wasm");
const targetPhaseMode = ref<TweezerTargetPhaseMode>("PHASE_LOCKED_WGS");
const amplitudeTolerancePercent = ref(0.01);
const wavelengthNm = ref(DEFAULT_WAVELENGTH_NM);
const focalLengthMm = ref(DEFAULT_FOCAL_LENGTH_MM);
const pixelPitchUm = ref(DEFAULT_PIXEL_PITCH_UM);
const phaseResponseLut = shallowRef<number[] | undefined>(undefined);
const phaseResponseFilename = ref("");
const running = ref(false);
const outputState = ref<OutputState>("idle");
const outputPixels = shallowRef<Uint8Array | Uint16Array | null>(null);
const metrics = shallowRef<FrameMetrics | null>(null);
const elapsedMs = ref<number | null>(null);
const errorMessage = ref("");
const qualityWarning = ref("");
const checksum = ref<number | null>(null);
const resultBackendId = ref("wasm-exact-nudft-phase-locked-wgs");
let suppressTableSync = false;
let generation = 0;
let frameWorker: Worker | null = null;
let elapsedTimer = 0;
let calculationStarted = 0;

const fftWidth = computed(() => fftDimensionFor(slmWidth.value));
const fftHeight = computed(() => fftDimensionFor(slmHeight.value));
const opticalCalibrationInput = computed<OpticalCalibrationInput>(() => ({
  wavelengthNm: wavelengthNm.value,
  focalLengthMm: focalLengthMm.value,
  pixelPitchUm: pixelPitchUm.value,
  ...(phaseResponseLut.value ? { phaseResponseLut: [...phaseResponseLut.value] } : {}),
}));
const fieldOfViewUm = computed(() => {
  try {
    return opticalFieldOfViewUm(opticalCalibrationInput.value);
  } catch {
    return null;
  }
});
const opticalResolution = computed(() => {
  try {
    return opticalFirstNullResolutionUm({
      activeWidth: slmWidth.value,
      activeHeight: slmHeight.value,
    }, opticalCalibrationInput.value);
  } catch {
    return null;
  }
});
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
  targetPhaseMode.value = "REFERENCE_WGS";
  amplitudeTolerancePercent.value = 10;
  iterations.value = Math.max(iterations.value, 12);
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
  targetPhaseMode.value = "PHASE_LOCKED_WGS";
  amplitudeTolerancePercent.value = 0.01;
  wavelengthNm.value = DEFAULT_WAVELENGTH_NM;
  focalLengthMm.value = DEFAULT_FOCAL_LENGTH_MM;
  pixelPitchUm.value = DEFAULT_PIXEL_PITCH_UM;
  phaseResponseLut.value = undefined;
  phaseResponseFilename.value = "";
  running.value = false;
  outputState.value = "idle";
  outputPixels.value = null;
  metrics.value = null;
  elapsedMs.value = null;
  checksum.value = null;
  resultBackendId.value = "wasm-exact-nudft-phase-locked-wgs";
  errorMessage.value = "";
  qualityWarning.value = "";
  nextTick(() => {
    suppressTableSync = false;
    targetImageImporter.value?.reset();
    forwardSimulator.value?.reset();
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

function updateTargetPhaseMode(event: Event): void {
  targetPhaseMode.value = (event.target as HTMLSelectElement).value as TweezerTargetPhaseMode;
  invalidateOutput();
}

function updateAmplitudeTolerance(): void {
  if (!Number.isFinite(amplitudeTolerancePercent.value) ||
      amplitudeTolerancePercent.value <= 0 || amplitudeTolerancePercent.value > 100) {
    errorMessage.value = "Amplitude certificate tolerance must be greater than 0% and no more than 100%";
    outputState.value = "rejected";
    return;
  }
  invalidateOutput();
}

function calibrationFor(): CalibrationPackage {
  return createOpticalCalibration({
    activeWidth: slmWidth.value,
    activeHeight: slmHeight.value,
    fftWidth: fftWidth.value,
    fftHeight: fftHeight.value,
  }, opticalCalibrationInput.value, "browser-single-frame-optical-calibration");
}

function updateOpticalCalibration(): void {
  try {
    validateOpticalCalibration(opticalCalibrationInput.value);
    errorMessage.value = "";
    invalidateOutput();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Invalid optical calibration";
    outputState.value = "rejected";
  }
}

async function loadPhaseResponse(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  try {
    phaseResponseLut.value = parsePhaseResponseLut(await file.text());
    phaseResponseFilename.value = file.name;
    invalidateOutput();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Unable to load the measured phase-response LUT";
    outputState.value = "rejected";
  }
}

function clearPhaseResponse(): void {
  phaseResponseLut.value = undefined;
  phaseResponseFilename.value = "";
  invalidateOutput();
}

function generateFrame(): void {
  if (running.value) return;
  const input = validatedTweezers();
  if (!input) return;
  let calibration: CalibrationPackage;
  try {
    validateOpticalCalibration(opticalCalibrationInput.value);
    updateAmplitudeToleranceValue();
    calibration = calibrationFor();
    const resolution = analyzeOpticalTrapResolution(input, calibration, fftWidth.value, fftHeight.value);
    if (resolution.unresolvedPairCount > 0 && resolution.worstPair) {
      const first = input[resolution.worstPair.firstIndex]!;
      const second = input[resolution.worstPair.secondIndex]!;
      throw new Error(
        `Traps ${first.trapId} and ${second.trapId} are not independently resolvable under the current calibration `
        + `(mode correlation ${resolution.worstPair.correlation.toFixed(4)}, separation ${resolution.worstPair.distanceUm.toFixed(4)} µm). `
        + "Increase their separation, enlarge the image field, or correct the optical calibration.",
      );
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Invalid optical calibration";
    outputState.value = "rejected";
    return;
  }
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
      qualityWarning.value = convergenceWarning(response.metrics);
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
      targetPhaseMode: targetPhaseMode.value,
      convergenceTolerance: amplitudeTolerancePercent.value / 100,
      backend: computeBackend.value,
      opticalCalibration: opticalCalibrationInput.value,
    },
  };
  worker.postMessage(request);
}

function updateAmplitudeToleranceValue(): void {
  if (!Number.isFinite(amplitudeTolerancePercent.value) ||
      amplitudeTolerancePercent.value <= 0 || amplitudeTolerancePercent.value > 100) {
    throw new Error("Amplitude certificate tolerance must be greater than 0% and no more than 100%");
  }
}

function convergenceWarning(metric: FrameMetrics): string {
  const failed: string[] = [];
  if (metric.maximumRelativeAmplitudeError > metric.amplitudeConvergenceTolerance) {
    failed.push(
      `relative amplitude error ${(metric.maximumRelativeAmplitudeError * 100).toFixed(4)}% `
      + `(limit ${(metric.amplitudeConvergenceTolerance * 100).toFixed(4)}%)`,
    );
  }
  if (metric.maximumTargetPhaseErrorRad > metric.phaseConvergenceToleranceRad) {
    failed.push(
      `phase error ${metric.maximumTargetPhaseErrorRad.toFixed(4)} rad `
      + `(limit ${metric.phaseConvergenceToleranceRad.toFixed(4)} rad)`,
    );
  }
  const detail = failed.length > 0 ? failed.join("; ") : "the configured convergence gates";
  return `The best exportable frame was retained, but it is not certified: ${detail}. `
    + "The requested tolerance was not reached; inspect trap spacing and intensity, or increase the iteration budget.";
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
  const calibration = calibrationFor();
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
      targetPhaseMode: targetPhaseMode.value,
      amplitudeConvergenceTolerance: amplitudeTolerancePercent.value / 100,
      backgroundPolicy: "ZERO",
      effectiveAmplitudeFeedbackGain: targetPhaseMode.value === "REFERENCE_WGS"
        ? WGS_REFERENCE_TRAP_AMPLITUDE_GAIN
        : WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN,
      phasePrecompensationGain: targetPhaseMode.value === "REFERENCE_WGS"
        ? 0
        : WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN,
      retainBestQuantizedCandidate: true,
      elapsedMs: elapsedMs.value,
    },
    calibration: {
      manifest: calibration.manifest,
      coordinateTransform: calibration.coordinateTransform,
      phaseResponseLut: phaseResponseLut.value ?? "ideal linear 2π response",
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
          <p>Coordinates are physical focal-plane micrometres. Wavelength, Fourier-lens focal length, and SLM pixel pitch determine the exact NUDFT frequency; phase is the requested complex-field phase.</p>
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
          <span>FOREGROUND FIT / RESOLUTION SAFE / FREE PHASE</span>
        </div>
        <TargetImageImporter
          ref="targetImageImporter"
          destination="tweezers"
          :disabled="running"
          :minimum-separation-x-um="opticalResolution?.xUm ?? 0"
          :minimum-separation-y-um="opticalResolution?.yUm ?? 0"
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
            </div>
            <div class="optical-calibration-block">
              <div class="control-label"><span>OPTICAL CALIBRATION</span><output>FRAUNHOFER / NUDFT</output></div>
              <div class="optical-calibration-fields">
                <label>WAVELENGTH / NM
                  <input v-model.number="wavelengthNm" type="number" min="0.001" step="any" :disabled="running" @change="updateOpticalCalibration">
                </label>
                <label>FOCAL LENGTH / MM
                  <input v-model.number="focalLengthMm" type="number" min="0.001" step="any" :disabled="running" @change="updateOpticalCalibration">
                </label>
                <label>PIXEL PITCH / UM
                  <input v-model.number="pixelPitchUm" type="number" min="0.001" step="any" :disabled="running" @change="updateOpticalCalibration">
                </label>
              </div>
              <p class="tweezer-calibration-note">
                FOCAL-PLANE FOV {{ fieldOfViewUm === null ? "INVALID" : `${fieldOfViewUm.toFixed(1)} × ${fieldOfViewUm.toFixed(1)} UM` }}
                <template v-if="opticalResolution"> / FIRST-NULL {{ opticalResolution.xUm.toFixed(3) }} × {{ opticalResolution.yUm.toFixed(3) }} UM</template>
              </p>
              <div class="phase-lut-actions">
                <button type="button" :disabled="running" @click="calibrationUpload?.click()">
                  {{ phaseResponseLut ? "Replace measured phase LUT" : "Upload measured phase LUT" }}
                </button>
                <button v-if="phaseResponseLut" type="button" :disabled="running" @click="clearPhaseResponse">Use ideal linear response</button>
                <input ref="calibrationUpload" class="sr-only" type="file" accept="application/json,.json,.csv,.txt,text/plain,text/csv" @change="loadPhaseResponse">
              </div>
              <p class="resolution-note">
                {{ phaseResponseLut ? `MEASURED RESPONSE / ${phaseResponseLut.length} SAMPLES / ${phaseResponseFilename}` : "IDEAL LINEAR 2π RESPONSE / UPLOAD MEASURED VALUES FOR HARDWARE" }}
              </p>
            </div>
            <label class="tweezer-backend-choice">COMPUTE BACKEND
              <select :value="computeBackend" :disabled="running" @change="updateComputeBackend">
                <option value="wasm">WebAssembly / exact NUDFT</option>
                <option value="webgpu" :disabled="!webgpuAvailable">WebGPU / GPU-resident exact NUDFT</option>
              </select>
              <small :class="{ 'is-available': webgpuAvailable }">{{ webgpuStatus }}</small>
            </label>
            <label class="tweezer-backend-choice">TARGET PHASE CONSTRAINT
              <select :value="targetPhaseMode" :disabled="running" @change="updateTargetPhaseMode">
                <option value="PHASE_LOCKED_WGS">Use phases from the table / JSON</option>
                <option value="REFERENCE_WGS">Free phase / intensity-only target</option>
              </select>
              <small>{{ targetPhaseMode === "REFERENCE_WGS" ? "OUTPUT PHASES FLOAT TO IMPROVE PATTERN UNIFORMITY" : "EVERY REQUESTED PHASE IS INCLUDED IN CERTIFICATION" }}</small>
            </label>
            <label class="tweezer-backend-choice tweezer-certificate-choice">MAX AMPLITUDE ERROR / %
              <input v-model.number="amplitudeTolerancePercent" type="number" min="0.0001" max="100" step="0.01" :disabled="running" @change="updateAmplitudeTolerance">
              <small>EXPLICIT CERTIFICATE LIMIT / ACTUAL ERROR REMAINS VISIBLE IN METRICS</small>
            </label>
            <div class="control-block">
              <div class="control-label"><span>WGS ITERATIONS</span><output>{{ String(iterations).padStart(2, "0") }} / FRAME</output></div>
              <input type="range" min="1" max="12" step="1" :value="iterations" :disabled="running" @input="updateIterations">
            </div>
            <ComputationActivity
              v-if="running"
              label="SYNTHESIZING PHASE FIELD"
              :detail="`${fftWidth} × ${fftHeight} EXACT TRAP NUDFT / DEDICATED WORKER`"
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
      <div><small>MAX AMPLITUDE ERROR</small><strong>{{ metrics ? `${(metrics.maximumRelativeAmplitudeError * 100).toFixed(4)}%` : "--" }}</strong></div>
      <div><small>MAX PHASE ERROR</small><strong>{{ targetPhaseMode === "REFERENCE_WGS" ? "FREE" : metrics ? `${metrics.maximumTargetPhaseErrorRad.toFixed(2)} rad` : "--" }}</strong></div>
    </div>

    <ForwardSimulator
      ref="forwardSimulator"
      :generated-pixels="outputPixels"
      :generated-width="slmWidth"
      :generated-height="slmHeight"
      :webgpu-available="webgpuAvailable"
      :webgpu-status="webgpuStatus"
    />
  </section>
</template>
