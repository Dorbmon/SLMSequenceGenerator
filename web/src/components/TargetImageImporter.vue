<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from "vue";
import ComputationActivity from "./ComputationActivity.vue";
import {
  mapImagePointsToField,
  type DetectedImagePoint,
  type ImageExtractionMode,
  type SpotDetectionOptions,
  type SpotPolarity,
} from "../lib/image-points.js";
import type {
  TargetImageWorkerRequest,
  TargetImageWorkerResponse,
} from "../workers/target-image-messages.js";

interface ImportedTargetPoint {
  xUm: number;
  yUm: number;
  relativeIntensity: number;
}

const props = withDefaults(defineProps<{
  disabled: boolean;
  destination?: "targets" | "tweezers";
}>(), {
  destination: "targets",
});

const emit = defineEmits<{
  apply: [points: ImportedTargetPoint[], complete: (accepted: boolean) => void];
}>();

const MAX_FILE_BYTES = 32 * 1024 * 1024;
const MAX_ANALYSIS_EDGE = 1280;
const MAX_ANALYSIS_PIXELS = 1_500_000;

const upload = ref<HTMLInputElement | null>(null);
const preview = ref<HTMLCanvasElement | null>(null);
const imageName = ref("");
const sourceWidth = ref(0);
const sourceHeight = ref(0);
const analysisWidth = ref(0);
const analysisHeight = ref(0);
const fieldWidthUm = ref(20);
const fieldHeightUm = ref(20);
const extractionMode = ref<ImageExtractionMode>(props.destination === "tweezers" ? "PATTERN" : "CENTROIDS");
const thresholdPercent = ref(props.destination === "tweezers" ? 20 : 62);
const minimumAreaPx = ref(3);
const maximumPoints = ref(256);
const patternSpacingPx = ref(2);
const polarity = ref<SpotPolarity>("BRIGHT");
const detections = ref<DetectedImagePoint[]>([]);
const running = ref(false);
const draggingOver = ref(false);
const errorMessage = ref("");
const elapsedMs = ref(0);
const thresholdSignal = ref(0);
const discardedSmall = ref(0);
const discardedLarge = ref(0);
const discardedByLimit = ref(0);
const sourcePixelCount = ref(0);
const effectiveSpacingPx = ref(0);
const appliedCount = ref<number | null>(null);
let sourceCanvas: HTMLCanvasElement | null = null;
let detector: Worker | null = null;
let loadGeneration = 0;
let jobGeneration = 0;
let latestJob = 0;
let detectionTimer = 0;

const hasImage = computed(() => Boolean(imageName.value && sourceCanvas));
const mappingValid = computed(() => (
  Number.isFinite(fieldWidthUm.value)
  && fieldWidthUm.value > 0
  && Number.isFinite(fieldHeightUm.value)
  && fieldHeightUm.value > 0
));
const analysisWasReduced = computed(() => (
  sourceWidth.value !== analysisWidth.value || sourceHeight.value !== analysisHeight.value
));
const pointSummary = computed(() => {
  if (running.value) return "ANALYZING";
  if (!hasImage.value) return "NO IMAGE";
  return `${detections.value.length} POINT${detections.value.length === 1 ? "" : "S"}`;
});
const discardedSummary = computed(() => {
  if (extractionMode.value === "PATTERN") {
    const noise = discardedSmall.value > 0
      ? ` · ${discardedSmall.value} noise component${discardedSmall.value === 1 ? "" : "s"} removed`
      : "";
    const consolidated = discardedByLimit.value > 0 ? " · density reduced to point limit" : "";
    return `${sourcePixelCount.value} foreground px · ${effectiveSpacingPx.value}px sampling${noise}${consolidated}`;
  }
  const discarded = discardedSmall.value + discardedLarge.value + discardedByLimit.value;
  return discarded > 0 ? `${discarded} component${discarded === 1 ? "" : "s"} filtered` : "No extra components filtered";
});
const destinationCopy = computed(() => props.destination === "tweezers" ? {
  title: "Extract optical tweezers from an image",
  description: "Upload a target-field image. Image pattern mode traces shapes or text with a point cloud; centroid mode extracts isolated spots.",
  upload: "Upload tweezer image",
  applied: "tweezers",
} : {
  title: "Extract target sites from an image",
  description: "Upload a target-field image. Bright or dark connected spots are converted into calibrated point centroids.",
  upload: "Upload target image",
  applied: "targets",
});

function detectionOptions(): SpotDetectionOptions {
  return {
    mode: extractionMode.value,
    polarity: polarity.value,
    threshold: thresholdPercent.value / 100,
    minimumAreaPx: minimumAreaPx.value,
    maximumPoints: maximumPoints.value,
    patternSpacingPx: patternSpacingPx.value,
  };
}

async function handleUpload(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (file) await loadImage(file);
}

async function handleDrop(event: DragEvent): Promise<void> {
  draggingOver.value = false;
  if (props.disabled) return;
  const file = event.dataTransfer?.files[0];
  if (file) await loadImage(file);
}

async function loadImage(file: File): Promise<void> {
  if (props.disabled) return;
  if (!file.type.startsWith("image/")) {
    showError("Choose a PNG, JPEG, WebP, or other browser-readable image");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError("Target-field images must be 32 MB or smaller");
    return;
  }

  const activeLoad = ++loadGeneration;
  terminateDetector();
  running.value = true;
  errorMessage.value = "";
  appliedCount.value = null;
  sourceCanvas = null;
  imageName.value = "";
  sourceWidth.value = 0;
  sourceHeight.value = 0;
  analysisWidth.value = 0;
  analysisHeight.value = 0;
  detections.value = [];
  elapsedMs.value = 0;

  try {
    const bitmap = await createImageBitmap(file);
    if (activeLoad !== loadGeneration) {
      bitmap.close();
      return;
    }
    if (bitmap.width <= 0 || bitmap.height <= 0) {
      bitmap.close();
      throw new Error("The selected image has no readable pixels");
    }
    const scale = Math.min(
      1,
      MAX_ANALYSIS_EDGE / bitmap.width,
      MAX_ANALYSIS_EDGE / bitmap.height,
      Math.sqrt(MAX_ANALYSIS_PIXELS / (bitmap.width * bitmap.height)),
    );
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));
    const nextSource = document.createElement("canvas");
    nextSource.width = width;
    nextSource.height = height;
    const sourceContext = nextSource.getContext("2d", { willReadFrequently: true });
    if (!sourceContext) {
      bitmap.close();
      throw new Error("This browser could not create an image-analysis canvas");
    }
    sourceContext.drawImage(bitmap, 0, 0, width, height);
    sourceWidth.value = bitmap.width;
    sourceHeight.value = bitmap.height;
    bitmap.close();
    analysisWidth.value = width;
    analysisHeight.value = height;
    imageName.value = file.name;
    fieldWidthUm.value = 20;
    fieldHeightUm.value = roundedFieldSize(20 * height / width);
    sourceCanvas = nextSource;
    await nextTick();
    drawPreview();
    startDetector(sourceContext.getImageData(0, 0, width, height));
  } catch (error) {
    if (activeLoad !== loadGeneration) return;
    running.value = false;
    showError(error instanceof Error ? error.message : "Unable to decode the target-field image");
  }
}

function startDetector(image: ImageData): void {
  const worker = new Worker(new URL("../workers/target-image.worker.ts", import.meta.url), { type: "module" });
  detector = worker;
  worker.onmessage = (event: MessageEvent<TargetImageWorkerResponse>) => {
    const response = event.data;
    if (worker !== detector || response.jobId !== latestJob) return;
    if (response.kind === "TARGET_IMAGE_ERROR") {
      running.value = false;
      showError(response.message);
      return;
    }
    detections.value = response.points;
    thresholdSignal.value = response.thresholdSignal;
    discardedSmall.value = response.discardedSmallComponents;
    discardedLarge.value = response.discardedLargeComponents;
    discardedByLimit.value = response.discardedByLimit;
    sourcePixelCount.value = response.sourcePixelCount;
    effectiveSpacingPx.value = response.effectiveSpacingPx;
    elapsedMs.value = response.elapsedMs;
    running.value = false;
    errorMessage.value = "";
    drawPreview();
  };
  worker.onerror = (event: ErrorEvent) => {
    if (worker !== detector) return;
    running.value = false;
    showError(event.message || "Target-field detector worker failed");
  };
  const jobId = ++jobGeneration;
  latestJob = jobId;
  const rgba = image.data.buffer as ArrayBuffer;
  const request: TargetImageWorkerRequest = {
    kind: "LOAD_TARGET_IMAGE",
    jobId,
    width: image.width,
    height: image.height,
    rgba,
    options: detectionOptions(),
  };
  worker.postMessage(request, [rgba]);
}

function scheduleDetection(): void {
  appliedCount.value = null;
  window.clearTimeout(detectionTimer);
  if (!detector || !hasImage.value) return;
  running.value = true;
  detectionTimer = window.setTimeout(requestDetection, 100);
}

function requestDetection(): void {
  if (!detector) return;
  const jobId = ++jobGeneration;
  latestJob = jobId;
  const request: TargetImageWorkerRequest = {
    kind: "DETECT_TARGET_IMAGE",
    jobId,
    options: detectionOptions(),
  };
  detector.postMessage(request);
}

function drawPreview(): void {
  const canvas = preview.value;
  if (!canvas || !sourceCanvas) return;
  canvas.width = sourceCanvas.width;
  canvas.height = sourceCanvas.height;
  const previewContext = canvas.getContext("2d");
  if (!previewContext) return;
  previewContext.drawImage(sourceCanvas, 0, 0);
  const scale = Math.max(1, Math.min(canvas.width, canvas.height) / 420);
  detections.value.forEach((point, index) => {
    const pattern = extractionMode.value === "PATTERN";
    const radius = pattern
      ? Math.max(1.4 * scale, effectiveSpacingPx.value * 0.32)
      : Math.max(5 * scale, Math.sqrt(point.areaPx / Math.PI) + 3 * scale);
    previewContext.save();
    previewContext.strokeStyle = "#b9f36b";
    previewContext.fillStyle = pattern ? "rgba(185,243,107,.62)" : "rgba(8,17,15,.84)";
    previewContext.lineWidth = pattern ? Math.max(0.75, 0.75 * scale) : Math.max(1.5, 1.5 * scale);
    previewContext.beginPath();
    previewContext.arc(point.xPx, point.yPx, radius, 0, Math.PI * 2);
    previewContext.fill();
    previewContext.stroke();
    if (!pattern) {
      previewContext.beginPath();
      previewContext.moveTo(point.xPx - radius * 0.55, point.yPx);
      previewContext.lineTo(point.xPx + radius * 0.55, point.yPx);
      previewContext.moveTo(point.xPx, point.yPx - radius * 0.55);
      previewContext.lineTo(point.xPx, point.yPx + radius * 0.55);
      previewContext.stroke();
    }
    if (!pattern && detections.value.length <= 100) {
      previewContext.fillStyle = "#e6eee8";
      previewContext.font = `${Math.max(9, 9 * scale)}px 'DM Mono', monospace`;
      previewContext.fillText(String(index + 1), point.xPx + radius + 2 * scale, point.yPx - radius * 0.45);
    }
    previewContext.restore();
  });
}

function applyDetectedTargets(): void {
  if (props.disabled || running.value || detections.value.length === 0) return;
  if (!mappingValid.value) {
    showError("Image field width and height must be positive numbers");
    return;
  }
  const mapped = mapImagePointsToField(
    detections.value,
    analysisWidth.value,
    analysisHeight.value,
    fieldWidthUm.value,
    fieldHeightUm.value,
  );
  const maximumSignal = Math.max(1, ...mapped.map((point) => point.integratedSignal));
  const points = mapped.map(({ xUm, yUm, integratedSignal }) => ({
    xUm,
    yUm,
    relativeIntensity: Math.max(0.001, integratedSignal / maximumSignal),
  }));
  emit("apply", points, (accepted) => {
    if (accepted) {
      appliedCount.value = points.length;
      errorMessage.value = "";
    } else {
      showError(props.destination === "tweezers"
        ? "Unable to replace optical tweezers while the editor is busy"
        : "Resolve the coordinate JSON error before replacing target sites");
    }
  });
}

function reset(): void {
  loadGeneration += 1;
  terminateDetector();
  sourceCanvas = null;
  imageName.value = "";
  sourceWidth.value = 0;
  sourceHeight.value = 0;
  analysisWidth.value = 0;
  analysisHeight.value = 0;
  detections.value = [];
  running.value = false;
  draggingOver.value = false;
  errorMessage.value = "";
  appliedCount.value = null;
  extractionMode.value = props.destination === "tweezers" ? "PATTERN" : "CENTROIDS";
  thresholdPercent.value = props.destination === "tweezers" ? 20 : 62;
  minimumAreaPx.value = 3;
  maximumPoints.value = 256;
  patternSpacingPx.value = 2;
  polarity.value = "BRIGHT";
  sourcePixelCount.value = 0;
  effectiveSpacingPx.value = 0;
}

function terminateDetector(): void {
  window.clearTimeout(detectionTimer);
  detectionTimer = 0;
  latestJob = ++jobGeneration;
  detector?.terminate();
  detector = null;
}

function showError(message: string): void {
  errorMessage.value = message;
}

function roundedFieldSize(value: number): number {
  return Math.max(0.001, Math.round(value * 1000) / 1000);
}

watch([extractionMode, thresholdPercent, minimumAreaPx, maximumPoints, patternSpacingPx, polarity], scheduleDetection);
watch([fieldWidthUm, fieldHeightUm], () => { appliedCount.value = null; });

onBeforeUnmount(() => {
  loadGeneration += 1;
  terminateDetector();
});

defineExpose({ reset });
</script>

<template>
  <div
    class="target-image-importer"
    :class="{ 'is-dragging': draggingOver, 'has-image': hasImage, 'is-disabled': disabled }"
    @dragenter.prevent="draggingOver = true"
    @dragover.prevent="draggingOver = true"
    @dragleave.prevent="draggingOver = false"
    @drop.prevent="handleDrop"
  >
    <div v-if="!hasImage" class="target-image-empty">
      <div class="image-import-mark" aria-hidden="true"><i></i><i></i><i></i><b>+</b></div>
      <div>
        <strong>{{ destinationCopy.title }}</strong>
        <p>{{ destinationCopy.description }}</p>
      </div>
      <button class="apply-button" type="button" :disabled="disabled || running" @click="upload?.click()">
        {{ running ? "Decoding image..." : destinationCopy.upload }}
      </button>
      <p v-if="errorMessage" class="target-image-error target-image-load-error" role="alert">{{ errorMessage }}</p>
    </div>

    <div v-else class="target-image-layout">
      <div class="target-image-preview">
        <canvas ref="preview" :aria-label="`Detected target points in ${imageName}`"></canvas>
        <div class="target-image-preview-bar">
          <span>{{ imageName }}</span>
          <strong>{{ pointSummary }}</strong>
        </div>
        <div v-if="running" class="canvas-compute-overlay" aria-hidden="true">
          <div class="field-orbits"><i></i><i></i><i></i><b></b></div>
          <span>DETECTOR WORKER / UI RESPONSIVE</span>
        </div>
      </div>

      <div class="target-image-controls">
        <div class="target-image-heading">
          <div><span>POINT DETECTION</span><strong>{{ detections.length }} FOUND</strong></div>
          <button type="button" :disabled="disabled" @click="reset">Remove image</button>
        </div>

        <label>EXTRACTION MODE
          <select v-model="extractionMode" :disabled="disabled">
            <option value="PATTERN">Image pattern / trace shape</option>
            <option value="CENTROIDS">Isolated spot centroids</option>
          </select>
        </label>

        <label>SIGNAL POLARITY
          <select v-model="polarity" :disabled="disabled">
            <option value="BRIGHT">Bright spots</option>
            <option value="DARK">Dark spots</option>
          </select>
        </label>

        <div class="image-control-block">
          <div><label for="target-image-threshold">CONTRAST THRESHOLD</label><output>{{ thresholdPercent }}%</output></div>
          <input id="target-image-threshold" v-model.number="thresholdPercent" type="range" min="5" max="98" step="1" :disabled="disabled">
        </div>

        <div class="target-image-number-grid">
          <label>{{ extractionMode === "PATTERN" ? "NOISE BLOB / PX" : "MIN BLOB / PX" }}
            <input v-model.number="minimumAreaPx" type="number" min="1" max="10000" step="1" :disabled="disabled">
          </label>
          <label v-if="extractionMode === 'PATTERN'">SAMPLE SPACING / PX
            <input v-model.number="patternSpacingPx" type="number" min="1" max="64" step="1" :disabled="disabled">
          </label>
          <label>POINT LIMIT
            <input v-model.number="maximumPoints" type="number" min="1" max="1000" step="1" :disabled="disabled">
          </label>
        </div>

        <div class="target-image-field-heading"><span>IMAGE FIELD / UM</span><small>CENTERED ORIGIN / +Y UP</small></div>
        <div class="target-image-number-grid">
          <label>WIDTH
            <input v-model.number="fieldWidthUm" type="number" min="0.001" step="0.1" inputmode="decimal" :disabled="disabled">
          </label>
          <label>HEIGHT
            <input v-model.number="fieldHeightUm" type="number" min="0.001" step="0.1" inputmode="decimal" :disabled="disabled">
          </label>
        </div>

        <ComputationActivity
          v-if="running"
          :label="extractionMode === 'PATTERN' ? 'SAMPLING IMAGE PATTERN' : 'DETECTING SPOT CENTROIDS'"
          :detail="`${analysisWidth} × ${analysisHeight} / DEDICATED WORKER`"
          :elapsed-ms="elapsedMs"
          :progress="null"
        />

        <p v-if="errorMessage" class="target-image-error" role="alert">{{ errorMessage }}</p>
        <div v-else class="target-image-diagnostics">
          <span>{{ sourceWidth }} × {{ sourceHeight }} PX<span v-if="analysisWasReduced"> / REDUCED FOR ANALYSIS</span></span>
          <span>THRESHOLD {{ thresholdSignal.toFixed(0) }} / 255 · {{ discardedSummary }}</span>
        </div>

        <button
          class="apply-button target-image-apply"
          type="button"
          :disabled="disabled || running || detections.length === 0 || !mappingValid"
          @click="applyDetectedTargets"
        >
          {{ appliedCount === detections.length ? `${appliedCount} ${destinationCopy.applied} applied` : `Replace ${destinationCopy.applied} with ${detections.length} points` }}
        </button>
      </div>
    </div>

    <input ref="upload" class="sr-only" type="file" accept="image/*" @change="handleUpload">
  </div>
</template>
