<script setup lang="ts">
import { computed, ref, watch } from "vue";
import {
  appendResolvableDrawingPoints,
  eraseDrawingPoints,
  mapViewportToDrawingField,
  pointDistance,
  resampleDrawingPath,
  type ManualDrawingPoint,
} from "../lib/manual-drawing.js";

interface AppliedDrawingPoint extends ManualDrawingPoint {
  relativeIntensity: number;
}

const props = withDefaults(defineProps<{
  disabled: boolean;
  sourcePoints?: readonly ManualDrawingPoint[];
  minimumSeparationXUm?: number;
  minimumSeparationYUm?: number;
}>(), {
  sourcePoints: () => [],
  minimumSeparationXUm: 0,
  minimumSeparationYUm: 0,
});

const emit = defineEmits<{
  apply: [points: AppliedDrawingPoint[], complete: (accepted: boolean) => void];
}>();

const VIEW_WIDTH = 800;
const VIEW_HEIGHT = 400;
const HISTORY_LIMIT = 50;
const RESOLUTION_MARGIN = 1.05;

const surface = ref<SVGSVGElement | null>(null);
const tool = ref<"DRAW" | "ERASE">("DRAW");
const fieldWidthUm = ref(80);
const fieldHeightUm = ref(40);
const pointSpacingUm = ref(2.5);
const pointLimit = ref(128);
const points = ref<ManualDrawingPoint[]>([]);
const history = ref<ManualDrawingPoint[][]>([]);
const cursor = ref<ManualDrawingPoint | null>(null);
const appliedCount = ref<number | null>(null);
const statusMessage = ref("Drag in the field to draw an evenly sampled trap path.");
let activePointerId: number | null = null;
let strokeBase: ManualDrawingPoint[] = [];
let strokePath: ManualDrawingPoint[] = [];
let gestureChanged = false;

const recommendedSpacingUm = computed(() => roundControl(Math.max(
  0.5,
  finiteNonNegative(props.minimumSeparationXUm) * 1.25,
  finiteNonNegative(props.minimumSeparationYUm) * 1.25,
)));
const effectiveSpacingUm = computed(() => Math.max(
  recommendedSpacingUm.value,
  finitePositive(pointSpacingUm.value, recommendedSpacingUm.value),
));
const effectivePointLimit = computed(() => Math.max(1, Math.min(512, Math.round(
  finitePositive(pointLimit.value, 128),
))));
const eraserRadiusUm = computed(() => Math.max(effectiveSpacingUm.value * 1.35, recommendedSpacingUm.value * 1.5));
const surfaceAspectRatio = computed(() => finitePositive(fieldWidthUm.value, 80) / finitePositive(fieldHeightUm.value, 40));
const eraserRadiusX = computed(() => eraserRadiusUm.value / finitePositive(fieldWidthUm.value, 80) * VIEW_WIDTH);
const eraserRadiusY = computed(() => eraserRadiusUm.value / finitePositive(fieldHeightUm.value, 40) * VIEW_HEIGHT);
const verticalGrid = Array.from({ length: 9 }, (_, index) => index / 8 * VIEW_WIDTH);
const horizontalGrid = Array.from({ length: 5 }, (_, index) => index / 4 * VIEW_HEIGHT);
const pointSummary = computed(() => `${points.value.length} / ${effectivePointLimit.value} POINTS`);
const cursorLabel = computed(() => cursor.value
  ? `X ${cursor.value.xUm.toFixed(2)} µm / Y ${cursor.value.yUm.toFixed(2)} µm`
  : "+Y UP / PHYSICAL FOCAL PLANE");
const strokePolyline = computed(() => strokePath.map((point) => `${svgX(point.xUm)},${svgY(point.yUm)}`).join(" "));

function beginGesture(event: PointerEvent): void {
  if (props.disabled || event.button !== 0 || activePointerId !== null) return;
  const point = pointFromEvent(event);
  if (!point) return;
  event.preventDefault();
  surface.value?.setPointerCapture(event.pointerId);
  activePointerId = event.pointerId;
  cursor.value = point;
  pushHistory();
  gestureChanged = false;
  appliedCount.value = null;
  if (tool.value === "DRAW") {
    strokeBase = clonePoints(points.value);
    strokePath = [point];
    refreshDrawnStroke();
  } else {
    eraseAt(point);
  }
}

function continueGesture(event: PointerEvent): void {
  const coalesced = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [event];
  for (const sample of coalesced.length > 0 ? coalesced : [event]) {
    const point = pointFromEvent(sample);
    if (!point) continue;
    cursor.value = point;
    if (activePointerId !== event.pointerId) continue;
    event.preventDefault();
    if (tool.value === "DRAW") {
      const previous = strokePath[strokePath.length - 1];
      if (!previous || pointDistance(previous, point) > 1e-6) {
        strokePath.push(point);
        refreshDrawnStroke();
      }
    } else {
      eraseAt(point);
    }
  }
}

function endGesture(event: PointerEvent): void {
  if (event.pointerId !== activePointerId) return;
  if (surface.value?.hasPointerCapture(event.pointerId)) surface.value.releasePointerCapture(event.pointerId);
  activePointerId = null;
  strokeBase = [];
  strokePath = [];
  if (!gestureChanged) history.value.pop();
  statusMessage.value = points.value.length >= effectivePointLimit.value
    ? `Point limit reached at ${effectivePointLimit.value}; raise the limit or erase points before drawing more.`
    : `${points.value.length} resolution-safe points ready.`;
}

function cancelGesture(event: PointerEvent): void {
  if (event.pointerId !== activePointerId) return;
  endGesture(event);
}

function refreshDrawnStroke(): void {
  const candidates = resampleDrawingPath(strokePath, effectiveSpacingUm.value);
  const next = appendResolvableDrawingPoints(
    strokeBase,
    candidates,
    finiteNonNegative(props.minimumSeparationXUm) * RESOLUTION_MARGIN,
    finiteNonNegative(props.minimumSeparationYUm) * RESOLUTION_MARGIN,
    effectivePointLimit.value,
  );
  gestureChanged ||= !samePoints(points.value, next);
  points.value = next;
}

function eraseAt(center: ManualDrawingPoint): void {
  const next = eraseDrawingPoints(points.value, center, eraserRadiusUm.value);
  gestureChanged ||= next.length !== points.value.length;
  points.value = next;
}

function undo(): void {
  if (props.disabled || history.value.length === 0) return;
  points.value = history.value.pop() ?? [];
  appliedCount.value = null;
  statusMessage.value = `Undo restored ${points.value.length} points.`;
}

function clearDrawing(): void {
  if (props.disabled || points.value.length === 0) return;
  pushHistory();
  points.value = [];
  appliedCount.value = null;
  statusMessage.value = "Drawing canvas cleared.";
}

function loadCurrentPoints(): void {
  if (props.disabled || props.sourcePoints.length === 0) return;
  const validSource = props.sourcePoints.filter((point) => Number.isFinite(point.xUm) && Number.isFinite(point.yUm));
  if (validSource.length === 0) {
    statusMessage.value = "Current target coordinates are invalid; correct the table or JSON before loading them.";
    return;
  }
  pushHistory();
  const source = validSource.slice(0, effectivePointLimit.value).map((point) => ({
    xUm: roundCoordinate(point.xUm),
    yUm: roundCoordinate(point.yUm),
  }));
  points.value = source;
  fitFieldAround(source);
  appliedCount.value = null;
  statusMessage.value = `${source.length} current target points loaded for editing.`;
}

function applyDrawing(): void {
  if (props.disabled || points.value.length === 0) return;
  const prepared = points.value.map((point) => ({ ...point, relativeIntensity: 1 }));
  emit("apply", prepared, (accepted) => {
    if (accepted) {
      appliedCount.value = prepared.length;
      statusMessage.value = `${prepared.length} hand-drawn tweezers applied to the coordinate table and JSON.`;
    } else {
      statusMessage.value = "The drawing could not be applied while the editor is busy.";
    }
  });
}

function normalizeControls(): void {
  fieldWidthUm.value = roundControl(Math.max(1, finitePositive(fieldWidthUm.value, 80)));
  fieldHeightUm.value = roundControl(Math.max(1, finitePositive(fieldHeightUm.value, 40)));
  pointSpacingUm.value = roundControl(effectiveSpacingUm.value);
  pointLimit.value = effectivePointLimit.value;
  appliedCount.value = null;
}

function fitFieldAround(source: readonly ManualDrawingPoint[]): void {
  if (source.length === 0) return;
  const requiredWidth = Math.max(20, Math.max(...source.map((point) => Math.abs(point.xUm))) * 2.3);
  const requiredHeight = Math.max(10, Math.max(...source.map((point) => Math.abs(point.yUm))) * 2.3);
  fieldWidthUm.value = roundControl(Math.max(fieldWidthUm.value, requiredWidth));
  fieldHeightUm.value = roundControl(Math.max(fieldHeightUm.value, requiredHeight));
}

function pointFromEvent(event: PointerEvent): ManualDrawingPoint | null {
  const rectangle = surface.value?.getBoundingClientRect();
  if (!rectangle || rectangle.width <= 0 || rectangle.height <= 0) return null;
  return mapViewportToDrawingField(
    event.clientX,
    event.clientY,
    rectangle,
    finitePositive(fieldWidthUm.value, 80),
    finitePositive(fieldHeightUm.value, 40),
  );
}

function pushHistory(): void {
  history.value.push(clonePoints(points.value));
  if (history.value.length > HISTORY_LIMIT) history.value.shift();
}

function svgX(value: number): number {
  return (value / finitePositive(fieldWidthUm.value, 80) + 0.5) * VIEW_WIDTH;
}

function svgY(value: number): number {
  return (0.5 - value / finitePositive(fieldHeightUm.value, 40)) * VIEW_HEIGHT;
}

function reset(): void {
  activePointerId = null;
  strokeBase = [];
  strokePath = [];
  tool.value = "DRAW";
  fieldWidthUm.value = 80;
  fieldHeightUm.value = 40;
  pointSpacingUm.value = recommendedSpacingUm.value;
  pointLimit.value = 128;
  points.value = [];
  history.value = [];
  cursor.value = null;
  appliedCount.value = null;
  statusMessage.value = "Drag in the field to draw an evenly sampled trap path.";
}

function clonePoints(source: readonly ManualDrawingPoint[]): ManualDrawingPoint[] {
  return source.map((point) => ({ ...point }));
}

function samePoints(first: readonly ManualDrawingPoint[], second: readonly ManualDrawingPoint[]): boolean {
  return first.length === second.length && first.every((point, index) => (
    point.xUm === second[index]?.xUm && point.yUm === second[index]?.yUm
  ));
}

function finitePositive(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function finiteNonNegative(value: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

function roundControl(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function roundCoordinate(value: number): number {
  const rounded = Math.round(value * 1e6) / 1e6;
  return Object.is(rounded, -0) ? 0 : rounded;
}

watch(recommendedSpacingUm, (minimum) => {
  if (!Number.isFinite(pointSpacingUm.value) || pointSpacingUm.value < minimum) pointSpacingUm.value = minimum;
}, { immediate: true });
watch([fieldWidthUm, fieldHeightUm, pointSpacingUm, pointLimit], () => {
  appliedCount.value = null;
});

defineExpose({ reset });
</script>

<template>
  <div class="manual-tweezer-drawer" :class="{ 'is-disabled': disabled }">
    <div class="manual-drawing-layout">
      <div class="manual-drawing-preview">
        <svg
          ref="surface"
          class="manual-drawing-surface"
          :class="`is-${tool.toLowerCase()}`"
          viewBox="0 0 800 400"
          preserveAspectRatio="none"
          role="application"
          aria-label="Draw optical tweezer paths in the calibrated focal plane"
          :style="{ aspectRatio: String(surfaceAspectRatio) }"
          @pointerdown="beginGesture"
          @pointermove="continueGesture"
          @pointerup="endGesture"
          @pointercancel="cancelGesture"
          @pointerleave="activePointerId === null && (cursor = null)"
          @contextmenu.prevent
        >
          <rect class="manual-field-background" width="800" height="400"></rect>
          <line v-for="position in verticalGrid" :key="`vx-${position}`" class="manual-field-grid" :x1="position" y1="0" :x2="position" y2="400"></line>
          <line v-for="position in horizontalGrid" :key="`hy-${position}`" class="manual-field-grid" x1="0" :y1="position" x2="800" :y2="position"></line>
          <line class="manual-field-axis" x1="0" y1="200" x2="800" y2="200"></line>
          <line class="manual-field-axis" x1="400" y1="0" x2="400" y2="400"></line>
          <polyline v-if="strokePolyline" class="manual-stroke-guide" :points="strokePolyline"></polyline>
          <circle
            v-for="(point, index) in points"
            :key="`${index}:${point.xUm}:${point.yUm}`"
            class="manual-trap-point"
            :cx="svgX(point.xUm)"
            :cy="svgY(point.yUm)"
            r="3.2"
          ></circle>
          <ellipse
            v-if="cursor && tool === 'ERASE'"
            class="manual-eraser-cursor"
            :cx="svgX(cursor.xUm)"
            :cy="svgY(cursor.yUm)"
            :rx="eraserRadiusX"
            :ry="eraserRadiusY"
          ></ellipse>
          <circle
            v-else-if="cursor"
            class="manual-draw-cursor"
            :cx="svgX(cursor.xUm)"
            :cy="svgY(cursor.yUm)"
            r="6"
          ></circle>
        </svg>
        <div class="manual-drawing-label"><span>{{ tool }} TOOL</span><span>{{ cursorLabel }}</span></div>
        <div class="manual-drawing-scale"><span>&minus;{{ (fieldWidthUm / 2).toFixed(1) }} µm</span><strong>0</strong><span>+{{ (fieldWidthUm / 2).toFixed(1) }} µm</span></div>
      </div>

      <div class="manual-drawing-controls">
        <div class="manual-drawing-heading">
          <div><small>FREEHAND COORDINATE INPUT</small><strong>{{ pointSummary }}</strong></div>
          <span>{{ effectiveSpacingUm.toFixed(3) }} µm PITCH</span>
        </div>

        <div class="manual-tool-choice" role="group" aria-label="Manual drawing tool">
          <button type="button" :class="{ 'is-active': tool === 'DRAW' }" :aria-pressed="tool === 'DRAW'" :disabled="disabled" @click="tool = 'DRAW'">Draw path</button>
          <button type="button" :class="{ 'is-active': tool === 'ERASE' }" :aria-pressed="tool === 'ERASE'" :disabled="disabled" @click="tool = 'ERASE'">Erase</button>
        </div>

        <div class="manual-control-grid">
          <label>FIELD WIDTH / µm
            <input v-model.number="fieldWidthUm" type="number" min="1" step="1" :disabled="disabled" @change="normalizeControls">
          </label>
          <label>FIELD HEIGHT / µm
            <input v-model.number="fieldHeightUm" type="number" min="1" step="1" :disabled="disabled" @change="normalizeControls">
          </label>
          <label>POINT SPACING / µm
            <input v-model.number="pointSpacingUm" type="number" :min="recommendedSpacingUm" step="0.1" :disabled="disabled" @change="normalizeControls">
          </label>
          <label>POINT LIMIT
            <input v-model.number="pointLimit" type="number" min="1" max="512" step="1" :disabled="disabled" @change="normalizeControls">
          </label>
        </div>

        <p class="manual-resolution-note">OPTICAL SAFE MINIMUM / {{ recommendedSpacingUm.toFixed(3) }} µm · CROSSINGS ARE CONSOLIDATED</p>

        <div class="manual-history-actions">
          <button type="button" :disabled="disabled || history.length === 0" @click="undo">Undo</button>
          <button type="button" :disabled="disabled || points.length === 0" @click="clearDrawing">Clear</button>
          <button type="button" :disabled="disabled || sourcePoints.length === 0" @click="loadCurrentPoints">Edit current</button>
        </div>

        <p class="manual-drawing-status" aria-live="polite">{{ statusMessage }}</p>
        <button class="apply-button manual-drawing-apply" type="button" :disabled="disabled || points.length === 0" @click="applyDrawing">
          {{ appliedCount === points.length ? `${appliedCount} hand-drawn tweezers applied` : `Replace tweezers with ${points.length} drawn points` }}
        </button>
      </div>
    </div>
  </div>
</template>
