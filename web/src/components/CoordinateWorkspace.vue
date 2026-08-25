<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import type { InitialAtom, TargetSite } from "../../../src/types.js";
import {
  cloneAtoms,
  cloneTargets,
  coordinateLabel,
  nextPointId,
  parseAtomList,
  parseTargetList,
  serializePoints,
  type CoordinateKind,
  type CoordinatePoint,
} from "../lib/coordinates.js";

type CoordinateTool = "select" | "atom" | "target" | "delete";
type JsonStatus = "synced" | "dirty" | "invalid";

interface Selection {
  kind: CoordinateKind;
  index: number;
}

interface CoordinateEntry extends Selection {
  point: CoordinatePoint;
}

const props = defineProps<{
  initialAtoms: readonly InitialAtom[];
  targetSites: readonly TargetSite[];
  disabled: boolean;
  error: string;
}>();

const emit = defineEmits<{
  change: [initialAtoms: InitialAtom[], targetSites: TargetSite[]];
  error: [message: string];
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
const initialUpload = ref<HTMLInputElement | null>(null);
const targetUpload = ref<HTMLInputElement | null>(null);
const requestUpload = ref<HTMLInputElement | null>(null);
const atoms = ref<InitialAtom[]>(cloneAtoms(props.initialAtoms));
const targets = ref<TargetSite[]>(cloneTargets(props.targetSites));
const initialDraft = ref(serializePoints(atoms.value));
const targetDraft = ref(serializePoints(targets.value));
const jsonStatus = ref<JsonStatus>("synced");
const tool = ref<CoordinateTool>("select");
const selected = ref<Selection | null>(null);
const hover = ref<Selection | null>(null);
const dragging = ref(false);
const snapEnabled = ref(true);
const snapStep = ref(0.5);
const readout = ref("X 0.00 / Y 0.00 UM");
const view = reactive({ centerX: 0, centerY: 0, scale: 32 });
let context: CanvasRenderingContext2D | null = null;
let pointerId: number | null = null;
let resizeObserver: ResizeObserver | null = null;

const entries = computed<CoordinateEntry[]>(() => [
  ...targets.value.map((point, index) => ({ kind: "target" as const, index, point })),
  ...atoms.value.map((point, index) => ({ kind: "atom" as const, index, point })),
]);

const selectedEntry = computed(() => entryForSelection(selected.value));
const inspectorDisabled = computed(() => !selectedEntry.value || props.disabled);
const statusLabel = computed(() => {
  if (jsonStatus.value === "dirty") return "UNAPPLIED CHANGES";
  if (jsonStatus.value === "invalid") return "INVALID JSON";
  return "SYNCHRONIZED";
});

function entryForSelection(selection: Selection | null): CoordinateEntry | null {
  if (!selection) return null;
  const list = selection.kind === "atom" ? atoms.value : targets.value;
  const point = list[selection.index];
  return point ? { ...selection, point } : null;
}

function selectionKey(selection: Selection | null = selected.value): string {
  return selection ? `${selection.kind}:${selection.index}` : "";
}

function pointLabel(entry: CoordinateEntry): string {
  return coordinateLabel(entry.point, entry.kind, entry.index);
}

function syncDrafts(): void {
  initialDraft.value = serializePoints(atoms.value);
  targetDraft.value = serializePoints(targets.value);
  jsonStatus.value = "synced";
}

function markJsonDirty(): void {
  jsonStatus.value = "dirty";
  selected.value = null;
  hover.value = null;
  emit("error", "");
  draw();
}

function publishCoordinates(): void {
  emit("change", cloneAtoms(atoms.value), cloneTargets(targets.value));
}

function applyDraft(options: { fitView?: boolean } = {}): boolean {
  try {
    atoms.value = parseAtomList(initialDraft.value);
    targets.value = parseTargetList(targetDraft.value);
    selected.value = null;
    hover.value = null;
    jsonStatus.value = "synced";
    emit("error", "");
    publishCoordinates();
    if (options.fitView ?? true) nextTick(fitCoordinateView);
    else nextTick(draw);
    return true;
  } catch (error) {
    jsonStatus.value = "invalid";
    emit("error", error instanceof Error ? error.message : "Invalid coordinate input");
    draw();
    return false;
  }
}

function synchronizeBeforeVisualEdit(preserveSelection = false): boolean {
  if (jsonStatus.value === "synced") return true;
  const previous = preserveSelection ? entryForSelection(selected.value) : null;
  const previousId = previous?.kind === "atom"
    ? (previous.point as InitialAtom).atomId
    : (previous?.point as TargetSite | undefined)?.siteId;
  const previousKind = previous?.kind;
  if (!applyDraft({ fitView: true })) return false;
  if (previousKind && previousId !== undefined) {
    const list = previousKind === "atom" ? atoms.value : targets.value;
    const id = previousKind === "atom" ? "atomId" : "siteId";
    const index = list.findIndex((point) => point[id as keyof typeof point] === previousId);
    if (index >= 0) selected.value = { kind: previousKind, index };
  }
  return true;
}

function commitVisualCoordinates(): void {
  syncDrafts();
  emit("error", "");
  publishCoordinates();
  draw();
}

function setSelection(selection: Selection | null): void {
  selected.value = entryForSelection(selection) ? selection : null;
  draw();
}

function setTool(value: CoordinateTool): void {
  tool.value = value;
}

function addPoint(kind: CoordinateKind, point: { xUm: number; yUm: number }): void {
  if (kind === "atom") {
    atoms.value.push({ atomId: nextPointId(atoms.value, kind), ...point });
    selected.value = { kind, index: atoms.value.length - 1 };
  } else {
    targets.value.push({ siteId: nextPointId(targets.value, kind), ...point });
    selected.value = { kind, index: targets.value.length - 1 };
  }
  commitVisualCoordinates();
}

function deleteSelected(): void {
  if (props.disabled || !synchronizeBeforeVisualEdit(true)) return;
  const selection = selected.value;
  if (!entryForSelection(selection)) return;
  const list = selection?.kind === "atom" ? atoms.value : targets.value;
  if (!selection) return;
  list.splice(selection.index, 1);
  selected.value = null;
  hover.value = null;
  commitVisualCoordinates();
  canvas.value?.focus();
}

function snapCoordinate(value: number): number {
  if (!snapEnabled.value) return Math.round(value * 1e3) / 1e3;
  const step = snapStep.value || 0.5;
  const rounded = Math.round(Math.round(value / step) * step * 1e8) / 1e8;
  return Object.is(rounded, -0) ? 0 : rounded;
}

function canvasBounds(): { width: number; height: number } {
  const bounds = canvas.value?.getBoundingClientRect();
  return { width: bounds?.width || 960, height: bounds?.height || 420 };
}

function projectCoordinate(point: { xUm: number; yUm: number }, width: number, height: number): { x: number; y: number } {
  return {
    x: width / 2 + (point.xUm - view.centerX) * view.scale,
    y: height / 2 - (point.yUm - view.centerY) * view.scale,
  };
}

function unprojectCoordinate(x: number, y: number, width: number, height: number): { xUm: number; yUm: number } {
  return {
    xUm: view.centerX + (x - width / 2) / view.scale,
    yUm: view.centerY - (y - height / 2) / view.scale,
  };
}

function fitCoordinateView(): void {
  const { width, height } = canvasBounds();
  const points = entries.value.map((entry) => entry.point);
  if (points.length === 0) {
    view.centerX = 0;
    view.centerY = 0;
    view.scale = Math.min((width - 80) / 12, (height - 70) / 8);
    draw();
    return;
  }
  const minX = Math.min(...points.map((point) => point.xUm));
  const maxX = Math.max(...points.map((point) => point.xUm));
  const minY = Math.min(...points.map((point) => point.yUm));
  const maxY = Math.max(...points.map((point) => point.yUm));
  const spanX = Math.max(12, (maxX - minX) * 1.3);
  const spanY = Math.max(8, (maxY - minY) * 1.3);
  view.centerX = (minX + maxX) / 2;
  view.centerY = (minY + maxY) / 2;
  view.scale = Math.max(0.25, Math.min(240, (width - 80) / spanX, (height - 70) / spanY));
  draw();
}

function zoomCoordinateView(factor: number, anchor: { x: number; y: number } | null = null): void {
  const { width, height } = canvasBounds();
  const point = anchor ?? { x: width / 2, y: height / 2 };
  const world = unprojectCoordinate(point.x, point.y, width, height);
  const scale = Math.max(0.25, Math.min(240, view.scale * factor));
  view.scale = scale;
  view.centerX = world.xUm - (point.x - width / 2) / scale;
  view.centerY = world.yUm + (point.y - height / 2) / scale;
  draw();
}

function niceGridStep(): number {
  const targetUnits = 46 / view.scale;
  const magnitude = 10 ** Math.floor(Math.log10(targetUnits));
  const normalized = targetUnits / magnitude;
  const multiple = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiple * magnitude;
}

function gridLabel(value: number, step: number): string {
  const decimals = Math.max(0, Math.min(4, -Math.floor(Math.log10(step))));
  return (Object.is(value, -0) ? 0 : value).toFixed(decimals);
}

function drawGrid(width: number, height: number): void {
  if (!context) return;
  const step = niceGridStep();
  const upperLeft = unprojectCoordinate(0, 0, width, height);
  const lowerRight = unprojectCoordinate(width, height, width, height);
  const startX = Math.ceil(upperLeft.xUm / step) * step;
  const startY = Math.ceil(lowerRight.yUm / step) * step;
  context.save();
  context.font = "8px 'DM Mono', monospace";
  context.textBaseline = "bottom";
  for (let value = startX; value <= lowerRight.xUm + step * 0.01; value += step) {
    const screen = projectCoordinate({ xUm: value, yUm: 0 }, width, height);
    const axis = Math.abs(value) < step * 1e-6;
    context.strokeStyle = axis ? "rgba(129,216,208,.25)" : "rgba(129,216,208,.07)";
    context.beginPath();
    context.moveTo(screen.x, 0);
    context.lineTo(screen.x, height);
    context.stroke();
    if (!axis) {
      context.fillStyle = "rgba(82,99,91,.65)";
      context.fillText(gridLabel(value, step), screen.x + 4, height - 6);
    }
  }
  context.textBaseline = "top";
  for (let value = startY; value <= upperLeft.yUm + step * 0.01; value += step) {
    const screen = projectCoordinate({ xUm: 0, yUm: value }, width, height);
    const axis = Math.abs(value) < step * 1e-6;
    context.strokeStyle = axis ? "rgba(129,216,208,.25)" : "rgba(129,216,208,.07)";
    context.beginPath();
    context.moveTo(0, screen.y);
    context.lineTo(width, screen.y);
    context.stroke();
    if (!axis) {
      context.fillStyle = "rgba(82,99,91,.65)";
      context.fillText(gridLabel(value, step), 6, screen.y + 4);
    }
  }
  context.restore();
}

function drawPoint(entry: CoordinateEntry, width: number, height: number): void {
  if (!context) return;
  const point = projectCoordinate(entry.point, width, height);
  const key = `${entry.kind}:${entry.index}`;
  const isSelected = key === selectionKey();
  const isHovered = key === selectionKey(hover.value);
  context.save();
  if (entry.kind === "target") {
    context.fillStyle = "rgba(185,243,107,.08)";
    context.beginPath();
    context.arc(point.x, point.y, 13, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "rgba(185,243,107,.88)";
    context.lineWidth = 1.2;
    context.beginPath();
    context.arc(point.x, point.y, 6.5, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(point.x - 3, point.y);
    context.lineTo(point.x + 3, point.y);
    context.moveTo(point.x, point.y - 3);
    context.lineTo(point.x, point.y + 3);
    context.stroke();
  } else {
    const glow = context.createRadialGradient(point.x, point.y, 0, point.x, point.y, 15);
    glow.addColorStop(0, "rgba(129,216,208,.28)");
    glow.addColorStop(1, "rgba(129,216,208,0)");
    context.fillStyle = glow;
    context.beginPath();
    context.arc(point.x, point.y, 15, 0, Math.PI * 2);
    context.fill();
    context.fillStyle = "#bce7df";
    context.beginPath();
    context.arc(point.x, point.y, 4.2, 0, Math.PI * 2);
    context.fill();
  }
  if (isHovered || isSelected) {
    context.strokeStyle = isSelected ? "#ffffff" : "rgba(255,255,255,.48)";
    context.lineWidth = isSelected ? 1.5 : 1;
    context.beginPath();
    context.arc(point.x, point.y, isSelected ? 12 : 10, 0, Math.PI * 2);
    context.stroke();
  }
  if (isSelected) {
    context.fillStyle = "rgba(230,238,232,.9)";
    context.font = "9px 'DM Mono', monospace";
    context.fillText(pointLabel(entry).toUpperCase(), point.x + 16, point.y - 12);
  }
  context.restore();
}

function draw(): void {
  if (!context) return;
  const { width, height } = canvasBounds();
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0b1612";
  context.fillRect(0, 0, width, height);
  drawGrid(width, height);
  targets.value.forEach((point, index) => drawPoint({ kind: "target", index, point }, width, height));
  atoms.value.forEach((point, index) => drawPoint({ kind: "atom", index, point }, width, height));
  if (entries.value.length === 0) {
    context.fillStyle = "rgba(141,157,149,.65)";
    context.font = "11px 'DM Mono', monospace";
    context.textAlign = "center";
    context.fillText("ADD AN ATOM OR TARGET TO BEGIN", width / 2, height / 2);
    context.textAlign = "start";
  }
}

function eventPosition(event: PointerEvent | WheelEvent): { x: number; y: number } {
  const bounds = canvas.value?.getBoundingClientRect();
  return { x: event.clientX - (bounds?.left ?? 0), y: event.clientY - (bounds?.top ?? 0) };
}

function hitTest(position: { x: number; y: number }): Selection | null {
  const { width, height } = canvasBounds();
  for (let index = entries.value.length - 1; index >= 0; index -= 1) {
    const entry = entries.value[index];
    if (!entry) continue;
    const projected = projectCoordinate(entry.point, width, height);
    if (Math.hypot(projected.x - position.x, projected.y - position.y) <= 14) {
      return { kind: entry.kind, index: entry.index };
    }
  }
  return null;
}

function beginDrag(event: PointerEvent): void {
  if (!canvas.value) return;
  dragging.value = true;
  pointerId = event.pointerId;
  canvas.value.setPointerCapture(event.pointerId);
}

function endDrag(event: PointerEvent): void {
  if (pointerId !== null && event.pointerId !== pointerId) return;
  if (canvas.value && pointerId !== null && canvas.value.hasPointerCapture(pointerId)) {
    canvas.value.releasePointerCapture(pointerId);
  }
  dragging.value = false;
  pointerId = null;
}

function handlePointerDown(event: PointerEvent): void {
  if (event.button !== 0 || dragging.value || props.disabled || !synchronizeBeforeVisualEdit()) return;
  event.preventDefault();
  canvas.value?.focus();
  const position = eventPosition(event);
  const hit = hitTest(position);
  if (tool.value === "delete") {
    if (hit) {
      setSelection(hit);
      deleteSelected();
    }
    return;
  }
  if (tool.value === "atom" || tool.value === "target") {
    const { width, height } = canvasBounds();
    const point = unprojectCoordinate(position.x, position.y, width, height);
    addPoint(tool.value, { xUm: snapCoordinate(point.xUm), yUm: snapCoordinate(point.yUm) });
    beginDrag(event);
    return;
  }
  setSelection(hit);
  if (hit) beginDrag(event);
}

function handlePointerMove(event: PointerEvent): void {
  const position = eventPosition(event);
  const { width, height } = canvasBounds();
  const world = unprojectCoordinate(position.x, position.y, width, height);
  readout.value = `X ${world.xUm.toFixed(2)} / Y ${world.yUm.toFixed(2)} UM`;
  if (dragging.value && event.pointerId === pointerId) {
    const entry = selectedEntry.value;
    if (!entry) return;
    const xUm = snapCoordinate(world.xUm);
    const yUm = snapCoordinate(world.yUm);
    if (entry.point.xUm !== xUm || entry.point.yUm !== yUm) {
      entry.point.xUm = xUm;
      entry.point.yUm = yUm;
      commitVisualCoordinates();
    }
    return;
  }
  hover.value = hitTest(position);
  draw();
}

function handlePointerLeave(): void {
  if (dragging.value) return;
  hover.value = null;
  draw();
}

function handleWheel(event: WheelEvent): void {
  if (props.disabled) return;
  zoomCoordinateView(Math.exp(-event.deltaY * 0.0015), eventPosition(event));
}

function handleKeyDown(event: KeyboardEvent): void {
  if (props.disabled) return;
  const key = event.key.toLowerCase();
  if (["v", "a", "t"].includes(key) && !event.metaKey && !event.ctrlKey && !event.altKey) {
    setTool(key === "v" ? "select" : key === "a" ? "atom" : "target");
    event.preventDefault();
    return;
  }
  if (event.key === "Delete" || event.key === "Backspace") {
    deleteSelected();
    event.preventDefault();
    return;
  }
  if (event.key === "Escape") {
    setTool("select");
    setSelection(null);
    return;
  }
  const directions: Partial<Record<string, readonly [number, number]>> = {
    ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1],
  };
  const direction = directions[event.key];
  if (!direction || !synchronizeBeforeVisualEdit(true)) return;
  const entry = selectedEntry.value;
  if (!entry) return;
  const base = snapEnabled.value ? snapStep.value || 0.5 : 0.1;
  const amount = base * (event.shiftKey ? 10 : 1);
  entry.point.xUm = snapCoordinate(entry.point.xUm + direction[0] * amount);
  entry.point.yUm = snapCoordinate(entry.point.yUm + direction[1] * amount);
  commitVisualCoordinates();
  event.preventDefault();
}

function selectFromInspector(event: Event): void {
  if (!synchronizeBeforeVisualEdit()) return;
  const value = (event.target as HTMLSelectElement).value;
  const match = /^(atom|target):(\d+)$/.exec(value);
  setSelection(match ? { kind: match[1] as CoordinateKind, index: Number(match[2]) } : null);
}

function updateSelectedCoordinate(axis: "xUm" | "yUm", event: Event): void {
  if (!synchronizeBeforeVisualEdit(true)) return;
  const entry = selectedEntry.value;
  const value = Number((event.target as HTMLInputElement).value);
  if (!entry || !Number.isFinite(value)) {
    emit("error", "Selected coordinates must be finite numbers");
    return;
  }
  entry.point[axis] = value;
  commitVisualCoordinates();
}

function formatCoordinate(value: number): string {
  const rounded = Math.round(value * 1e6) / 1e6;
  return String(Object.is(rounded, -0) ? 0 : rounded);
}

async function importJson(file: File, target: "initial" | "target" | "request"): Promise<void> {
  try {
    const parsed: unknown = JSON.parse(await file.text());
    if (target === "request") {
      if (!isRecord(parsed) || !Array.isArray(parsed.initialAtoms) || !Array.isArray(parsed.targetSites)) {
        throw new Error("Expected a full request with initialAtoms and targetSites arrays");
      }
      initialDraft.value = JSON.stringify(parsed.initialAtoms, null, 2);
      targetDraft.value = JSON.stringify(parsed.targetSites, null, 2);
    } else {
      const property = target === "initial" ? "initialAtoms" : "targetSites";
      const list = Array.isArray(parsed)
        ? parsed
        : isRecord(parsed) && Array.isArray(parsed.points)
          ? parsed.points
          : isRecord(parsed) && Array.isArray(parsed[property])
            ? parsed[property]
            : null;
      if (!list) throw new Error("Expected a point array or a full request JSON object");
      if (target === "initial") initialDraft.value = JSON.stringify(list, null, 2);
      else targetDraft.value = JSON.stringify(list, null, 2);
    }
    applyDraft();
  } catch (error) {
    jsonStatus.value = "invalid";
    emit("error", error instanceof Error ? error.message : "Unable to import JSON");
  }
}

function handleUpload(event: Event, target: "initial" | "target" | "request"): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) void importJson(file, target);
  input.value = "";
}

function resizeCanvas(fit = false): void {
  if (!canvas.value || !context) return;
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.value.getBoundingClientRect();
  canvas.value.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.value.height = Math.max(1, Math.round(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  if (fit) fitCoordinateView();
  else draw();
}

function resetEditor(): void {
  atoms.value = cloneAtoms(props.initialAtoms);
  targets.value = cloneTargets(props.targetSites);
  selected.value = null;
  hover.value = null;
  tool.value = "select";
  syncDrafts();
  emit("error", "");
  nextTick(() => resizeCanvas(true));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

watch(
  () => [props.initialAtoms, props.targetSites] as const,
  ([nextAtoms, nextTargets]) => {
    atoms.value = cloneAtoms(nextAtoms);
    targets.value = cloneTargets(nextTargets);
    if (jsonStatus.value === "synced") syncDrafts();
    nextTick(draw);
  },
);

watch(snapStep, () => nextTick(draw));

onMounted(() => {
  context = canvas.value?.getContext("2d") ?? null;
  if (!context || !canvas.value) return;
  resizeObserver = new ResizeObserver(() => resizeCanvas(false));
  resizeObserver.observe(canvas.value);
  requestAnimationFrame(() => resizeCanvas(true));
});

onBeforeUnmount(() => resizeObserver?.disconnect());

defineExpose({ applyDraft, fitCoordinateView, resetEditor });
</script>

<template>
  <section class="data-panel panel" aria-labelledby="data-title">
    <div class="panel-bar">
      <span id="data-title" class="panel-kicker">COORDINATES / VISUAL + JSON</span>
      <span class="data-format">UM PLANE / DIRECT EDIT</span>
    </div>
    <div class="coordinate-visual" :class="{ 'is-disabled': disabled }">
      <div class="coordinate-toolbar" role="toolbar" aria-label="Coordinate editor tools">
        <div class="coordinate-tool-group">
          <button class="coordinate-tool" :class="{ 'is-active': tool === 'select' }" type="button" :aria-pressed="tool === 'select'" :disabled="disabled" @click="setTool('select')"><span aria-hidden="true">&#8599;</span> Select / move</button>
          <button class="coordinate-tool" :class="{ 'is-active': tool === 'atom' }" type="button" :aria-pressed="tool === 'atom'" :disabled="disabled" @click="setTool('atom')"><span class="tool-dot atom-dot" aria-hidden="true"></span> Add atom</button>
          <button class="coordinate-tool" :class="{ 'is-active': tool === 'target' }" type="button" :aria-pressed="tool === 'target'" :disabled="disabled" @click="setTool('target')"><span class="tool-dot target-dot" aria-hidden="true"></span> Add target</button>
          <button class="coordinate-tool" :class="{ 'is-active': tool === 'delete' }" type="button" :aria-pressed="tool === 'delete'" :disabled="disabled" @click="setTool('delete')"><span aria-hidden="true">&#215;</span> Remove</button>
        </div>
        <div class="coordinate-view-tools">
          <label class="snap-control"><input v-model="snapEnabled" type="checkbox" :disabled="disabled"> SNAP
            <select v-model.number="snapStep" aria-label="Coordinate snap interval" :disabled="disabled">
              <option :value="0.1">0.1 um</option>
              <option :value="0.25">0.25 um</option>
              <option :value="0.5">0.5 um</option>
              <option :value="1">1.0 um</option>
            </select>
          </label>
          <div class="zoom-controls" aria-label="Coordinate view zoom">
            <button type="button" aria-label="Zoom out" :disabled="disabled" @click="zoomCoordinateView(0.8)">&#8722;</button>
            <button type="button" aria-label="Zoom in" :disabled="disabled" @click="zoomCoordinateView(1.25)">+</button>
            <button type="button" :disabled="disabled" @click="fitCoordinateView">Fit</button>
          </div>
        </div>
      </div>
      <div class="coordinate-layout">
        <div class="coordinate-stage">
          <canvas
            ref="canvas"
            width="960"
            height="420"
            tabindex="0"
            :data-tool="tool"
            :class="{ 'is-dragging': dragging }"
            aria-label="Visual coordinate editor. Select a tool, then click or drag points in the micrometer plane."
            @pointerdown="handlePointerDown"
            @pointermove="handlePointerMove"
            @pointerup="endDrag"
            @pointercancel="endDrag"
            @pointerleave="handlePointerLeave"
            @wheel.prevent="handleWheel"
            @keydown="handleKeyDown"
          ></canvas>
          <div class="coordinate-legend" aria-hidden="true"><span><i class="atom-key"></i>Initial atom</span><span><i class="target-key"></i>Target site</span></div>
          <output class="coordinate-readout">{{ readout }}</output>
        </div>
        <aside class="coordinate-inspector" aria-label="Selected coordinate">
          <p class="inspector-kicker">POINT INSPECTOR</p>
          <label>SELECTED POINT
            <select :value="selectionKey()" aria-label="Selected atom or target" :disabled="disabled || jsonStatus !== 'synced' || entries.length === 0" @change="selectFromInspector">
              <option value="">No point selected</option>
              <optgroup v-if="atoms.length" label="Initial atoms">
                <option v-for="(point, index) in atoms" :key="`atom-${point.atomId ?? index}`" :value="`atom:${index}`">{{ coordinateLabel(point, "atom", index) }}</option>
              </optgroup>
              <optgroup v-if="targets.length" label="Target sites">
                <option v-for="(point, index) in targets" :key="`target-${point.siteId ?? index}`" :value="`target:${index}`">{{ coordinateLabel(point, "target", index) }}</option>
              </optgroup>
            </select>
          </label>
          <div class="coordinate-fields">
            <label>X / UM<input type="number" :step="snapStep" inputmode="decimal" :value="selectedEntry ? formatCoordinate(selectedEntry.point.xUm) : ''" :disabled="inspectorDisabled" @change="updateSelectedCoordinate('xUm', $event)"></label>
            <label>Y / UM<input type="number" :step="snapStep" inputmode="decimal" :value="selectedEntry ? formatCoordinate(selectedEntry.point.yUm) : ''" :disabled="inspectorDisabled" @change="updateSelectedCoordinate('yUm', $event)"></label>
          </div>
          <button class="delete-point-button" type="button" :disabled="inspectorDisabled" @click="deleteSelected">Delete selected point</button>
          <p class="coordinate-instructions">Drag to move. Scroll to zoom. Arrow keys nudge the selected point; press Delete to remove it.</p>
        </aside>
      </div>
    </div>
    <div class="data-subbar">
      <span>JSON INPUT / ADVANCED</span>
      <span class="json-status" :class="{ 'is-dirty': jsonStatus === 'dirty', 'is-invalid': jsonStatus === 'invalid' }">{{ statusLabel }}</span>
    </div>
    <div class="data-grid">
      <label class="editor-block">INITIAL ATOMS
        <textarea v-model="initialDraft" spellcheck="false" aria-label="Initial atom coordinates" :disabled="disabled" @input="markJsonDirty"></textarea>
        <span class="editor-actions"><button class="upload-button" type="button" :disabled="disabled" @click="initialUpload?.click()">Upload JSON</button><input ref="initialUpload" type="file" accept="application/json,.json" hidden @change="handleUpload($event, 'initial')"></span>
      </label>
      <label class="editor-block">TARGET SITES
        <textarea v-model="targetDraft" spellcheck="false" aria-label="Target site coordinates" :disabled="disabled" @input="markJsonDirty"></textarea>
        <span class="editor-actions"><button class="upload-button" type="button" :disabled="disabled" @click="targetUpload?.click()">Upload JSON</button><input ref="targetUpload" type="file" accept="application/json,.json" hidden @change="handleUpload($event, 'target')"></span>
      </label>
    </div>
    <div class="data-footer">
      <div class="data-help">Use <code>[xUm, yUm]</code> or <code>{"xUm": 0, "yUm": 0}</code>. Coordinates use +x right and +y up.</div>
      <div class="data-actions">
        <button class="upload-button request-button" type="button" :disabled="disabled" @click="requestUpload?.click()">Import request</button>
        <input ref="requestUpload" type="file" accept="application/json,.json" hidden @change="handleUpload($event, 'request')">
        <button class="apply-button" type="button" :disabled="disabled" @click="applyDraft()">Apply coordinates</button>
      </div>
    </div>
    <p class="input-error" :class="{ 'is-visible': Boolean(error) }" role="alert">{{ error }}</p>
  </section>
</template>
