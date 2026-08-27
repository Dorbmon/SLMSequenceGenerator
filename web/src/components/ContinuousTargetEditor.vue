<script setup lang="ts">
import { onMounted, ref, shallowRef } from "vue";

export interface ContinuousTargetChange {
  intensity: Float32Array;
  width: number;
  height: number;
}

const WIDTH = 640;
const HEIGHT = 400;
const MAX_HISTORY = 12;

const props = defineProps<{ disabled: boolean }>();
const emit = defineEmits<{ change: [value: ContinuousTargetChange] }>();
const canvas = ref<HTMLCanvasElement | null>(null);
const upload = ref<HTMLInputElement | null>(null);
const tool = ref<"draw" | "erase">("draw");
const brushRadius = ref(18);
const brushIntensity = ref(1);
const filename = ref("");
const errorMessage = ref("");
const history = shallowRef<Float32Array<ArrayBufferLike>[]>([]);
let values: Float32Array<ArrayBufferLike> = new Float32Array(WIDTH * HEIGHT);
let drawing = false;
let lastPoint: { x: number; y: number } | null = null;

function reset(): void {
  values = defaultTarget();
  filename.value = "Built-in continuous target";
  errorMessage.value = "";
  history.value = [];
  draw();
  emitChange();
}

function defaultTarget(): Float32Array {
  const output = new Float32Array(WIDTH * HEIGHT);
  const centerX = WIDTH / 2;
  const centerY = HEIGHT / 2;
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const nx = (x - centerX) / 112;
      const ny = (y - centerY) / 112;
      const radius = Math.hypot(nx, ny);
      const ring = Math.exp(-(((radius - 0.72) / 0.055) ** 2));
      const bar = Math.exp(-((nx / 0.045) ** 2)) * Math.exp(-((ny / 0.52) ** 8));
      output[y * WIDTH + x] = Math.min(1, Math.max(ring, bar * 0.78));
    }
  }
  return output;
}

function emitChange(): void {
  emit("change", { intensity: new Float32Array(values), width: WIDTH, height: HEIGHT });
}

function draw(): void {
  const context = canvas.value?.getContext("2d");
  if (!context) return;
  if (context.canvas.width !== WIDTH) context.canvas.width = WIDTH;
  if (context.canvas.height !== HEIGHT) context.canvas.height = HEIGHT;
  const image = context.createImageData(WIDTH, HEIGHT);
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.round(Math.max(0, Math.min(1, values[index]!)) * 255);
    const offset = index * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function pointerPosition(event: PointerEvent): { x: number; y: number } {
  const bounds = canvas.value!.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) / bounds.width * WIDTH,
    y: (event.clientY - bounds.top) / bounds.height * HEIGHT,
  };
}

function beginStroke(event: PointerEvent): void {
  if (props.disabled || event.button !== 0 || !canvas.value) return;
  pushHistory();
  drawing = true;
  canvas.value.setPointerCapture(event.pointerId);
  lastPoint = pointerPosition(event);
  paintSegment(lastPoint, lastPoint);
}

function continueStroke(event: PointerEvent): void {
  if (!drawing || !lastPoint) return;
  const point = pointerPosition(event);
  paintSegment(lastPoint, point);
  lastPoint = point;
}

function endStroke(event: PointerEvent): void {
  if (!drawing) return;
  drawing = false;
  lastPoint = null;
  if (canvas.value?.hasPointerCapture(event.pointerId)) canvas.value.releasePointerCapture(event.pointerId);
  emitChange();
}

function paintSegment(from: { x: number; y: number }, to: { x: number; y: number }): void {
  const distance = Math.hypot(to.x - from.x, to.y - from.y);
  const spacing = Math.max(1, brushRadius.value * 0.24);
  const steps = Math.max(1, Math.ceil(distance / spacing));
  for (let step = 0; step <= steps; step += 1) {
    const fraction = step / steps;
    paintPoint(
      from.x + (to.x - from.x) * fraction,
      from.y + (to.y - from.y) * fraction,
    );
  }
  draw();
}

function paintPoint(centerX: number, centerY: number): void {
  const radius = brushRadius.value;
  const minimumX = Math.max(0, Math.floor(centerX - radius));
  const maximumX = Math.min(WIDTH - 1, Math.ceil(centerX + radius));
  const minimumY = Math.max(0, Math.floor(centerY - radius));
  const maximumY = Math.min(HEIGHT - 1, Math.ceil(centerY + radius));
  for (let y = minimumY; y <= maximumY; y += 1) {
    for (let x = minimumX; x <= maximumX; x += 1) {
      const normalized = Math.hypot(x - centerX, y - centerY) / radius;
      if (normalized > 1) continue;
      const edge = normalized <= 0.78 ? 1 : 0.5 + 0.5 * Math.cos(Math.PI * (normalized - 0.78) / 0.22);
      const index = y * WIDTH + x;
      const amount = brushIntensity.value * edge;
      values[index] = tool.value === "draw"
        ? Math.max(values[index]!, amount)
        : values[index]! * (1 - amount);
    }
  }
}

function pushHistory(): void {
  history.value = [...history.value.slice(-(MAX_HISTORY - 1)), new Float32Array(values)];
}

function undo(): void {
  const previous = history.value.at(-1);
  if (!previous || props.disabled) return;
  values = previous;
  history.value = history.value.slice(0, -1);
  filename.value = "Edited target";
  draw();
  emitChange();
}

function clear(): void {
  if (props.disabled) return;
  pushHistory();
  values = new Float32Array(WIDTH * HEIGHT);
  filename.value = "Empty target";
  draw();
  emitChange();
}

function invert(): void {
  if (props.disabled) return;
  pushHistory();
  for (let index = 0; index < values.length; index += 1) values[index] = 1 - values[index]!;
  filename.value = "Inverted target";
  draw();
  emitChange();
}

async function importImage(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file || props.disabled) return;
  try {
    const bitmap = await createImageBitmap(file);
    const surface = document.createElement("canvas");
    surface.width = WIDTH;
    surface.height = HEIGHT;
    const context = surface.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("The browser could not create an image canvas");
    context.fillStyle = "#000";
    context.fillRect(0, 0, WIDTH, HEIGHT);
    const scale = Math.min(WIDTH / bitmap.width, HEIGHT / bitmap.height);
    const width = bitmap.width * scale;
    const height = bitmap.height * scale;
    context.drawImage(bitmap, (WIDTH - width) / 2, (HEIGHT - height) / 2, width, height);
    bitmap.close();
    const data = context.getImageData(0, 0, WIDTH, HEIGHT).data;
    const next = new Float32Array(WIDTH * HEIGHT);
    let maximum = 0;
    for (let index = 0; index < next.length; index += 1) {
      const offset = index * 4;
      const alpha = data[offset + 3]! / 255;
      const luminance = (0.2126 * data[offset]! + 0.7152 * data[offset + 1]! + 0.0722 * data[offset + 2]!) / 255 * alpha;
      next[index] = luminance;
      maximum = Math.max(maximum, luminance);
    }
    if (maximum <= 1e-6) throw new Error("The uploaded image contains no visible intensity");
    pushHistory();
    for (let index = 0; index < next.length; index += 1) next[index] = next[index]! / maximum;
    values = next;
    filename.value = file.name;
    errorMessage.value = "";
    draw();
    emitChange();
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : "Unable to decode the target image";
  }
}

onMounted(reset);
defineExpose({ reset });
</script>

<template>
  <section class="panel continuous-target-editor" aria-labelledby="continuous-editor-title">
    <div class="panel-bar">
      <span id="continuous-editor-title">Target intensity</span>
      <span>{{ filename || "No target" }} / {{ WIDTH }} × {{ HEIGHT }}</span>
    </div>
    <div class="continuous-editor-body">
      <div class="continuous-canvas-wrap">
        <canvas
          ref="canvas"
          class="continuous-target-canvas"
          :class="`is-${tool}`"
          aria-label="Draw the continuous target intensity field"
          @pointerdown="beginStroke"
          @pointermove="continueStroke"
          @pointerup="endStroke"
          @pointercancel="endStroke"
        ></canvas>
        <span>Target intensity / brighter means stronger</span>
      </div>
      <div class="continuous-editor-tools">
        <input ref="upload" type="file" accept="image/*" hidden @change="importImage">
        <button type="button" :disabled="disabled" @click="upload?.click()">Upload image</button>
        <div class="continuous-tool-choice" aria-label="Drawing tool">
          <button type="button" :class="{ 'is-active': tool === 'draw' }" :disabled="disabled" @click="tool = 'draw'">Draw</button>
          <button type="button" :class="{ 'is-active': tool === 'erase' }" :disabled="disabled" @click="tool = 'erase'">Erase</button>
        </div>
        <label>
          Brush size <output>{{ brushRadius }} px</output>
          <input v-model.number="brushRadius" type="range" min="2" max="72" step="1" :disabled="disabled">
        </label>
        <label>
          Intensity <output>{{ Math.round(brushIntensity * 100) }}%</output>
          <input v-model.number="brushIntensity" type="range" min="0.05" max="1" step="0.05" :disabled="disabled">
        </label>
        <div class="continuous-edit-actions">
          <button type="button" :disabled="disabled || history.length === 0" @click="undo">Undo</button>
          <button type="button" :disabled="disabled" @click="invert">Invert</button>
          <button type="button" :disabled="disabled" @click="clear">Clear</button>
        </div>
        <p v-if="errorMessage" class="inline-error">{{ errorMessage }}</p>
        <p>Upload any grayscale image, or draw a continuous shape directly. Black is zero target intensity.</p>
      </div>
    </div>
  </section>
</template>
