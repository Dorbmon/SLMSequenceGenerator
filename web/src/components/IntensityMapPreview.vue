<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from "vue";

const props = defineProps<{
  intensity: Float32Array | null;
  width: number;
  height: number;
  running: boolean;
  status: string;
  logarithmic: boolean;
  floorDb: number;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
let frameRequest = 0;
let pooledSource: Float32Array | null = null;
let pooledWidth = 0;
let pooledHeight = 0;
let pooledIntensity = new Float32Array();

function scheduleDraw(): void {
  cancelAnimationFrame(frameRequest);
  frameRequest = requestAnimationFrame(draw);
}

function draw(): void {
  const context = canvas.value?.getContext("2d");
  if (!context) return;
  const source = props.intensity;
  if (!source || source.length !== props.width * props.height || props.width <= 0 || props.height <= 0) {
    drawPlaceholder(context);
    return;
  }
  updatePool(source);
  context.canvas.width = pooledWidth;
  context.canvas.height = pooledHeight;
  const image = context.createImageData(pooledWidth, pooledHeight);
  for (let index = 0; index < pooledIntensity.length; index += 1) {
    const value = displayValue(pooledIntensity[index]!);
    const [red, green, blue] = heatColor(value);
    const offset = index * 4;
    image.data[offset] = red;
    image.data[offset + 1] = green;
    image.data[offset + 2] = blue;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function updatePool(source: Float32Array): void {
  if (pooledSource === source && pooledWidth > 0 && pooledHeight > 0) return;
  const scale = Math.max(1, props.width / 1024, props.height / 640);
  pooledWidth = Math.max(1, Math.round(props.width / scale));
  pooledHeight = Math.max(1, Math.round(props.height / scale));
  pooledIntensity = new Float32Array(pooledWidth * pooledHeight);
  const xMap = Uint32Array.from({ length: props.width }, (_, x) => Math.min(pooledWidth - 1, Math.floor(x * pooledWidth / props.width)));
  const yMap = Uint32Array.from({ length: props.height }, (_, y) => Math.min(pooledHeight - 1, Math.floor(y * pooledHeight / props.height)));
  for (let y = 0; y < props.height; y += 1) {
    const outputRow = yMap[y]! * pooledWidth;
    const sourceRow = y * props.width;
    for (let x = 0; x < props.width; x += 1) {
      const destination = outputRow + xMap[x]!;
      pooledIntensity[destination] = Math.max(pooledIntensity[destination]!, source[sourceRow + x]!);
    }
  }
  pooledSource = source;
}

function displayValue(value: number): number {
  if (!props.logarithmic) return clamp(value, 0, 1);
  const decibels = 10 * Math.log10(Math.max(value, 1e-12));
  return clamp((decibels - props.floorDb) / -props.floorDb, 0, 1);
}

function heatColor(value: number): [number, number, number] {
  if (value < 0.34) {
    const t = value / 0.34;
    return [Math.round(4 + 7 * t), Math.round(10 + 87 * t), Math.round(12 + 105 * t)];
  }
  if (value < 0.76) {
    const t = (value - 0.34) / 0.42;
    return [Math.round(11 + 174 * t), Math.round(97 + 146 * t), Math.round(117 - 10 * t)];
  }
  const t = (value - 0.76) / 0.24;
  return [Math.round(185 + 70 * t), Math.round(243 + 12 * t), Math.round(107 + 148 * t)];
}

function drawPlaceholder(context: CanvasRenderingContext2D): void {
  pooledSource = null;
  context.canvas.width = 760;
  context.canvas.height = 430;
  context.fillStyle = "#07100d";
  context.fillRect(0, 0, context.canvas.width, context.canvas.height);
  context.strokeStyle = "rgba(129,216,208,.07)";
  for (let x = 0; x <= context.canvas.width; x += 38) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, context.canvas.height);
    context.stroke();
  }
  for (let y = 0; y <= context.canvas.height; y += 38) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(context.canvas.width, y);
    context.stroke();
  }
  context.fillStyle = "rgba(141,157,149,.66)";
  context.font = "11px 'DM Mono', monospace";
  context.textAlign = "center";
  context.fillText("UPLOAD AN SLM FRAME TO SIMULATE ITS FOCAL INTENSITY", context.canvas.width / 2, context.canvas.height / 2);
  context.textAlign = "start";
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

watch(
  () => [props.intensity, props.width, props.height, props.logarithmic, props.floorDb],
  scheduleDraw,
  { flush: "post" },
);

onMounted(scheduleDraw);
onBeforeUnmount(() => cancelAnimationFrame(frameRequest));
</script>

<template>
  <section class="forward-map-preview" aria-labelledby="forward-map-title">
    <div class="panel-bar">
      <span id="forward-map-title"><b class="live-pulse"></b> SIMULATED INTENSITY MAP</span>
      <span>{{ width > 0 && height > 0 ? `${width} × ${height}` : "FFT GRID / --" }}</span>
    </div>
    <div class="forward-map-stage" :style="width > 0 && height > 0 ? { aspectRatio: `${width} / ${height}` } : undefined">
      <canvas ref="canvas" aria-label="FFT-shifted simulated focal-plane intensity"></canvas>
      <div v-if="running" class="canvas-compute-overlay" aria-hidden="true">
        <div class="field-orbits"><i></i><i></i><i></i><b></b></div>
        <span>FORWARD PROPAGATION / WORKER THREAD</span>
      </div>
      <div class="canvas-label"><span>{{ status }}</span><span>FOCAL PLANE / NORMALIZED INTENSITY</span></div>
    </div>
    <div class="preview-footer">
      <span><small>ORIGIN</small> FFT-shifted center</span>
      <span><small>DISPLAY</small> {{ logarithmic ? `${floorDb} dB to 0 dB` : "Linear 0 to 1" }}</span>
      <span><small>POOLING</small> Peak-preserving preview</span>
    </div>
  </section>
</template>
