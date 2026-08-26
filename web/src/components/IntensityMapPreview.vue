<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { ForwardSimulationRegion } from "../lib/forward-simulation.js";

interface TargetMarker {
  x: number;
  y: number;
  label: string;
}

const props = defineProps<{
  intensity: Float32Array | null;
  width: number;
  height: number;
  running: boolean;
  status: string;
  logarithmic: boolean;
  floorDb: number;
  region: ForwardSimulationRegion | null;
  targetMarkers: readonly TargetMarker[];
  physicalAspectRatio: number;
  viewLabel: string;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
let frameRequest = 0;
let pooledSource: Float32Array | null = null;
let pooledWidth = 0;
let pooledHeight = 0;
let pooledIntensity = new Float32Array();
let pooledRegionKey = "";

const normalizedRegion = computed<ForwardSimulationRegion | null>(() => {
  if (props.width <= 0 || props.height <= 0) return null;
  const candidate = props.region ?? { x: 0, y: 0, width: props.width, height: props.height };
  const x = Math.max(0, Math.min(props.width - 1, Math.floor(candidate.x)));
  const y = Math.max(0, Math.min(props.height - 1, Math.floor(candidate.y)));
  const width = Math.max(1, Math.min(props.width - x, Math.floor(candidate.width)));
  const height = Math.max(1, Math.min(props.height - y, Math.floor(candidate.height)));
  return { x, y, width, height };
});
const visibleMarkers = computed(() => {
  const region = normalizedRegion.value;
  if (!region) return [];
  return props.targetMarkers
    .filter((marker) => (
      marker.x >= region.x && marker.x <= region.x + region.width - 1 &&
      marker.y >= region.y && marker.y <= region.y + region.height - 1
    ))
    .map((marker) => ({
      ...marker,
      left: `${(marker.x - region.x) / Math.max(1, region.width - 1) * 100}%`,
      top: `${(marker.y - region.y) / Math.max(1, region.height - 1) * 100}%`,
    }));
});

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
  const region = normalizedRegion.value;
  if (!region) return;
  const regionKey = `${region.x}:${region.y}:${region.width}:${region.height}`;
  if (pooledSource === source && pooledRegionKey === regionKey && pooledWidth > 0 && pooledHeight > 0) return;
  const scale = Math.max(1, region.width / 1024, region.height / 640);
  pooledWidth = Math.max(1, Math.round(region.width / scale));
  pooledHeight = Math.max(1, Math.round(region.height / scale));
  pooledIntensity = new Float32Array(pooledWidth * pooledHeight);
  for (let localY = 0; localY < region.height; localY += 1) {
    const outputY = Math.min(pooledHeight - 1, Math.floor(localY * pooledHeight / region.height));
    const outputRow = outputY * pooledWidth;
    const sourceRow = (region.y + localY) * props.width + region.x;
    for (let localX = 0; localX < region.width; localX += 1) {
      const outputX = Math.min(pooledWidth - 1, Math.floor(localX * pooledWidth / region.width));
      const destination = outputRow + outputX;
      pooledIntensity[destination] = Math.max(pooledIntensity[destination]!, source[sourceRow + localX]!);
    }
  }
  pooledSource = source;
  pooledRegionKey = regionKey;
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
  pooledRegionKey = "";
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
  () => [props.intensity, props.width, props.height, props.logarithmic, props.floorDb, props.region],
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
      <span>{{ normalizedRegion ? `${normalizedRegion.width} × ${normalizedRegion.height} VIEW / ${width} × ${height} FFT` : "FFT GRID / --" }}</span>
    </div>
    <div class="forward-map-stage" :style="{ aspectRatio: String(physicalAspectRatio) }">
      <canvas ref="canvas" aria-label="FFT-shifted simulated focal-plane intensity"></canvas>
      <div v-if="visibleMarkers.length > 0" class="forward-target-markers" aria-hidden="true">
        <i
          v-for="marker in visibleMarkers"
          :key="`${marker.label}:${marker.x}:${marker.y}`"
          :style="{ left: marker.left, top: marker.top }"
        ></i>
      </div>
      <div v-if="running" class="canvas-compute-overlay" aria-hidden="true">
        <div class="field-orbits"><i></i><i></i><i></i><b></b></div>
        <span>FORWARD PROPAGATION / WORKER THREAD</span>
      </div>
      <div class="canvas-label"><span>{{ status }}</span><span>{{ viewLabel }} / NORMALIZED INTENSITY</span></div>
    </div>
    <div class="preview-footer">
      <span><small>ORIGIN</small> FFT-shifted center</span>
      <span><small>DISPLAY</small> {{ logarithmic ? `${floorDb} dB to 0 dB` : "Linear 0 to 1" }}</span>
      <span><small>GEOMETRY</small> Calibrated physical aspect</span>
    </div>
  </section>
</template>
