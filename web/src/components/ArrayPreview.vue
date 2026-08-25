<script setup lang="ts">
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { InitialAtom, Point2D, TargetSite, TrapFrame } from "../../../src/types.js";

const props = defineProps<{
  initialAtoms: readonly InitialAtom[];
  targetSites: readonly TargetSite[];
  frames: readonly TrapFrame[];
  frame: number;
  total: number;
  separation: number;
  canvasState: string;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);
let context: CanvasRenderingContext2D | null = null;
let resizeObserver: ResizeObserver | null = null;

function project(point: Point2D, width: number, height: number): { x: number; y: number } {
  const scale = Math.min(width, height) / 15;
  return { x: width / 2 + point.xUm * scale, y: height / 2 - point.yUm * scale };
}

function ease(value: number): number {
  const t = Math.max(0, Math.min(1, value));
  return t * t * t * (10 + t * (-15 + 6 * t));
}

function drawGrid(width: number, height: number): void {
  if (!context) return;
  context.save();
  context.strokeStyle = "rgba(129,216,208,.07)";
  context.lineWidth = 1;
  const spacing = Math.min(width, height) / 15;
  for (let x = (width / 2) % spacing; x < width; x += spacing) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = (height / 2) % spacing; y < height; y += spacing) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  context.restore();
}

function drawTarget(point: TargetSite, width: number, height: number, index: number): void {
  if (!context) return;
  const target = project(point, width, height);
  const pulse = 1 + Math.sin(performance.now() / 700 + index) * 0.12;
  context.save();
  context.strokeStyle = "rgba(185,243,107,.7)";
  context.lineWidth = 1;
  context.beginPath();
  context.arc(target.x, target.y, 5 * pulse, 0, Math.PI * 2);
  context.stroke();
  context.fillStyle = "rgba(185,243,107,.12)";
  context.beginPath();
  context.arc(target.x, target.y, 11, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function currentAtoms(): Point2D[] {
  if (props.frames.length > 0) {
    const frame = props.frames[props.frame] ?? props.frames[0];
    return frame
      ? frame.traps
          .filter((trap) => trap.atomId !== null && trap.intensity > 0)
          .map(({ xUm, yUm }) => ({ xUm, yUm }))
      : [];
  }

  const progress = props.frame / Math.max(1, props.total - 1);
  return props.initialAtoms.map((atom, index) => {
    const target = props.targetSites[index % Math.max(1, props.targetSites.length)] ?? atom;
    const local = ease(Math.max(0, Math.min(1, progress * 1.08 - index * 0.018)));
    return {
      xUm: atom.xUm + (target.xUm - atom.xUm) * local,
      yUm: atom.yUm + (target.yUm - atom.yUm) * local,
    };
  });
}

function drawAtom(point: Point2D, width: number, height: number, index: number): void {
  if (!context) return;
  const atom = project(point, width, height);
  const glow = context.createRadialGradient(atom.x, atom.y, 0, atom.x, atom.y, 18);
  glow.addColorStop(0, "rgba(129,216,208,.34)");
  glow.addColorStop(1, "rgba(129,216,208,0)");
  context.fillStyle = glow;
  context.beginPath();
  context.arc(atom.x, atom.y, 18, 0, Math.PI * 2);
  context.fill();
  context.fillStyle = index >= props.targetSites.length ? "#81d8d0" : "#d8eee3";
  context.beginPath();
  context.arc(atom.x, atom.y, 3.2, 0, Math.PI * 2);
  context.fill();
}

function draw(): void {
  if (!canvas.value || !context) return;
  const bounds = canvas.value.getBoundingClientRect();
  const width = bounds.width || 760;
  const height = bounds.height || 540;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0b1612";
  context.fillRect(0, 0, width, height);
  drawGrid(width, height);
  props.targetSites.forEach((target, index) => drawTarget(target, width, height, index));
  currentAtoms().forEach((atom, index) => drawAtom(atom, width, height, index));
  context.save();
  context.fillStyle = "rgba(185,243,107,.8)";
  context.font = "9px 'DM Mono', monospace";
  context.fillText("TARGET FIELD", 18, 25);
  context.fillStyle = "rgba(141,157,149,.75)";
  context.fillText(`${props.separation.toFixed(1)} UM CLEARANCE`, 18, height - 18);
  context.restore();
}

function resizeCanvas(): void {
  if (!canvas.value || !context) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const bounds = canvas.value.getBoundingClientRect();
  canvas.value.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.value.height = Math.max(1, Math.round(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

watch(
  () => [props.initialAtoms, props.targetSites, props.frames, props.frame, props.separation],
  () => nextTick(draw),
  { deep: true },
);

onMounted(() => {
  context = canvas.value?.getContext("2d") ?? null;
  if (!context || !canvas.value) return;
  resizeObserver = new ResizeObserver(resizeCanvas);
  resizeObserver.observe(canvas.value);
  resizeCanvas();
});

onBeforeUnmount(() => {
  resizeObserver?.disconnect();
});
</script>

<template>
  <section class="preview panel">
    <div class="panel-bar">
      <span><b class="live-pulse"></b> ARRAY PREVIEW</span>
      <span class="frame-counter">FRAME {{ String(frame).padStart(2, "0") }} / {{ String(total).padStart(2, "0") }}</span>
    </div>
    <div class="canvas-wrap">
      <canvas ref="canvas" class="array-canvas" width="760" height="540" aria-label="Atom rearrangement preview"></canvas>
      <div class="canvas-label"><span>{{ canvasState }}</span><span>UM PLANE / TOP VIEW</span></div>
    </div>
    <div class="preview-footer">
      <span><small>SOURCE</small> {{ String(initialAtoms.length).padStart(2, "0") }} occupied</span>
      <span><small>TARGET</small> {{ String(targetSites.length).padStart(2, "0") }} required</span>
      <span><small>MODE</small> OFFLINE / SAFE</span>
    </div>
  </section>
</template>
