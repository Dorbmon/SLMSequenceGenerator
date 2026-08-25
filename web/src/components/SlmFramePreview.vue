<script setup lang="ts">
import { onMounted, ref, watch } from "vue";

const props = defineProps<{
  pixels: Uint8Array | Uint16Array | null;
  width: number;
  height: number;
  status: string;
}>();

const canvas = ref<HTMLCanvasElement | null>(null);

function drawPlaceholder(context: CanvasRenderingContext2D): void {
  const width = 760;
  const height = 540;
  context.canvas.width = width;
  context.canvas.height = height;
  context.fillStyle = "#0b1612";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(129,216,208,.07)";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += 38) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
  for (let y = 0; y <= height; y += 38) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  context.fillStyle = "rgba(141,157,149,.66)";
  context.font = "11px 'DM Mono', monospace";
  context.textAlign = "center";
  context.fillText("GENERATE A FRAME TO PREVIEW SLM PHASE CODES", width / 2, height / 2);
  context.textAlign = "start";
}

function draw(): void {
  const context = canvas.value?.getContext("2d");
  if (!context) return;
  const pixels = props.pixels;
  if (!pixels || pixels.length !== props.width * props.height) {
    drawPlaceholder(context);
    return;
  }

  context.canvas.width = props.width;
  context.canvas.height = props.height;
  const image = context.createImageData(props.width, props.height);
  const divisor = pixels instanceof Uint16Array ? 257 : 1;
  for (let index = 0; index < pixels.length; index += 1) {
    const value = Math.round(pixels[index]! / divisor);
    const offset = index * 4;
    image.data[offset] = value;
    image.data[offset + 1] = value;
    image.data[offset + 2] = value;
    image.data[offset + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

watch(
  () => [props.pixels, props.width, props.height],
  draw,
  { flush: "post" },
);

onMounted(draw);
</script>

<template>
  <section class="panel slm-frame-preview" aria-labelledby="slm-preview-title">
    <div class="panel-bar">
      <span id="slm-preview-title"><b class="live-pulse"></b> SLM FRAME PREVIEW</span>
      <span class="frame-counter">{{ width }} &times; {{ height }} / U8</span>
    </div>
    <div class="slm-frame-stage" :style="{ aspectRatio: `${width} / ${height}` }">
      <canvas ref="canvas" class="slm-phase-canvas" :aria-label="`SLM phase-code frame at ${width} by ${height} pixels`"></canvas>
      <div class="canvas-label"><span>{{ status }}</span><span>DISPLAY PLANE / PHASE CODE</span></div>
    </div>
    <div class="preview-footer">
      <span><small>FORMAT</small> 8-bit phase code</span>
      <span><small>OUTPUT</small> Raw SLM pixels</span>
      <span><small>MAP</small> Black 0 / white 255</span>
    </div>
  </section>
</template>
