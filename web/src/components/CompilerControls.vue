<script setup lang="ts">
import ComputationActivity from "./ComputationActivity.vue";
import { MAX_SLM_DIMENSION, MIN_SLM_DIMENSION } from "../lib/resolution.js";
import type { ComputeBackend } from "../workers/compiler-messages.js";

interface CompilerLogLine {
  text: string;
  state?: "active" | "done";
}

const props = defineProps<{
  separation: number;
  iterations: number;
  slmWidth: number;
  slmHeight: number;
  fftWidth: number;
  fftHeight: number;
  phaseMode: string;
  computeBackend: ComputeBackend;
  webgpuAvailable: boolean;
  webgpuStatus: string;
  badge: string;
  running: boolean;
  elapsedMs: number;
  progress: number | null;
  progressLabel: string;
  canExport: boolean;
  logs: readonly CompilerLogLine[];
}>();

const emit = defineEmits<{
  "update:separation": [value: number];
  "update:iterations": [value: number];
  "update:slmWidth": [value: number];
  "update:slmHeight": [value: number];
  "update:phaseMode": [value: string];
  "update:computeBackend": [value: ComputeBackend];
  compile: [];
  cancel: [];
  step: [];
  exportFrames: [];
  exportManifest: [];
}>();

function numberFromEvent(event: Event): number {
  return Number((event.target as HTMLInputElement).value);
}

function stringFromEvent(event: Event): string {
  return (event.target as HTMLSelectElement).value;
}

function backendFromEvent(event: Event): ComputeBackend {
  return (event.target as HTMLSelectElement).value as ComputeBackend;
}

function dimensionFromEvent(event: Event, fallback: number): number {
  const input = event.target as HTMLInputElement;
  if (!Number.isFinite(input.valueAsNumber)) {
    input.value = String(fallback);
    return fallback;
  }
  const value = Math.max(MIN_SLM_DIMENSION, Math.min(MAX_SLM_DIMENSION, Math.round(input.valueAsNumber)));
  input.value = String(value);
  return value;
}
</script>

<template>
  <section class="controls panel" aria-labelledby="controls-title">
    <div class="panel-bar">
      <span class="panel-kicker">INPUT / OUTPUT</span>
      <span class="valid-badge">{{ badge }}</span>
    </div>
    <h2 id="controls-title">Compiler controls</h2>
    <p class="control-description">Set the safety margin and solver budget, then compile the complete offline sequence.</p>

    <div class="control-block">
      <div class="control-label">
        <span>MINIMUM SEPARATION</span>
        <output>{{ separation.toFixed(1) }} um</output>
      </div>
      <input
        type="range"
        min="0.8"
        max="5"
        step="0.1"
        :value="separation"
        :disabled="running"
        @input="emit('update:separation', numberFromEvent($event))"
      >
    </div>
    <div class="control-block">
      <div class="control-label">
        <span>WGS ITERATIONS</span>
        <output>{{ String(iterations).padStart(2, "0") }} / FRAME</output>
      </div>
      <input
        type="range"
        min="2"
        max="12"
        step="1"
        :value="iterations"
        :disabled="running"
        @input="emit('update:iterations', numberFromEvent($event))"
      >
    </div>
    <div class="resolution-block">
      <div class="control-label">
        <span>SLM RESOLUTION</span>
        <output>{{ slmWidth }} &times; {{ slmHeight }} px</output>
      </div>
      <div class="resolution-fields">
        <label>WIDTH / PX
          <input
            type="number"
            :min="MIN_SLM_DIMENSION"
            :max="MAX_SLM_DIMENSION"
            step="1"
            inputmode="numeric"
            :value="slmWidth"
            :disabled="running"
            @change="emit('update:slmWidth', dimensionFromEvent($event, slmWidth))"
          >
        </label>
        <span aria-hidden="true">&times;</span>
        <label>HEIGHT / PX
          <input
            type="number"
            :min="MIN_SLM_DIMENSION"
            :max="MAX_SLM_DIMENSION"
            step="1"
            inputmode="numeric"
            :value="slmHeight"
            :disabled="running"
            @change="emit('update:slmHeight', dimensionFromEvent($event, slmHeight))"
          >
        </label>
      </div>
      <p class="resolution-note">FFT COMPUTE GRID {{ fftWidth }} &times; {{ fftHeight }} / POWER-OF-TWO PADDED</p>
    </div>
    <div class="select-row">
      <label>ASSIGNMENT
        <select :disabled="running">
          <option>Hungarian / squared distance</option>
          <option>Hungarian / weighted obstacles</option>
        </select>
      </label>
      <label>PHASE MODE
        <select :value="phaseMode" :disabled="running" @change="emit('update:phaseMode', stringFromEvent($event))">
          <option>Phase locked WGS</option>
          <option>Soft phase locked</option>
        </select>
      </label>
      <label class="backend-choice">COMPUTE BACKEND
        <select :value="computeBackend" :disabled="running" @change="emit('update:computeBackend', backendFromEvent($event))">
          <option value="wasm">WebAssembly / CPU FFT</option>
          <option value="webgpu" :disabled="!webgpuAvailable">WebGPU / GPU-resident WGS</option>
        </select>
        <small :class="{ 'is-available': webgpuAvailable }">{{ webgpuStatus }}</small>
      </label>
    </div>
    <div class="compile-log" aria-live="polite">
      <div
        v-for="(line, index) in props.logs"
        :key="index"
        class="log-row"
        :class="{ 'is-active': line.state === 'active', 'is-done': line.state === 'done' }"
      >
        <span>{{ String(index + 1).padStart(2, "0") }}</span><span>{{ line.text }}</span><b>&bull;</b>
      </div>
    </div>
    <ComputationActivity
      v-if="running"
      :label="progressLabel"
      detail="DEDICATED WORKER / UI THREAD AVAILABLE"
      :elapsed-ms="elapsedMs"
      :progress="progress"
    />
    <button class="compile-button" :class="{ 'is-running': running }" type="button" @click="running ? emit('cancel') : emit('compile')">
      <span></span> {{ running ? "Cancel compilation" : "Compile sequence" }}
    </button>
    <button class="step-button" type="button" :disabled="running" @click="emit('step')">
      Step one frame <b aria-hidden="true">&#8594;</b>
    </button>
    <button class="export-button" type="button" :disabled="!canExport || running" @click="emit('exportFrames')">
      Export SLM frames <b aria-hidden="true">&#8595;</b>
    </button>
    <button class="export-button secondary-export" type="button" :disabled="!canExport || running" @click="emit('exportManifest')">
      Export manifest <b aria-hidden="true">&#8595;</b>
    </button>
  </section>
</template>
