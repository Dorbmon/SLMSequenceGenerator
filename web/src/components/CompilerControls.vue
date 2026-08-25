<script setup lang="ts">
interface CompilerLogLine {
  text: string;
  state?: "active" | "done";
}

const props = defineProps<{
  separation: number;
  iterations: number;
  phaseMode: string;
  badge: string;
  running: boolean;
  canExport: boolean;
  logs: readonly CompilerLogLine[];
}>();

const emit = defineEmits<{
  "update:separation": [value: number];
  "update:iterations": [value: number];
  "update:phaseMode": [value: string];
  compile: [];
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
    <button class="compile-button" type="button" :disabled="running" @click="emit('compile')">
      <span></span> {{ running ? "Compiling..." : "Compile sequence" }}
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
