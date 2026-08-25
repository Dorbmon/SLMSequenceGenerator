<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, shallowRef } from "vue";
import {
  SlmSequenceCompiler,
  type CalibrationPackage,
  type CompileProgress,
  type CompiledSequenceHandle,
  type InitialAtom,
  type TargetSite,
  type TrapFrame,
} from "../../src/index.js";
import ArrayPreview from "./components/ArrayPreview.vue";
import CompilerControls from "./components/CompilerControls.vue";
import CoordinateWorkspace from "./components/CoordinateWorkspace.vue";
import { DEFAULT_INITIAL_ATOMS, DEFAULT_TARGET_SITES } from "./data/defaults.js";
import { cloneAtoms, cloneTargets } from "./lib/coordinates.js";

interface CoordinateEditorHandle {
  applyDraft(options?: { fitView?: boolean }): boolean;
  resetEditor(): void;
}

interface LogLine {
  text: string;
  state?: "active" | "done";
}

type CompilationState = "idle" | "running" | "accepted" | "rejected";

const coordinateEditor = ref<CoordinateEditorHandle | null>(null);
const initialAtoms = ref<InitialAtom[]>(cloneAtoms(DEFAULT_INITIAL_ATOMS));
const targetSites = ref<TargetSite[]>(cloneTargets(DEFAULT_TARGET_SITES));
const sequence = shallowRef<CompiledSequenceHandle | null>(null);
const frames = shallowRef<TrapFrame[]>([]);
const frame = ref(0);
const total = ref(64);
const separation = ref(1.2);
const iterations = ref(4);
const phaseMode = ref("Phase locked WGS");
const running = ref(false);
const compilationState = ref<CompilationState>("idle");
const inputError = ref("");
const logs = ref<LogLine[]>(defaultLogs());
let playbackFrame = 0;
let compileGeneration = 0;

const badge = computed(() => {
  if (running.value) return "PROCESSING";
  if (compilationState.value === "rejected") return "REJECTED";
  if (inputError.value) return "CHECK INPUT";
  if (sequence.value) return "ACCEPTED";
  return "VALIDATED";
});

const canvasState = computed(() => {
  if (running.value) return "SOLVING / SEQUENTIAL";
  if (sequence.value) return "SEQUENCE ACCEPTED";
  return "READY";
});

const averageEfficiency = computed(() => {
  const metrics = sequence.value?.frameMetrics ?? [];
  return metrics.length
    ? metrics.reduce((sum, metric) => sum + metric.diffractionEfficiency, 0) / metrics.length
    : null;
});

const metricClearance = computed(() => {
  const measured = sequence.value?.validation.minimumAtomSeparationUm;
  return measured !== undefined && Number.isFinite(measured) ? measured.toFixed(1) : separation.value.toFixed(1);
});

const metricCost = computed(() => sequence.value?.manifest.assignmentCost.toFixed(1) ?? "--");

function defaultLogs(): LogLine[] {
  return [
    { text: "Awaiting input arrays" },
    { text: "No conflicts detected" },
    { text: "Hologram state is idle" },
  ];
}

function setLog(index: number, text: string, state?: LogLine["state"]): void {
  const next = logs.value.slice();
  next[index] = state ? { text, state } : { text };
  logs.value = next;
}

function stopPlayback(): void {
  cancelAnimationFrame(playbackFrame);
  playbackFrame = 0;
}

function invalidateSequence(): void {
  stopPlayback();
  sequence.value = null;
  frames.value = [];
  frame.value = 0;
  total.value = 64;
  compilationState.value = "idle";
  logs.value = [
    { text: "Coordinates changed" },
    { text: "Compile to validate paths" },
    { text: "Hologram state is idle" },
  ];
}

function updateCoordinates(nextAtoms: InitialAtom[], nextTargets: TargetSite[]): void {
  initialAtoms.value = cloneAtoms(nextAtoms);
  targetSites.value = cloneTargets(nextTargets);
  inputError.value = "";
  invalidateSequence();
}

function updateSeparation(value: number): void {
  separation.value = value;
  invalidateSequence();
}

function updateIterations(value: number): void {
  iterations.value = value;
  invalidateSequence();
}

function updatePhaseMode(value: string): void {
  phaseMode.value = value;
  invalidateSequence();
}

function simulationCalibration(): CalibrationPackage {
  const points = [...initialAtoms.value, ...targetSites.value];
  const maximumCoordinate = Math.max(1, ...points.flatMap((point) => [Math.abs(point.xUm), Math.abs(point.yUm)]));
  const scale = 14 / maximumCoordinate;
  return {
    manifest: {
      calibrationId: "browser-simulation",
      wavelengthNm: 1,
      activeWidth: 32,
      activeHeight: 32,
      fftWidth: 32,
      fftHeight: 32,
      coordinateConvention: "+x right, +y up",
    },
    coordinateTransform: { originXUm: 16, originYUm: 16, scaleX: scale, scaleY: scale },
  };
}

function updateCompileLog(progress: CompileProgress): void {
  if (progress.stage === "PLANNING") setLog(1, "Planning collision-free paths", "active");
  if (progress.stage === "SOLVING_SLM_FRAMES") {
    setLog(2, `Solving frame ${progress.frameIndex ?? progress.completed}`, "active");
  }
}

async function compileSequence(): Promise<void> {
  if (running.value || !coordinateEditor.value?.applyDraft({ fitView: false })) return;
  const generation = ++compileGeneration;
  running.value = true;
  compilationState.value = "running";
  inputError.value = "";
  setLog(0, "Coordinates normalized", "done");
  setLog(1, "Planning collision-free paths", "active");
  setLog(2, "WGS context is idle");

  try {
    const compiler = await SlmSequenceCompiler.create({
      simulationMode: true,
      calibration: simulationCalibration(),
      hologram: {
        width: 32,
        height: 32,
        format: "UINT8",
        firstFrameIterations: iterations.value,
        subsequentFrameIterations: iterations.value,
        maxIterations: Math.max(8, iterations.value * 2),
        targetPhaseMode: phaseMode.value === "Soft phase locked" ? "SOFT_PHASE_LOCKED_WGS" : "PHASE_LOCKED_WGS",
        requireConvergence: false,
      },
      planner: {
        minimumSeparationUm: separation.value,
        geometricMarginUm: 0.1,
        gridResolutionUm: Math.max(0.5, separation.value / 2),
        planningTickUs: 100,
        maxSearchTicks: 256,
        maxCbsNodes: 500,
      },
      motion: {
        framePeriodUs: 100,
        preMoveDwellUs: 100,
        postMoveSettleUs: 100,
        maxVelocityUmPerUs: 1,
        maxAccelerationUmPerUs2: 1,
        maxJerkUmPerUs3: 1,
      },
    });
    const compiled = await compiler.compileRearrangement({
      initialAtoms: cloneAtoms(initialAtoms.value),
      targetSites: cloneTargets(targetSites.value),
      calibrationId: "browser-simulation",
    }, { onProgress: updateCompileLog });
    if (generation !== compileGeneration) return;

    sequence.value = compiled;
    frames.value = await Promise.resolve(compiled.trapFrameStore.toArray());
    total.value = frames.value.length;
    frame.value = 0;
    compilationState.value = "accepted";
    setLog(1, "Conflict-free route accepted", "done");
    setLog(2, `${total.value} calibrated frames accepted`, "done");
    running.value = false;
    startPlayback();
  } catch (error) {
    if (generation !== compileGeneration) return;
    sequence.value = null;
    frames.value = [];
    compilationState.value = "rejected";
    setLog(1, "Compilation rejected", "active");
    setLog(2, error instanceof Error ? error.message : "Compiler error");
    inputError.value = error instanceof Error ? error.message : "Compilation failed";
    running.value = false;
  }
}

function startPlayback(): void {
  stopPlayback();
  const started = performance.now();
  const duration = 2200;
  const tick = (now: number): void => {
    if (!sequence.value || frames.value.length === 0) return;
    const progress = Math.min(1, (now - started) / duration);
    frame.value = Math.min(total.value - 1, Math.floor(progress * total.value));
    if (progress < 1) playbackFrame = requestAnimationFrame(tick);
  };
  playbackFrame = requestAnimationFrame(tick);
}

function stepFrame(): void {
  if (total.value <= 0) return;
  stopPlayback();
  frame.value = (frame.value + 1) % total.value;
}

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function bytesFor(framePixels: Uint8Array | Uint16Array): Uint8Array {
  return new Uint8Array(framePixels.buffer, framePixels.byteOffset, framePixels.byteLength);
}

async function exportSlmFrames(): Promise<void> {
  const compiled = sequence.value;
  if (!compiled) return;
  const slmFrames = await Promise.resolve(compiled.slmFrameStore.toArray());
  const first = slmFrames[0];
  if (!first) return;
  const headerSize = 24;
  const output = new Uint8Array(headerSize + slmFrames.reduce((sum, item) => sum + item.byteLength, 0));
  const view = new DataView(output.buffer);
  output.set([83, 76, 77, 70], 0);
  view.setUint16(4, 1, true);
  view.setUint16(6, compiled.manifest.outputWidth, true);
  view.setUint16(8, compiled.manifest.outputHeight, true);
  view.setUint8(10, compiled.manifest.pixelFormat === "UINT16" ? 16 : 8);
  view.setUint32(12, slmFrames.length, true);
  view.setUint32(16, first.byteLength, true);
  view.setUint32(20, compiled.manifest.framePeriodUs, true);
  let offset = headerSize;
  for (const pixels of slmFrames) {
    output.set(bytesFor(pixels), offset);
    offset += pixels.byteLength;
  }
  download(new Blob([output.buffer], { type: "application/octet-stream" }), "slm-frames.slmf");
}

function exportManifest(): void {
  const compiled = sequence.value;
  if (!compiled) return;
  const payload = {
    manifest: compiled.manifest,
    assignment: compiled.assignment,
    trajectories: compiled.trajectories,
    frameDescriptors: compiled.slmFrameDescriptors,
    frameMetrics: compiled.frameMetrics,
    validation: compiled.validation,
  };
  const json = JSON.stringify(payload, (_key, value: unknown) => typeof value === "bigint" ? value.toString() : value, 2);
  download(new Blob([json], { type: "application/json" }), "slm-sequence-manifest.json");
}

function reset(): void {
  compileGeneration += 1;
  running.value = false;
  compilationState.value = "idle";
  initialAtoms.value = cloneAtoms(DEFAULT_INITIAL_ATOMS);
  targetSites.value = cloneTargets(DEFAULT_TARGET_SITES);
  separation.value = 1.2;
  iterations.value = 4;
  phaseMode.value = "Phase locked WGS";
  inputError.value = "";
  sequence.value = null;
  frames.value = [];
  frame.value = 0;
  total.value = 64;
  logs.value = defaultLogs();
  stopPlayback();
  nextTick(() => coordinateEditor.value?.resetEditor());
}

onBeforeUnmount(() => {
  compileGeneration += 1;
  stopPlayback();
});
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="SLM compiler home">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
        <span>SLM COMPILER</span>
      </a>
      <div class="topbar-status"><span class="status-dot"></span> READY / WASM CORE</div>
      <button class="reset-button" type="button" @click="reset">Reset</button>
    </header>

    <section class="workspace">
      <div class="workspace-heading">
        <div>
          <p class="eyebrow">Sequence workspace</p>
          <h1>Compile rearrangement</h1>
        </div>
        <p class="heading-note">Initial atoms to target sites<br>with collision-free phase frames.</p>
      </div>

      <div class="workspace-grid">
        <ArrayPreview
          :initial-atoms="initialAtoms"
          :target-sites="targetSites"
          :frames="frames"
          :frame="frame"
          :total="total"
          :separation="separation"
          :canvas-state="canvasState"
        />
        <CompilerControls
          :separation="separation"
          :iterations="iterations"
          :phase-mode="phaseMode"
          :badge="badge"
          :running="running"
          :can-export="Boolean(sequence)"
          :logs="logs"
          @update:separation="updateSeparation"
          @update:iterations="updateIterations"
          @update:phase-mode="updatePhaseMode"
          @compile="compileSequence"
          @step="stepFrame"
          @export-frames="exportSlmFrames"
          @export-manifest="exportManifest"
        />
      </div>

      <CoordinateWorkspace
        ref="coordinateEditor"
        :initial-atoms="initialAtoms"
        :target-sites="targetSites"
        :disabled="running"
        :error="inputError"
        @change="updateCoordinates"
        @error="inputError = $event"
      />

      <div class="metrics-row" aria-label="Sequence metrics">
        <div><small>TRAP FRAMES</small><strong>{{ sequence ? total : "--" }}</strong></div>
        <div><small>MIN CLEARANCE / UM</small><strong>{{ metricClearance }}</strong></div>
        <div><small>DIFFRACTION EFFICIENCY</small><strong>{{ averageEfficiency === null ? "--" : `${(averageEfficiency * 100).toFixed(1)}%` }}</strong></div>
        <div><small>ASSIGNMENT COST</small><strong>{{ metricCost }}</strong></div>
      </div>
    </section>
  </main>
</template>
