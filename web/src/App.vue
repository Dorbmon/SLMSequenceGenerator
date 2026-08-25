<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef } from "vue";
import {
  MemoryFrameStore,
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
import OpticalTweezersPage from "./pages/OpticalTweezersPage.vue";
import {
  DEFAULT_SLM_HEIGHT,
  DEFAULT_SLM_WIDTH,
  fftDimensionFor,
  normalizeSlmDimension,
} from "./lib/resolution.js";
import type {
  CompilerWorkerRequest,
  CompilerWorkerResponse,
  SerializedSequence,
} from "./workers/compiler-messages.js";

interface CoordinateEditorHandle {
  applyDraft(options?: { fitView?: boolean }): boolean;
  resetEditor(): void;
}

interface OpticalTweezersPageHandle {
  reset(): void;
}

interface LogLine {
  text: string;
  state?: "active" | "done";
}

type CompilationState = "idle" | "running" | "accepted" | "rejected";

const coordinateEditor = ref<CoordinateEditorHandle | null>(null);
const opticalTweezersPage = ref<OpticalTweezersPageHandle | null>(null);
const activePath = ref(pagePath());
const isTweezerPage = computed(() => activePath.value === "/tweezers");
const initialAtoms = ref<InitialAtom[]>(cloneAtoms(DEFAULT_INITIAL_ATOMS));
const targetSites = ref<TargetSite[]>(cloneTargets(DEFAULT_TARGET_SITES));
const sequence = shallowRef<CompiledSequenceHandle | null>(null);
const frames = shallowRef<TrapFrame[]>([]);
const frame = ref(0);
const total = ref(64);
const separation = ref(1.2);
const iterations = ref(4);
const slmWidth = ref(DEFAULT_SLM_WIDTH);
const slmHeight = ref(DEFAULT_SLM_HEIGHT);
const phaseMode = ref("Phase locked WGS");
const running = ref(false);
const compileElapsedMs = ref(0);
const compileProgress = ref<number | null>(null);
const compileProgressLabel = ref("STARTING WORKER");
const compilationState = ref<CompilationState>("idle");
const inputError = ref("");
const logs = ref<LogLine[]>(defaultLogs());
let playbackFrame = 0;
let compileGeneration = 0;
let compilationWorker: Worker | null = null;
let compileTimer = 0;
let compileStarted = 0;

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
const fftWidth = computed(() => fftDimensionFor(slmWidth.value));
const fftHeight = computed(() => fftDimensionFor(slmHeight.value));

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

function invalidateSequence(message = "Compiler settings changed"): void {
  stopPlayback();
  sequence.value = null;
  frames.value = [];
  frame.value = 0;
  total.value = 64;
  compilationState.value = "idle";
  compileElapsedMs.value = 0;
  compileProgress.value = null;
  compileProgressLabel.value = "STARTING WORKER";
  logs.value = [
    { text: message },
    { text: "Compile to validate paths" },
    { text: "Hologram state is idle" },
  ];
}

function updateCoordinates(nextAtoms: InitialAtom[], nextTargets: TargetSite[]): void {
  initialAtoms.value = cloneAtoms(nextAtoms);
  targetSites.value = cloneTargets(nextTargets);
  inputError.value = "";
  invalidateSequence("Coordinates changed");
}

function updateSeparation(value: number): void {
  separation.value = value;
  invalidateSequence();
}

function updateIterations(value: number): void {
  iterations.value = value;
  invalidateSequence();
}

function updateSlmWidth(value: number): void {
  try {
    slmWidth.value = normalizeSlmDimension(value, "SLM width");
    inputError.value = "";
    invalidateSequence("SLM resolution changed");
  } catch (error) {
    inputError.value = error instanceof Error ? error.message : "Invalid SLM width";
  }
}

function updateSlmHeight(value: number): void {
  try {
    slmHeight.value = normalizeSlmDimension(value, "SLM height");
    inputError.value = "";
    invalidateSequence("SLM resolution changed");
  } catch (error) {
    inputError.value = error instanceof Error ? error.message : "Invalid SLM height";
  }
}

function updatePhaseMode(value: string): void {
  phaseMode.value = value;
  invalidateSequence();
}

function updateCompileLog(progress: CompileProgress): void {
  const labels: Record<CompileProgress["stage"], string> = {
    VALIDATING: "VALIDATING INPUT",
    ASSIGNING: "ASSIGNING ATOMS",
    PLANNING: "PLANNING TRAJECTORIES",
    PARAMETERIZING: "PARAMETERIZING MOTION",
    GENERATING_TRAP_FRAMES: "SAMPLING TRAP FRAMES",
    SOLVING_SLM_FRAMES: "SOLVING SLM FRAMES",
    VALIDATING_SEQUENCE: "VALIDATING SEQUENCE",
    WRITING_OUTPUT: "PACKAGING OUTPUT",
  };
  compileProgressLabel.value = labels[progress.stage];
  compileProgress.value = progress.total > 0 ? progress.completed / progress.total : null;
  if (progress.stage === "PLANNING") setLog(1, "Planning collision-free paths", "active");
  if (progress.stage === "SOLVING_SLM_FRAMES") {
    total.value = Math.max(1, progress.total);
    frame.value = Math.min(total.value - 1, progress.completed);
    setLog(2, `Solving frame ${progress.completed} / ${progress.total}`, "active");
  }
}

function compileSequence(): void {
  if (running.value || !coordinateEditor.value?.applyDraft({ fitView: false })) return;
  terminateCompilationWorker();
  const jobId = ++compileGeneration;
  running.value = true;
  compilationState.value = "running";
  inputError.value = "";
  compileProgress.value = null;
  compileProgressLabel.value = "STARTING WORKER";
  setLog(0, "Coordinates normalized", "done");
  setLog(1, "Planning collision-free paths", "active");
  setLog(2, "WGS worker is starting", "active");
  startCompileClock();

  const worker = new Worker(new URL("./workers/compiler.worker.ts", import.meta.url), { type: "module" });
  compilationWorker = worker;
  worker.onmessage = (event: MessageEvent<CompilerWorkerResponse>) => {
    const response = event.data;
    if (response.jobId !== jobId || jobId !== compileGeneration) return;
    if (response.kind === "SEQUENCE_PROGRESS") {
      updateCompileLog(response.progress);
      return;
    }
    if (response.kind === "WORKER_ERROR") {
      rejectCompilation(response.message, jobId, worker);
      return;
    }
    if (response.kind !== "SEQUENCE_RESULT") return;
    const compiled = deserializeSequence(response.sequence);
    sequence.value = compiled;
    frames.value = response.sequence.trapFrames;
    total.value = frames.value.length;
    frame.value = 0;
    compilationState.value = "accepted";
    running.value = false;
    compileProgress.value = 1;
    compileProgressLabel.value = "COMPLETE";
    stopCompileClock(response.elapsedMs);
    setLog(1, "Conflict-free route accepted", "done");
    setLog(2, `${total.value} calibrated frames accepted`, "done");
    disposeCompilationWorker(worker);
    startPlayback();
  };
  worker.onerror = (event: ErrorEvent) => {
    if (jobId !== compileGeneration) return;
    rejectCompilation(event.message || "Compiler worker failed", jobId, worker);
  };
  const request: CompilerWorkerRequest = {
    kind: "COMPILE_SEQUENCE",
    jobId,
    input: {
      initialAtoms: cloneAtoms(initialAtoms.value),
      targetSites: cloneTargets(targetSites.value),
      separationUm: separation.value,
      iterations: iterations.value,
      slmWidth: slmWidth.value,
      slmHeight: slmHeight.value,
      fftWidth: fftWidth.value,
      fftHeight: fftHeight.value,
      targetPhaseMode: phaseMode.value === "Soft phase locked" ? "SOFT_PHASE_LOCKED_WGS" : "PHASE_LOCKED_WGS",
    },
  };
  worker.postMessage(request);
}

function deserializeSequence(serialized: SerializedSequence): CompiledSequenceHandle {
  const trapFrameStore = new MemoryFrameStore<TrapFrame>();
  const slmFrameStore = new MemoryFrameStore<Uint8Array | Uint16Array>();
  for (const trapFrame of serialized.trapFrames) trapFrameStore.append(trapFrame);
  for (const frame of serialized.slmFrames) {
    slmFrameStore.append(frame.format === "UINT16" ? new Uint16Array(frame.buffer) : new Uint8Array(frame.buffer));
  }
  return {
    manifest: serialized.manifest,
    assignment: serialized.assignment,
    trajectories: serialized.trajectories,
    trapFrameStore,
    slmFrameStore,
    slmFrameDescriptors: serialized.slmFrameDescriptors,
    frameMetrics: serialized.frameMetrics,
    validation: serialized.validation,
  };
}

function rejectCompilation(message: string, jobId: number, worker: Worker): void {
  if (jobId !== compileGeneration) return;
  sequence.value = null;
  frames.value = [];
  compilationState.value = "rejected";
  running.value = false;
  stopCompileClock();
  setLog(1, "Compilation rejected", "active");
  setLog(2, message);
  inputError.value = message;
  disposeCompilationWorker(worker);
}

function cancelCompilation(): void {
  if (!running.value) return;
  compileGeneration += 1;
  terminateCompilationWorker();
  stopCompileClock();
  running.value = false;
  compilationState.value = "idle";
  frame.value = 0;
  total.value = frames.value.length || 64;
  compileProgress.value = null;
  compileProgressLabel.value = "CANCELLED";
  logs.value = [
    { text: "Compilation cancelled", state: "done" },
    { text: "Worker terminated safely", state: "done" },
    { text: "Ready for another run" },
  ];
}

function startCompileClock(): void {
  stopCompileClock();
  compileStarted = performance.now();
  compileElapsedMs.value = 0;
  compileTimer = window.setInterval(() => {
    compileElapsedMs.value = performance.now() - compileStarted;
  }, 100);
}

function stopCompileClock(finalElapsedMs?: number): void {
  window.clearInterval(compileTimer);
  compileTimer = 0;
  if (finalElapsedMs !== undefined) compileElapsedMs.value = finalElapsedMs;
}

function disposeCompilationWorker(worker: Worker): void {
  worker.terminate();
  if (compilationWorker === worker) compilationWorker = null;
}

function terminateCompilationWorker(): void {
  compilationWorker?.terminate();
  compilationWorker = null;
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
  terminateCompilationWorker();
  stopCompileClock();
  running.value = false;
  compilationState.value = "idle";
  initialAtoms.value = cloneAtoms(DEFAULT_INITIAL_ATOMS);
  targetSites.value = cloneTargets(DEFAULT_TARGET_SITES);
  separation.value = 1.2;
  iterations.value = 4;
  slmWidth.value = DEFAULT_SLM_WIDTH;
  slmHeight.value = DEFAULT_SLM_HEIGHT;
  phaseMode.value = "Phase locked WGS";
  inputError.value = "";
  sequence.value = null;
  frames.value = [];
  frame.value = 0;
  total.value = 64;
  compileElapsedMs.value = 0;
  compileProgress.value = null;
  compileProgressLabel.value = "STARTING WORKER";
  logs.value = defaultLogs();
  stopPlayback();
  nextTick(() => coordinateEditor.value?.resetEditor());
}

function resetActivePage(): void {
  if (isTweezerPage.value) opticalTweezersPage.value?.reset();
  else reset();
}

function pagePath(): "/" | "/tweezers" {
  return window.location.pathname === "/tweezers" || window.location.pathname.startsWith("/tweezers/") ? "/tweezers" : "/";
}

function navigate(event: MouseEvent, path: "/" | "/tweezers"): void {
  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  if (activePath.value !== path) window.history.pushState(null, "", path);
  activePath.value = path;
  document.title = path === "/tweezers" ? "Optical Tweezer Frame | SLM Compiler" : "SLM Sequence Compiler";
  window.scrollTo({ top: 0, behavior: "instant" });
}

function handlePopState(): void {
  activePath.value = pagePath();
}

onMounted(() => {
  window.addEventListener("popstate", handlePopState);
  document.title = isTweezerPage.value ? "Optical Tweezer Frame | SLM Compiler" : "SLM Sequence Compiler";
});

onBeforeUnmount(() => {
  compileGeneration += 1;
  terminateCompilationWorker();
  stopCompileClock();
  stopPlayback();
  window.removeEventListener("popstate", handlePopState);
});
</script>

<template>
  <main class="app-shell">
    <header class="topbar">
      <a class="brand" href="/" aria-label="SLM compiler home" @click="navigate($event, '/')">
        <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
        <span>SLM COMPILER</span>
      </a>
      <nav class="page-nav" aria-label="Compiler workspaces">
        <a href="/" :class="{ 'is-active': !isTweezerPage }" :aria-current="!isTweezerPage ? 'page' : undefined" @click="navigate($event, '/')">Sequence</a>
        <a href="/tweezers" :class="{ 'is-active': isTweezerPage }" :aria-current="isTweezerPage ? 'page' : undefined" @click="navigate($event, '/tweezers')">Tweezer frame</a>
      </nav>
      <div class="topbar-status"><span class="status-dot"></span> READY / WASM CORE</div>
      <button class="reset-button" type="button" @click="resetActivePage">Reset</button>
    </header>

    <section v-show="!isTweezerPage" class="workspace">
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
          :running="running"
        />
        <CompilerControls
          :separation="separation"
          :iterations="iterations"
          :slm-width="slmWidth"
          :slm-height="slmHeight"
          :fft-width="fftWidth"
          :fft-height="fftHeight"
          :phase-mode="phaseMode"
          :badge="badge"
          :running="running"
          :elapsed-ms="compileElapsedMs"
          :progress="compileProgress"
          :progress-label="compileProgressLabel"
          :can-export="Boolean(sequence)"
          :logs="logs"
          @update:separation="updateSeparation"
          @update:iterations="updateIterations"
          @update:slm-width="updateSlmWidth"
          @update:slm-height="updateSlmHeight"
          @update:phase-mode="updatePhaseMode"
          @compile="compileSequence"
          @cancel="cancelCompilation"
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
    <OpticalTweezersPage v-show="isTweezerPage" ref="opticalTweezersPage" />
  </main>
</template>
