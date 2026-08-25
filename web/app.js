import { SlmSequenceCompiler } from "./compiler.js?v=2";

const $ = (selector) => document.querySelector(selector);
const canvas = $("#arrayCanvas");
const context = canvas.getContext("2d");
const compileButton = $("#compileButton");
const stepButton = $("#stepButton");
const exportButton = $("#exportButton");
const manifestButton = $("#manifestButton");
const resetButton = $("#resetButton");
const separationRange = $("#separationRange");
const iterationRange = $("#iterationRange");

const defaultInitialAtoms = [
  { atomId: 1, xUm: -6, yUm: -3 }, { atomId: 2, xUm: -3, yUm: -3 },
  { atomId: 3, xUm: 0, yUm: -3 }, { atomId: 4, xUm: 3, yUm: -3 },
  { atomId: 5, xUm: 6, yUm: -3 }, { atomId: 6, xUm: -6, yUm: 1 },
  { atomId: 7, xUm: -3, yUm: 1 }, { atomId: 8, xUm: 0, yUm: 1 },
  { atomId: 9, xUm: 3, yUm: 1 }, { atomId: 10, xUm: 6, yUm: 1 },
  { atomId: 11, xUm: -3, yUm: 5 }, { atomId: 12, xUm: 3, yUm: 5 },
];
const defaultTargetSites = [
  { siteId: 101, xUm: -4.5, yUm: -1.5 }, { siteId: 102, xUm: -1.5, yUm: -1.5 },
  { siteId: 103, xUm: 1.5, yUm: -1.5 }, { siteId: 104, xUm: 4.5, yUm: -1.5 },
  { siteId: 105, xUm: -4.5, yUm: 1.5 }, { siteId: 106, xUm: -1.5, yUm: 1.5 },
  { siteId: 107, xUm: 1.5, yUm: 1.5 }, { siteId: 108, xUm: 4.5, yUm: 1.5 },
  { siteId: 109, xUm: 0, yUm: 4.5 },
];

const state = {
  initialAtoms: clone(defaultInitialAtoms),
  targetSites: clone(defaultTargetSites),
  sequence: null,
  frames: [],
  frame: 0,
  total: 64,
  running: false,
  playbackStart: 0,
  playbackDuration: 2200,
  separation: Number(separationRange.value),
  iterations: Number(iterationRange.value),
};

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function setInputError(message = "") {
  const error = $("#inputError");
  error.textContent = message;
  error.classList.toggle("is-visible", Boolean(message));
}

function updateEditors() {
  $("#initialEditor").value = JSON.stringify(state.initialAtoms, null, 2);
  $("#targetEditor").value = JSON.stringify(state.targetSites, null, 2);
}

function parsePointList(raw, kind) {
  const value = JSON.parse(raw);
  const list = Array.isArray(value)
    ? value
    : value && Array.isArray(value[kind === "atom" ? "initialAtoms" : "targetSites"])
      ? value[kind === "atom" ? "initialAtoms" : "targetSites"]
      : value && Array.isArray(value.points) ? value.points : null;
  if (!list) throw new Error(`${kind === "atom" ? "Initial atoms" : "Target sites"} must be a JSON array`);
  return list.map((entry, index) => {
    const xUm = Array.isArray(entry) ? entry[0] : entry?.xUm ?? entry?.x;
    const yUm = Array.isArray(entry) ? entry[1] : entry?.yUm ?? entry?.y;
    if (!Number.isFinite(xUm) || !Number.isFinite(yUm)) throw new Error(`Point ${index + 1} has invalid coordinates`);
    const point = Array.isArray(entry) ? {} : { ...entry };
    point.xUm = xUm;
    point.yUm = yUm;
    if (kind === "atom") point.atomId = point.atomId ?? index + 1;
    else point.siteId = point.siteId ?? index + 1;
    return point;
  });
}

function readEditors() {
  return {
    initialAtoms: parsePointList($("#initialEditor").value, "atom"),
    targetSites: parsePointList($("#targetEditor").value, "target"),
  };
}

function invalidateSequence() {
  state.sequence = null;
  state.frames = [];
  state.frame = 0;
  state.total = 64;
  $("#frameCounter").textContent = "FRAME 00 / 64";
  $("#exportButton").disabled = true;
  $("#manifestButton").disabled = true;
  $("#validBadge").textContent = "VALIDATED";
  setLog($("#logOne"), "Coordinates changed");
  setLog($("#logTwo"), "Compile to validate paths");
  setLog($("#logThree"), "Hologram state is idle");
  updateMetrics();
  draw();
}

function applyCoordinates() {
  try {
    const parsed = readEditors();
    state.initialAtoms = parsed.initialAtoms;
    state.targetSites = parsed.targetSites;
    setInputError();
    updateCounts();
    invalidateSequence();
    return true;
  } catch (error) {
    setInputError(error instanceof Error ? error.message : "Invalid coordinate input");
    $("#validBadge").textContent = "CHECK INPUT";
    return false;
  }
}

function updateCounts() {
  $("#sourceCount").textContent = String(state.initialAtoms.length).padStart(2, "0");
  $("#targetCount").textContent = String(state.targetSites.length).padStart(2, "0");
}

function setLog(element, text, mode = "") {
  element.textContent = text;
  const row = element.closest(".log-row");
  row?.classList.toggle("is-active", mode === "active");
  row?.classList.toggle("is-done", mode === "done");
}

function updateMetrics() {
  $("#separationValue").textContent = `${state.separation.toFixed(1)} um`;
  $("#iterationValue").textContent = `${String(state.iterations).padStart(2, "0")} / FRAME`;
  $("#metricClearance").textContent = state.separation.toFixed(1);
  if (!state.sequence) {
    $("#metricFrames").textContent = "--";
    $("#metricEfficiency").textContent = "--";
    $("#metricCost").textContent = "--";
    return;
  }
  const metrics = state.sequence.frameMetrics;
  const averageEfficiency = metrics.length ? metrics.reduce((sum, metric) => sum + metric.diffractionEfficiency, 0) / metrics.length : 0;
  $("#metricFrames").textContent = String(state.total);
  $("#metricClearance").textContent = Number.isFinite(state.sequence.validation.minimumAtomSeparationUm)
    ? state.sequence.validation.minimumAtomSeparationUm.toFixed(1)
    : "--";
  $("#metricEfficiency").textContent = `${(averageEfficiency * 100).toFixed(1)}%`;
  $("#metricCost").textContent = state.sequence.manifest.assignmentCost.toFixed(1);
}

function setRunning(running) {
  state.running = running;
  compileButton.disabled = running;
  compileButton.innerHTML = running ? '<span></span> Compiling...' : '<span></span> Compile sequence';
  $("#validBadge").textContent = running ? "PROCESSING" : state.sequence ? "ACCEPTED" : "VALIDATED";
  $("#canvasState").textContent = running ? "SOLVING / SEQUENTIAL" : state.sequence ? "SEQUENCE ACCEPTED" : "READY";
}

async function compileSequence() {
  if (state.running || !applyCoordinates()) return;
  setRunning(true);
  setLog($("#logOne"), "Coordinates normalized", "done");
  setLog($("#logTwo"), "Planning collision-free paths", "active");
  setLog($("#logThree"), "WGS context is idle");
  try {
    const compiler = await SlmSequenceCompiler.create({
      simulationMode: true,
      calibration: simulationCalibration(),
      hologram: {
        width: 32,
        height: 32,
        format: "UINT8",
        firstFrameIterations: state.iterations,
        subsequentFrameIterations: state.iterations,
        maxIterations: Math.max(8, state.iterations * 2),
        targetPhaseMode: $("#phaseMode").value === "Soft phase locked" ? "SOFT_PHASE_LOCKED_WGS" : "PHASE_LOCKED_WGS",
        requireConvergence: false,
      },
      planner: {
        minimumSeparationUm: state.separation,
        geometricMarginUm: 0.1,
        gridResolutionUm: Math.max(0.5, state.separation / 2),
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
    const sequence = await compiler.compileRearrangement({
      initialAtoms: state.initialAtoms,
      targetSites: state.targetSites,
      calibrationId: "browser-simulation",
    }, {
      onProgress: (progress) => updateCompileLog(progress),
    });
    state.sequence = sequence;
    state.frames = await Promise.resolve(sequence.trapFrameStore.toArray());
    state.total = state.frames.length;
    state.frame = 0;
    $("#exportButton").disabled = false;
    $("#manifestButton").disabled = false;
    $("#frameCounter").textContent = `FRAME 00 / ${String(state.total).padStart(2, "0")}`;
    setLog($("#logTwo"), "Conflict-free route accepted", "done");
    setLog($("#logThree"), `${state.total} calibrated frames accepted`, "done");
    updateMetrics();
    setRunning(false);
    state.playbackStart = performance.now();
    requestAnimationFrame(playback);
  } catch (error) {
    state.sequence = null;
    state.frames = [];
    $("#exportButton").disabled = true;
    $("#manifestButton").disabled = true;
    $("#validBadge").textContent = "REJECTED";
    setLog($("#logTwo"), "Compilation rejected", "active");
    setLog($("#logThree"), error instanceof Error ? error.message : "Compiler error");
    setInputError(error instanceof Error ? error.message : "Compilation failed");
    setRunning(false);
  }
}

function simulationCalibration() {
  const points = [...state.initialAtoms, ...state.targetSites];
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

function updateCompileLog(progress) {
  if (progress.stage === "PLANNING") setLog($("#logTwo"), "Planning collision-free paths", "active");
  if (progress.stage === "SOLVING_SLM_FRAMES") setLog($("#logThree"), `Solving frame ${progress.frameIndex ?? 0}`, "active");
}

function playback(now) {
  if (!state.sequence || state.frames.length === 0) return;
  const progress = Math.min(1, (now - state.playbackStart) / state.playbackDuration);
  state.frame = Math.min(state.total - 1, Math.floor(progress * state.total));
  $("#frameCounter").textContent = `FRAME ${String(state.frame).padStart(2, "0")} / ${String(state.total).padStart(2, "0")}`;
  draw();
  if (progress < 1) requestAnimationFrame(playback);
}

function pointOf(point) {
  return { xUm: point.xUm, yUm: point.yUm };
}

function project(point, width, height) {
  const scale = Math.min(width, height) / 15;
  return { x: width / 2 + point.xUm * scale, y: height / 2 - point.yUm * scale };
}

function ease(value) {
  const t = Math.max(0, Math.min(1, value));
  return t * t * t * (10 + t * (-15 + 6 * t));
}

function drawGrid(width, height) {
  context.save();
  context.strokeStyle = "rgba(129,216,208,.07)";
  context.lineWidth = 1;
  const spacing = Math.min(width, height) / 15;
  for (let x = width / 2 % spacing; x < width; x += spacing) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = height / 2 % spacing; y < height; y += spacing) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  context.restore();
}

function drawTarget(point, width, height, index) {
  const target = project(pointOf(point), width, height);
  const pulse = 1 + Math.sin(performance.now() / 700 + index) * .12;
  context.save();
  context.strokeStyle = "rgba(185,243,107,.7)";
  context.lineWidth = 1;
  context.beginPath(); context.arc(target.x, target.y, 5 * pulse, 0, Math.PI * 2); context.stroke();
  context.fillStyle = "rgba(185,243,107,.12)";
  context.beginPath(); context.arc(target.x, target.y, 11, 0, Math.PI * 2); context.fill();
  context.restore();
}

function currentAtoms() {
  if (state.frames.length > 0) {
    const frame = state.frames[state.frame] ?? state.frames[0];
    return frame.traps.filter((trap) => trap.atomId !== null && trap.intensity > 0).map(pointOf);
  }
  const progress = state.frame / Math.max(1, state.total - 1);
  return state.initialAtoms.map((atom, index) => {
    const target = state.targetSites[index % Math.max(1, state.targetSites.length)] ?? atom;
    const local = ease(Math.max(0, Math.min(1, progress * 1.08 - index * .018)));
    return { xUm: atom.xUm + (target.xUm - atom.xUm) * local, yUm: atom.yUm + (target.yUm - atom.yUm) * local };
  });
}

function drawAtom(point, width, height, index) {
  const atom = project(point, width, height);
  const glow = context.createRadialGradient(atom.x, atom.y, 0, atom.x, atom.y, 18);
  glow.addColorStop(0, "rgba(129,216,208,.34)"); glow.addColorStop(1, "rgba(129,216,208,0)");
  context.fillStyle = glow; context.beginPath(); context.arc(atom.x, atom.y, 18, 0, Math.PI * 2); context.fill();
  context.fillStyle = index >= state.targetSites.length ? "#81d8d0" : "#d8eee3";
  context.beginPath(); context.arc(atom.x, atom.y, 3.2, 0, Math.PI * 2); context.fill();
}

function draw() {
  const bounds = canvas.getBoundingClientRect();
  const width = bounds.width || 760;
  const height = bounds.height || 540;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0b1612"; context.fillRect(0, 0, width, height);
  drawGrid(width, height);
  state.targetSites.forEach((target, index) => drawTarget(target, width, height, index));
  currentAtoms().forEach((atom, index) => drawAtom(atom, width, height, index));
  context.save();
  context.fillStyle = "rgba(185,243,107,.8)"; context.font = "9px 'DM Mono', monospace"; context.fillText("TARGET FIELD", 18, 25);
  context.fillStyle = "rgba(141,157,149,.75)"; context.fillText(`${state.separation.toFixed(1)} UM CLEARANCE`, 18, height - 18);
  context.restore();
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function bytesFor(frame) {
  return new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
}

async function exportSlmFrames() {
  if (!state.sequence) return;
  const frames = await Promise.resolve(state.sequence.slmFrameStore.toArray());
  const first = frames[0];
  if (!first) return;
  const headerSize = 24;
  const output = new Uint8Array(headerSize + frames.reduce((sum, frame) => sum + frame.byteLength, 0));
  const view = new DataView(output.buffer);
  output.set([83, 76, 77, 70], 0); // SLMF
  view.setUint16(4, 1, true);
  view.setUint16(6, state.sequence.manifest.outputWidth, true);
  view.setUint16(8, state.sequence.manifest.outputHeight, true);
  view.setUint8(10, state.sequence.manifest.pixelFormat === "UINT16" ? 16 : 8);
  view.setUint32(12, frames.length, true);
  view.setUint32(16, first.byteLength, true);
  view.setUint32(20, state.sequence.manifest.framePeriodUs, true);
  let offset = headerSize;
  for (const frame of frames) { output.set(bytesFor(frame), offset); offset += frame.byteLength; }
  download(new Blob([output], { type: "application/octet-stream" }), "slm-frames.slmf");
}

function exportManifest() {
  if (!state.sequence) return;
  const payload = {
    manifest: state.sequence.manifest,
    assignment: state.sequence.assignment,
    trajectories: state.sequence.trajectories,
    frameDescriptors: state.sequence.slmFrameDescriptors,
    frameMetrics: state.sequence.frameMetrics,
    validation: state.sequence.validation,
  };
  download(new Blob([JSON.stringify(payload, (_, value) => typeof value === "bigint" ? value.toString() : value, 2)], { type: "application/json" }), "slm-sequence-manifest.json");
}

async function importJson(file, target) {
  const parsed = JSON.parse(await file.text());
  if (target === "request" && parsed && parsed.initialAtoms && parsed.targetSites) {
    $("#initialEditor").value = JSON.stringify(parsed.initialAtoms, null, 2);
    $("#targetEditor").value = JSON.stringify(parsed.targetSites, null, 2);
  } else {
    const list = Array.isArray(parsed) ? parsed : parsed?.points ?? parsed?.[target === "initial" ? "initialAtoms" : "targetSites"];
    if (!Array.isArray(list)) throw new Error("Expected a point array or a full request JSON object");
    $(target === "initial" ? "#initialEditor" : "#targetEditor").value = JSON.stringify(list, null, 2);
  }
  applyCoordinates();
}

function reset() {
  state.initialAtoms = clone(defaultInitialAtoms);
  state.targetSites = clone(defaultTargetSites);
  state.separation = Number(separationRange.value);
  state.iterations = Number(iterationRange.value);
  updateEditors(); updateCounts(); setInputError(); invalidateSequence();
}

$("#applyButton").addEventListener("click", applyCoordinates);
$("#initialUploadButton").addEventListener("click", () => $("#initialUpload").click());
$("#targetUploadButton").addEventListener("click", () => $("#targetUpload").click());
$("#requestUploadButton").addEventListener("click", () => $("#requestUpload").click());
$("#initialUpload").addEventListener("change", (event) => event.target.files[0] && importJson(event.target.files[0], "initial").catch((error) => setInputError(error.message)));
$("#targetUpload").addEventListener("change", (event) => event.target.files[0] && importJson(event.target.files[0], "target").catch((error) => setInputError(error.message)));
$("#requestUpload").addEventListener("change", (event) => event.target.files[0] && importJson(event.target.files[0], "request").catch((error) => setInputError(error.message)));
compileButton.addEventListener("click", compileSequence);
stepButton.addEventListener("click", () => {
  if (state.total <= 0) return;
  state.frame = (state.frame + 1) % state.total;
  $("#frameCounter").textContent = `FRAME ${String(state.frame).padStart(2, "0")} / ${String(state.total).padStart(2, "0")}`;
  draw();
});
exportButton.addEventListener("click", exportSlmFrames);
manifestButton.addEventListener("click", exportManifest);
resetButton.addEventListener("click", reset);
separationRange.addEventListener("input", () => { state.separation = Number(separationRange.value); invalidateSequence(); });
iterationRange.addEventListener("input", () => { state.iterations = Number(iterationRange.value); invalidateSequence(); });
window.addEventListener("resize", resizeCanvas);

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const bounds = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(bounds.width * ratio));
  canvas.height = Math.max(1, Math.round(bounds.height * ratio));
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

updateEditors();
updateCounts();
updateMetrics();
requestAnimationFrame(resizeCanvas);
