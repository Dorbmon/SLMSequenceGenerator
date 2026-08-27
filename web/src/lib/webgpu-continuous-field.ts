import {
  buildContinuousTargetGrid,
  continuousTargetScale,
  createContinuousInversePhaseLut,
  finalizeContinuousFieldResult,
  validateContinuousFieldInput,
  type ContinuousFieldInput,
  type ContinuousFieldResult,
} from "./continuous-field.js";
import { createForwardPhaseTable } from "./forward-simulation.js";
import { createWebGpuFft } from "./webgpu-fft.js";

const WORKGROUP_SIZE = 256;
const INVERSE_LUT_SIZE = 4096;

/** GPU-resident MRAF: target upload once, iterations stay in GPU buffers. */
export async function solveContinuousFieldWebGpu(input: ContinuousFieldInput): Promise<ContinuousFieldResult> {
  validateContinuousFieldInput(input);
  if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("WebGPU is unavailable in this browser worker");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No compatible WebGPU adapter was found");
  const device = await adapter.requestDevice();
  const started = performance.now();
  const target = buildContinuousTargetGrid(input);
  const pixelCount = input.fftWidth * input.fftHeight;
  const scalarBytes = pixelCount * 4;
  const complexBytes = pixelCount * 8;
  const activePixelCount = input.slmWidth * input.slmHeight;
  const packedBytes = Math.ceil(activePixelCount / 4) * 4;
  const largestBuffer = Math.max(complexBytes, scalarBytes);
  if (largestBuffer > device.limits.maxStorageBufferBindingSize || largestBuffer > device.limits.maxBufferSize) {
    device.destroy();
    throw new Error("The continuous-field FFT grid exceeds this GPU's storage-buffer limit");
  }
  if (Math.ceil(pixelCount / WORKGROUP_SIZE) > device.limits.maxComputeWorkgroupsPerDimension) {
    device.destroy();
    throw new Error("The continuous-field FFT grid exceeds this GPU's dispatch limit");
  }

  const storage = GPUBufferUsage.STORAGE;
  const copyDestination = GPUBufferUsage.COPY_DST;
  const buffers: GPUBuffer[] = [];
  let readback: GPUBuffer | undefined;
  let fftParameters: GPUBuffer | undefined;
  try {
    const parameters = serializeParameters(input, target.targetPower, target.region.width, target.region.height);
    const inverseLut = createContinuousInversePhaseLut(input.opticalCalibration.phaseResponseLut, INVERSE_LUT_SIZE);
    const phaseTable = createForwardPhaseTable(input.opticalCalibration.phaseResponseLut);
    const parameterBuffer = makeBuffer(device, "Continuous MRAF parameters", parameters.byteLength, GPUBufferUsage.UNIFORM | copyDestination);
    const targetBuffer = makeBuffer(device, "Continuous target amplitude", scalarBytes, storage | copyDestination);
    const maskBuffer = makeBuffer(device, "Continuous signal mask", scalarBytes, storage | copyDestination);
    const phaseBuffer = makeBuffer(device, "Continuous SLM phase", scalarBytes, storage);
    const fieldBuffer = makeBuffer(device, "Continuous complex field", complexBytes, storage);
    const inverseLutBuffer = makeBuffer(device, "Continuous inverse phase LUT", inverseLut.byteLength, storage | copyDestination);
    const phaseTableBuffer = makeBuffer(device, "Continuous phase response", phaseTable.byteLength, storage | copyDestination);
    const codeBuffer = makeBuffer(device, "Continuous phase codes", scalarBytes, storage);
    const intensityBuffer = makeBuffer(device, "Continuous shifted intensity", scalarBytes, storage | GPUBufferUsage.COPY_SRC);
    const packedBuffer = makeBuffer(device, "Continuous packed SLM frame", packedBytes, storage | GPUBufferUsage.COPY_SRC);
    buffers.push(
      parameterBuffer,
      targetBuffer,
      maskBuffer,
      phaseBuffer,
      fieldBuffer,
      inverseLutBuffer,
      phaseTableBuffer,
      codeBuffer,
      intensityBuffer,
      packedBuffer,
    );
    device.queue.writeBuffer(parameterBuffer, 0, parameters);
    device.queue.writeBuffer(targetBuffer, 0, target.amplitude.buffer as ArrayBuffer);
    device.queue.writeBuffer(maskBuffer, 0, target.signalMask.buffer as ArrayBuffer);
    device.queue.writeBuffer(inverseLutBuffer, 0, inverseLut.buffer as ArrayBuffer);
    device.queue.writeBuffer(phaseTableBuffer, 0, phaseTable.buffer as ArrayBuffer);

    const module = device.createShaderModule({ label: "Continuous MRAF", code: CONTINUOUS_FIELD_SHADER });
    const pipelineNames = [
      "initialize_target_field",
      "extract_phase",
      "make_slm_field",
      "apply_mraf",
      "quantize_codes",
      "make_quantized_field",
      "shift_intensity",
      "pack_active",
    ] as const;
    type PipelineName = typeof pipelineNames[number];
    const pipelines = new Map<PipelineName, GPUComputePipeline>();
    for (const name of pipelineNames) {
      pipelines.set(name, await device.createComputePipelineAsync({
        label: `Continuous ${name}`,
        layout: "auto",
        compute: { module, entryPoint: name },
      }));
    }
    const resourceByBinding: Record<number, GPUBuffer> = {
      0: parameterBuffer,
      1: targetBuffer,
      2: maskBuffer,
      3: phaseBuffer,
      4: fieldBuffer,
      5: inverseLutBuffer,
      6: phaseTableBuffer,
      7: codeBuffer,
      8: intensityBuffer,
      9: packedBuffer,
    };
    const bindingContract: Record<PipelineName, number[]> = {
      initialize_target_field: [0, 1, 4],
      extract_phase: [0, 3, 4],
      make_slm_field: [0, 3, 4],
      apply_mraf: [0, 1, 2, 4],
      quantize_codes: [0, 3, 5, 7],
      make_quantized_field: [0, 4, 6, 7],
      shift_intensity: [0, 4, 8],
      pack_active: [0, 7, 9],
    };
    const bindGroups = new Map<PipelineName, GPUBindGroup>();
    for (const name of pipelineNames) {
      bindGroups.set(name, device.createBindGroup({
        label: `Continuous ${name} bindings`,
        layout: pipelines.get(name)!.getBindGroupLayout(0),
        entries: bindingContract[name].map((binding) => ({ binding, resource: { buffer: resourceByBinding[binding]! } })),
      }));
    }

    const fft = await createWebGpuFft(device, fieldBuffer, input.fftWidth, input.fftHeight);
    fftParameters = fft.parameterBuffer;
    const encoder = device.createCommandEncoder({ label: "Continuous MRAF solve" });
    dispatch(encoder, pipelines, bindGroups, "initialize_target_field", pixelCount);
    fft.encode(encoder, true);
    dispatch(encoder, pipelines, bindGroups, "extract_phase", pixelCount);
    for (let iteration = 0; iteration < input.iterations; iteration += 1) {
      dispatch(encoder, pipelines, bindGroups, "make_slm_field", pixelCount);
      fft.encode(encoder, false);
      dispatch(encoder, pipelines, bindGroups, "apply_mraf", pixelCount);
      fft.encode(encoder, true);
      dispatch(encoder, pipelines, bindGroups, "extract_phase", pixelCount);
    }
    dispatch(encoder, pipelines, bindGroups, "quantize_codes", pixelCount);
    dispatch(encoder, pipelines, bindGroups, "make_quantized_field", pixelCount);
    fft.encode(encoder, false);
    dispatch(encoder, pipelines, bindGroups, "shift_intensity", pixelCount);
    dispatch(encoder, pipelines, bindGroups, "pack_active", Math.ceil(activePixelCount / 4));

    const intensityOffset = alignTo(packedBytes, 4);
    const readbackBytes = intensityOffset + scalarBytes;
    readback = device.createBuffer({
      label: "Continuous result readback",
      size: readbackBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(packedBuffer, 0, readback, 0, packedBytes);
    encoder.copyBufferToBuffer(intensityBuffer, 0, readback, intensityOffset, scalarBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const snapshot = readback.getMappedRange().slice(0);
    readback.unmap();
    const pixels = new Uint8Array(snapshot.slice(0, activePixelCount));
    const intensity = new Float32Array(snapshot.slice(intensityOffset, intensityOffset + scalarBytes));
    return finalizeContinuousFieldResult(
      input,
      target,
      pixels,
      intensity,
      "webgpu-mraf-radix2",
      performance.now() - started,
    );
  } finally {
    readback?.destroy();
    fftParameters?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}

function serializeParameters(
  input: ContinuousFieldInput,
  targetPower: number,
  regionWidth: number,
  regionHeight: number,
): ArrayBuffer {
  const output = new ArrayBuffer(96);
  const view = new DataView(output);
  const integers = [
    input.fftWidth,
    input.fftHeight,
    input.slmWidth,
    input.slmHeight,
    Math.floor((input.fftWidth - input.slmWidth) / 2),
    Math.floor((input.fftHeight - input.slmHeight) / 2),
    input.fftWidth * input.fftHeight,
    input.deterministicSeed >>> 0,
  ];
  integers.forEach((value, index) => view.setUint32(index * 4, value, true));
  const targetScale = continuousTargetScale(input, targetPower);
  view.setFloat32(32, input.mixingFactor, true);
  view.setFloat32(36, targetScale, true);
  view.setFloat32(40, regionWidth, true);
  view.setFloat32(44, regionHeight, true);
  view.setFloat32(48, input.opticalCalibration.pixelPitchUm, true);
  const beam = input.opticalCalibration.incidentBeam;
  view.setFloat32(52, (beam?.diameterXMm ?? 1) * 1000, true);
  view.setFloat32(56, (beam?.diameterYMm ?? 1) * 1000, true);
  view.setFloat32(60, (beam?.centerXMm ?? 0) * 1000, true);
  view.setFloat32(64, (beam?.centerYMm ?? 0) * 1000, true);
  view.setUint32(68, beam ? 1 : 0, true);
  view.setUint32(72, INVERSE_LUT_SIZE, true);
  view.setUint32(76, 256, true);
  return output;
}

function makeBuffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags): GPUBuffer {
  return device.createBuffer({ label, size, usage });
}

function dispatch<Name extends string>(
  encoder: GPUCommandEncoder,
  pipelines: Map<Name, GPUComputePipeline>,
  bindGroups: Map<Name, GPUBindGroup>,
  name: Name,
  invocationCount: number,
): void {
  const pass = encoder.beginComputePass({ label: name });
  pass.setPipeline(pipelines.get(name)!);
  pass.setBindGroup(0, bindGroups.get(name)!);
  pass.dispatchWorkgroups(Math.ceil(invocationCount / WORKGROUP_SIZE));
  pass.end();
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

const CONTINUOUS_FIELD_SHADER = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

struct Parameters {
  width: u32,
  height: u32,
  activeWidth: u32,
  activeHeight: u32,
  xStart: u32,
  yStart: u32,
  pixelCount: u32,
  seed: u32,
  mixing: f32,
  targetScale: f32,
  regionWidth: f32,
  regionHeight: f32,
  pixelPitchUm: f32,
  beamDiameterXUm: f32,
  beamDiameterYUm: f32,
  beamCenterXUm: f32,
  beamCenterYUm: f32,
  beamEnabled: u32,
  inverseLutSize: u32,
  phaseTableSize: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
  padding3: u32,
}

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> targetAmplitude: array<f32>;
@group(0) @binding(2) var<storage, read> signalMask: array<u32>;
@group(0) @binding(3) var<storage, read_write> phase: array<f32>;
@group(0) @binding(4) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> inverseLut: array<f32>;
@group(0) @binding(6) var<storage, read> phaseTable: array<f32>;
@group(0) @binding(7) var<storage, read_write> codes: array<u32>;
@group(0) @binding(8) var<storage, read_write> intensity: array<f32>;
@group(0) @binding(9) var<storage, read_write> packed: array<u32>;

fn wrap_phase(value: f32) -> f32 {
  let shifted = value + PI;
  return shifted - floor(shifted / TAU) * TAU - PI;
}

fn hash_value(key: u32) -> u32 {
  var value = (key ^ params.seed) + 1u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}

fn deterministic_phase(key: u32) -> f32 {
  return f32(hash_value(key) >> 8u) / 16777216.0 * TAU - PI;
}

fn signed_frequency(coordinate: u32, extent: u32) -> f32 {
  if (coordinate < extent / 2u) { return f32(coordinate); }
  return f32(i32(coordinate) - i32(extent));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn initialize_target_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let amplitude = targetAmplitude[index];
  let x = index % params.width;
  let y = index / params.width;
  let nx = signed_frequency(x, params.width) / max(1.0, params.regionWidth);
  let ny = signed_frequency(y, params.height) / max(1.0, params.regionHeight);
  let seedPhase = deterministic_phase(0u);
  let angle = wrap_phase(seedPhase + TAU * 6.0 * (nx * nx + ny * ny));
  field[index] = amplitude * vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn extract_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let value = field[index];
  phase[index] = select(deterministic_phase(index), atan2(value.y, value.x), dot(value, value) > 1e-20);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn make_slm_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let x = index % params.width;
  let y = index / params.width;
  if (x < params.xStart || y < params.yStart || x >= params.xStart + params.activeWidth || y >= params.yStart + params.activeHeight) {
    field[index] = vec2<f32>(0.0);
    return;
  }
  var amplitude = 1.0;
  if (params.beamEnabled != 0u) {
    let activeX = f32(x - params.xStart);
    let activeY = f32(y - params.yStart);
    let xUm = (activeX - (f32(params.activeWidth) - 1.0) * 0.5) * params.pixelPitchUm - params.beamCenterXUm;
    let yUm = (activeY - (f32(params.activeHeight) - 1.0) * 0.5) * params.pixelPitchUm - params.beamCenterYUm;
    let normalizedX = xUm / params.beamDiameterXUm;
    let normalizedY = yUm / params.beamDiameterYUm;
    amplitude = exp(-4.0 * (normalizedX * normalizedX + normalizedY * normalizedY));
  }
  field[index] = amplitude * vec2<f32>(cos(phase[index]), sin(phase[index]));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn apply_mraf(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let value = field[index];
  let magnitude = length(value);
  var constrained = (1.0 - params.mixing) * magnitude;
  if (signalMask[index] != 0u) {
    constrained = params.mixing * params.targetScale * targetAmplitude[index];
  }
  if (magnitude > 1e-20) {
    field[index] = constrained * value / magnitude;
  } else {
    field[index] = vec2<f32>(constrained, 0.0);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn quantize_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let position = (wrap_phase(phase[index]) + PI) / TAU * f32(params.inverseLutSize - 1u);
  let low = u32(floor(position));
  let high = min(params.inverseLutSize - 1u, low + 1u);
  codes[index] = u32(round(clamp(mix(inverseLut[low], inverseLut[high], position - f32(low)), 0.0, 255.0)));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn make_quantized_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let x = index % params.width;
  let y = index / params.width;
  if (x < params.xStart || y < params.yStart || x >= params.xStart + params.activeWidth || y >= params.yStart + params.activeHeight) {
    field[index] = vec2<f32>(0.0);
    return;
  }
  var amplitude = 1.0;
  if (params.beamEnabled != 0u) {
    let activeX = f32(x - params.xStart);
    let activeY = f32(y - params.yStart);
    let xUm = (activeX - (f32(params.activeWidth) - 1.0) * 0.5) * params.pixelPitchUm - params.beamCenterXUm;
    let yUm = (activeY - (f32(params.activeHeight) - 1.0) * 0.5) * params.pixelPitchUm - params.beamCenterYUm;
    let normalizedX = xUm / params.beamDiameterXUm;
    let normalizedY = yUm / params.beamDiameterYUm;
    amplitude = exp(-4.0 * (normalizedX * normalizedX + normalizedY * normalizedY));
  }
  let decoded = phaseTable[min(codes[index], params.phaseTableSize - 1u)];
  field[index] = amplitude * vec2<f32>(cos(decoded), sin(decoded));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn shift_intensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let outputX = index % params.width;
  let outputY = index / params.width;
  let sourceX = (outputX + params.width / 2u) % params.width;
  let sourceY = (outputY + params.height / 2u) % params.height;
  let value = field[sourceY * params.width + sourceX];
  intensity[index] = dot(value, value);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn pack_active(@builtin(global_invocation_id) gid: vec3<u32>) {
  let wordIndex = gid.x;
  let activePixels = params.activeWidth * params.activeHeight;
  let firstPixel = wordIndex * 4u;
  if (firstPixel >= activePixels) { return; }
  var word = 0u;
  for (var part = 0u; part < 4u; part = part + 1u) {
    let activeIndex = firstPixel + part;
    if (activeIndex < activePixels) {
      let activeX = activeIndex % params.activeWidth;
      let activeY = activeIndex / params.activeWidth;
      let source = (params.yStart + activeY) * params.width + params.xStart + activeX;
      word = word | (codes[source] << (part * 8u));
    }
  }
  packed[wordIndex] = word;
}
`;
