import {
  createForwardPhaseTable,
  finalizeForwardIntensity,
  validateForwardSimulationInput,
  type ForwardSimulationInput,
  type ForwardSimulationResult,
} from "./forward-simulation.js";
import { WEBGPU_RADIX2_FFT_SHADER } from "./webgpu-wgs.js";

const WORKGROUP_SIZE = 256;

export async function simulateSlmFrameWebGpu(input: ForwardSimulationInput): Promise<ForwardSimulationResult> {
  validateForwardSimulationInput(input);
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error("WebGPU is unavailable in this browser worker");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No compatible WebGPU adapter was found");
  const device = await adapter.requestDevice();
  const pixelCount = input.fftWidth * input.fftHeight;
  const complexBytes = pixelCount * 8;
  const scalarBytes = pixelCount * 4;
  if (complexBytes > device.limits.maxStorageBufferBindingSize || complexBytes > device.limits.maxBufferSize) {
    device.destroy();
    throw new Error("The selected FFT grid exceeds this GPU's storage-buffer limit");
  }
  if (Math.ceil(pixelCount / WORKGROUP_SIZE) > device.limits.maxComputeWorkgroupsPerDimension) {
    device.destroy();
    throw new Error("The selected FFT grid exceeds this GPU's dispatch limit");
  }

  const storage = GPUBufferUsage.STORAGE;
  const copyDestination = GPUBufferUsage.COPY_DST;
  const buffers: GPUBuffer[] = [];
  let fftParameters: GPUBuffer | undefined;
  let readback: GPUBuffer | undefined;
  try {
    const parameters = new ArrayBuffer(64);
    const parameterView = new DataView(parameters);
    [
      input.fftWidth,
      input.fftHeight,
      input.width,
      input.height,
      Math.floor((input.fftWidth - input.width) / 2),
      Math.floor((input.fftHeight - input.height) / 2),
      pixelCount,
      0,
    ].forEach((value, index) => parameterView.setUint32(index * 4, value, true));
    const beam = input.incidentBeam;
    parameterView.setFloat32(32, input.pixelPitchUm ?? 1, true);
    parameterView.setFloat32(36, (beam?.diameterXMm ?? 1) * 1000, true);
    parameterView.setFloat32(40, (beam?.diameterYMm ?? 1) * 1000, true);
    parameterView.setFloat32(44, (beam?.centerXMm ?? 0) * 1000, true);
    parameterView.setFloat32(48, (beam?.centerYMm ?? 0) * 1000, true);
    parameterView.setUint32(52, beam ? 1 : 0, true);
    const codes = Uint32Array.from(input.pixels);
    const phaseTable = createForwardPhaseTable(input.phaseResponseLut);
    const parameterBuffer = device.createBuffer({
      label: "Forward simulation parameters",
      size: parameters.byteLength,
      usage: GPUBufferUsage.UNIFORM | copyDestination,
    });
    const codeBuffer = device.createBuffer({
      label: "Forward simulation phase codes",
      size: codes.byteLength,
      usage: storage | copyDestination,
    });
    const fieldBuffer = device.createBuffer({
      label: "Forward simulation complex field",
      size: complexBytes,
      usage: storage,
    });
    const intensityBuffer = device.createBuffer({
      label: "Forward simulation shifted intensity",
      size: scalarBytes,
      usage: storage | GPUBufferUsage.COPY_SRC,
    });
    const phaseTableBuffer = device.createBuffer({
      label: "Forward simulation phase-response table",
      size: phaseTable.byteLength,
      usage: storage | copyDestination,
    });
    buffers.push(parameterBuffer, codeBuffer, fieldBuffer, intensityBuffer, phaseTableBuffer);
    device.queue.writeBuffer(parameterBuffer, 0, parameters);
    device.queue.writeBuffer(codeBuffer, 0, codes.buffer as ArrayBuffer);
    device.queue.writeBuffer(phaseTableBuffer, 0, phaseTable.buffer as ArrayBuffer);

    const forwardModule = device.createShaderModule({ label: "SLM forward propagation", code: FORWARD_SHADER });
    const [makeFieldPipeline, intensityPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: "Decode SLM phase codes",
        layout: "auto",
        compute: { module: forwardModule, entryPoint: "make_field" },
      }),
      device.createComputePipelineAsync({
        label: "FFT-shift focal intensity",
        layout: "auto",
        compute: { module: forwardModule, entryPoint: "shift_intensity" },
      }),
    ]);
    const makeFieldBindings = device.createBindGroup({
      label: "Forward field bindings",
      layout: makeFieldPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: parameterBuffer } },
        { binding: 1, resource: { buffer: codeBuffer } },
        { binding: 2, resource: { buffer: fieldBuffer } },
        { binding: 4, resource: { buffer: phaseTableBuffer } },
      ],
    });
    const intensityBindings = device.createBindGroup({
      label: "Forward intensity bindings",
      layout: intensityPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: parameterBuffer } },
        { binding: 2, resource: { buffer: fieldBuffer } },
        { binding: 3, resource: { buffer: intensityBuffer } },
      ],
    });

    const fft = await createForwardFft(device, fieldBuffer, input.fftWidth, input.fftHeight);
    fftParameters = fft.parameterBuffer;
    const encoder = device.createCommandEncoder({ label: "SLM forward simulation" });
    dispatch(encoder, makeFieldPipeline, makeFieldBindings, pixelCount);
    fft.encode(encoder);
    dispatch(encoder, intensityPipeline, intensityBindings, pixelCount);
    readback = device.createBuffer({
      label: "Forward simulation readback",
      size: scalarBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(intensityBuffer, 0, readback, 0, scalarBytes);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const snapshot = readback.getMappedRange().slice(0);
    readback.unmap();
    return finalizeForwardIntensity(
      input,
      new Float32Array(snapshot),
      "webgpu-radix2-forward-fft",
    );
  } finally {
    readback?.destroy();
    fftParameters?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device.destroy();
  }
}

interface ForwardFft {
  parameterBuffer: GPUBuffer;
  encode(encoder: GPUCommandEncoder): void;
}

async function createForwardFft(
  device: GPUDevice,
  field: GPUBuffer,
  width: number,
  height: number,
): Promise<ForwardFft> {
  const logWidth = Math.log2(width);
  const logHeight = Math.log2(height);
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  const stride = alignTo(32, alignment);
  const records: Array<{ key: string; values: number[] }> = [];
  records.push({ key: "reverse:h", values: [width, height, logWidth, 0, 0, 0, 0, 0] });
  for (let stage = 1; stage <= logWidth; stage += 1) {
    records.push({ key: `butterfly:h:${stage}`, values: [width, height, stage, 0, 0, 0, 0, 0] });
  }
  records.push({ key: "reverse:v", values: [width, height, logHeight, 0, 1, 0, 0, 0] });
  for (let stage = 1; stage <= logHeight; stage += 1) {
    records.push({ key: `butterfly:v:${stage}`, values: [width, height, stage, 0, 1, 0, 0, 0] });
  }
  const offsets = new Map<string, number>();
  const serialized = new Uint8Array(stride * records.length);
  records.forEach((record, index) => {
    const offset = index * stride;
    offsets.set(record.key, offset);
    const view = new DataView(serialized.buffer, offset, 32);
    record.values.forEach((value, valueIndex) => view.setUint32(valueIndex * 4, value, true));
  });
  const parameterBuffer = device.createBuffer({
    label: "Forward FFT stage parameters",
    size: serialized.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(parameterBuffer, 0, serialized.buffer as ArrayBuffer);
  const layout = device.createBindGroupLayout({
    label: "Forward FFT bindings",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const module = device.createShaderModule({ label: "Forward radix-2 FFT", code: WEBGPU_RADIX2_FFT_SHADER });
  const [reversePipeline, butterflyPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: "Forward FFT bit reversal",
      layout: pipelineLayout,
      compute: { module, entryPoint: "bit_reverse" },
    }),
    device.createComputePipelineAsync({
      label: "Forward FFT butterfly",
      layout: pipelineLayout,
      compute: { module, entryPoint: "butterfly" },
    }),
  ]);
  const bindGroup = device.createBindGroup({
    label: "Forward FFT field bindings",
    layout,
    entries: [
      { binding: 0, resource: { buffer: field } },
      { binding: 1, resource: { buffer: parameterBuffer, size: 32 } },
    ],
  });
  const pixelCount = width * height;
  return {
    parameterBuffer,
    encode(encoder) {
      const pass = encoder.beginComputePass({ label: "Forward 2D FFT" });
      pass.setPipeline(reversePipeline);
      pass.setBindGroup(0, bindGroup, [offsets.get("reverse:h")!]);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / WORKGROUP_SIZE));
      pass.setPipeline(butterflyPipeline);
      for (let stage = 1; stage <= logWidth; stage += 1) {
        pass.setBindGroup(0, bindGroup, [offsets.get(`butterfly:h:${stage}`)!]);
        pass.dispatchWorkgroups(Math.ceil((pixelCount / 2) / WORKGROUP_SIZE));
      }
      pass.setPipeline(reversePipeline);
      pass.setBindGroup(0, bindGroup, [offsets.get("reverse:v")!]);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / WORKGROUP_SIZE));
      pass.setPipeline(butterflyPipeline);
      for (let stage = 1; stage <= logHeight; stage += 1) {
        pass.setBindGroup(0, bindGroup, [offsets.get(`butterfly:v:${stage}`)!]);
        pass.dispatchWorkgroups(Math.ceil((pixelCount / 2) / WORKGROUP_SIZE));
      }
      pass.end();
    },
  };
}

function dispatch(
  encoder: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  count: number,
): void {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE));
  pass.end();
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

const FORWARD_SHADER = /* wgsl */ `
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
  padding: u32,
  pixelPitchUm: f32,
  beamDiameterXUm: f32,
  beamDiameterYUm: f32,
  beamCenterXUm: f32,
  beamCenterYUm: f32,
  beamEnabled: u32,
  padding2: u32,
  padding3: u32,
}

@group(0) @binding(0) var<uniform> params: Parameters;
@group(0) @binding(1) var<storage, read> codes: array<u32>;
@group(0) @binding(2) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> intensity: array<f32>;
@group(0) @binding(4) var<storage, read> phaseByCode: array<f32>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn make_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.pixelCount) { return; }
  let x = index % params.width;
  let y = index / params.width;
  if (x < params.xStart || y < params.yStart || x >= params.xStart + params.activeWidth || y >= params.yStart + params.activeHeight) {
    field[index] = vec2<f32>(0.0);
    return;
  }
  let activeIndex = (y - params.yStart) * params.activeWidth + (x - params.xStart);
  let phase = phaseByCode[codes[activeIndex]];
  var amplitude = 1.0;
  if (params.beamEnabled != 0u) {
    let activeX = f32(activeIndex % params.activeWidth);
    let activeY = f32(activeIndex / params.activeWidth);
    let xUm = (activeX - (f32(params.activeWidth) - 1.0) * 0.5) * params.pixelPitchUm - params.beamCenterXUm;
    let yUm = (activeY - (f32(params.activeHeight) - 1.0) * 0.5) * params.pixelPitchUm - params.beamCenterYUm;
    let normalizedX = xUm / params.beamDiameterXUm;
    let normalizedY = yUm / params.beamDiameterYUm;
    amplitude = exp(-4.0 * (normalizedX * normalizedX + normalizedY * normalizedY));
  }
  field[index] = amplitude * vec2<f32>(cos(phase), sin(phase));
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
`;
