import { WEBGPU_RADIX2_FFT_SHADER } from "./webgpu-wgs.js";

const WORKGROUP_SIZE = 256;

export interface WebGpuFft {
  readonly parameterBuffer: GPUBuffer;
  encode(encoder: GPUCommandEncoder, inverse?: boolean): void;
}

/** Shared in-place radix-2 FFT used by forward simulation and continuous MRAF. */
export async function createWebGpuFft(
  device: GPUDevice,
  field: GPUBuffer,
  width: number,
  height: number,
): Promise<WebGpuFft> {
  if (!isPowerOfTwo(width) || !isPowerOfTwo(height)) throw new Error("WebGPU FFT dimensions must be powers of two");
  const logWidth = Math.log2(width);
  const logHeight = Math.log2(height);
  const alignment = device.limits.minUniformBufferOffsetAlignment;
  const stride = alignTo(32, alignment);
  const records: Array<{ key: string; values: number[] }> = [];
  records.push({ key: "reverse:h", values: [width, height, logWidth, 0, 0, 0, 0, 0] });
  records.push({ key: "reverse:v", values: [width, height, logHeight, 0, 1, 0, 0, 0] });
  for (const inverse of [0, 1]) {
    for (let stage = 1; stage <= logWidth; stage += 1) {
      records.push({ key: `butterfly:${inverse}:h:${stage}`, values: [width, height, stage, inverse, 0, 0, 0, 0] });
    }
    for (let stage = 1; stage <= logHeight; stage += 1) {
      records.push({ key: `butterfly:${inverse}:v:${stage}`, values: [width, height, stage, inverse, 1, 0, 0, 0] });
    }
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
    label: "FFT stage parameters",
    size: serialized.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(parameterBuffer, 0, serialized.buffer as ArrayBuffer);
  const layout = device.createBindGroupLayout({
    label: "FFT bindings",
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const module = device.createShaderModule({ label: "Radix-2 FFT", code: WEBGPU_RADIX2_FFT_SHADER });
  const [reversePipeline, butterflyPipeline] = await Promise.all([
    device.createComputePipelineAsync({
      label: "FFT bit reversal",
      layout: pipelineLayout,
      compute: { module, entryPoint: "bit_reverse" },
    }),
    device.createComputePipelineAsync({
      label: "FFT butterfly",
      layout: pipelineLayout,
      compute: { module, entryPoint: "butterfly" },
    }),
  ]);
  const bindGroup = device.createBindGroup({
    label: "FFT field bindings",
    layout,
    entries: [
      { binding: 0, resource: { buffer: field } },
      { binding: 1, resource: { buffer: parameterBuffer, size: 32 } },
    ],
  });
  const pixelCount = width * height;
  return {
    parameterBuffer,
    encode(encoder, inverse = false) {
      const inverseFlag = inverse ? 1 : 0;
      const pass = encoder.beginComputePass({ label: inverse ? "Inverse 2D FFT" : "Forward 2D FFT" });
      pass.setPipeline(reversePipeline);
      pass.setBindGroup(0, bindGroup, [offsets.get("reverse:h")!]);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / WORKGROUP_SIZE));
      pass.setPipeline(butterflyPipeline);
      for (let stage = 1; stage <= logWidth; stage += 1) {
        pass.setBindGroup(0, bindGroup, [offsets.get(`butterfly:${inverseFlag}:h:${stage}`)!]);
        pass.dispatchWorkgroups(Math.ceil((pixelCount / 2) / WORKGROUP_SIZE));
      }
      pass.setPipeline(reversePipeline);
      pass.setBindGroup(0, bindGroup, [offsets.get("reverse:v")!]);
      pass.dispatchWorkgroups(Math.ceil(pixelCount / WORKGROUP_SIZE));
      pass.setPipeline(butterflyPipeline);
      for (let stage = 1; stage <= logHeight; stage += 1) {
        pass.setBindGroup(0, bindGroup, [offsets.get(`butterfly:${inverseFlag}:v:${stage}`)!]);
        pass.dispatchWorkgroups(Math.ceil((pixelCount / 2) / WORKGROUP_SIZE));
      }
      pass.end();
    },
  };
}

function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}
