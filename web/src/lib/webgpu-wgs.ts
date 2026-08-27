import { mapPhysicalPointToDftFrequency } from "../../../src/coordinates.js";
import { SlmError } from "../../../src/errors.js";
import type {
  CalibrationPackage,
  FrameMetrics,
  HologramConfig,
  HologramSolveResult,
  SequentialHologramBackend,
  TrapFrame,
  TrapState,
} from "../../../src/types.js";
import { angularDistance, clamp, wrapPhase } from "../../../src/util.js";
import {
  WGS_INITIALIZATION_CANCELLATION_RATIO,
  WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN,
  WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN,
  WGS_REFERENCE_TRAP_AMPLITUDE_GAIN,
  WGS_SOFT_PHASE_PRECOMPENSATION_GAIN,
} from "../../../src/wgs-constants.js";

export const WEBGPU_WGS_BACKEND_ID = "webgpu-exact-nudft-phase-locked-wgs";

export interface WebGpuCapability {
  available: boolean;
  reason: string;
  adapter?: string;
}

const TAU = Math.PI * 2;
const WORKGROUP_SIZE = 256;
const TARGET_WORKGROUP_SIZE = 64;
const EXACT_TARGET_WORKGROUP_SIZE = 256;
const TARGET_STRIDE = 32;
const MAX_TARGETS = 4096;
const INVERSE_LUT_SIZE = 4096;

type GeneralPipelineName =
  | "initialize_targets"
  | "initialize_optimizer"
  | "initialize_phase"
  | "sample_targets"
  | "evaluate_candidate"
  | "save_best_phase_codes"
  | "save_best_targets"
  | "restore_best_phase_codes"
  | "restore_best_targets"
  | "update_controls"
  | "synthesize_phase"
  | "clear_support"
  | "mark_support"
  | "quantize_codes"
  | "make_final_field"
  | "reduce_field"
  | "finish_reduction"
  | "pack_active";

/**
 * Explicit per-entry-point resources. WebGPU's auto layouts reject both
 * missing and surplus bind-group entries, so keep this contract beside the
 * shader and cover phase initialization with a regression test.
 */
export const WEBGPU_WGS_PIPELINE_BINDINGS: Readonly<Record<GeneralPipelineName, readonly number[]>> = {
  initialize_targets: [0, 1, 2, 3, 4],
  initialize_optimizer: [19],
  initialize_phase: [0, 1, 2, 5, 6],
  sample_targets: [0, 1, 2, 8],
  evaluate_candidate: [0, 1, 2, 19],
  save_best_phase_codes: [0, 5, 13, 19, 20, 21],
  save_best_targets: [0, 2, 19, 22],
  restore_best_phase_codes: [0, 5, 13, 20, 21],
  restore_best_targets: [0, 2, 22],
  update_controls: [0, 1, 2, 19],
  synthesize_phase: [0, 1, 2, 5, 19],
  clear_support: [0, 18],
  mark_support: [0, 1, 18],
  quantize_codes: [0, 5, 10, 11, 13],
  make_final_field: [0, 7, 8, 12, 13],
  reduce_field: [0, 8, 13, 14, 15, 18],
  finish_reduction: [0, 15, 16],
  pack_active: [0, 13, 17],
};

interface CandidateState {
  frameIndex: number;
  targetCount: number;
  measuredPhases: Map<number, number>;
  measuredIntensities: Map<number, number>;
}

interface ExpandedCalibration {
  amplitude: Float32Array;
  correction: Float32Array;
  inverseLut: Float32Array;
  decodeLut: Float32Array;
}

export async function inspectWebGpu(): Promise<WebGpuCapability> {
  if (typeof navigator === "undefined" || !navigator.gpu) {
    return {
      available: false,
      reason: "WebGPU is not exposed by this browser or graphics configuration",
    };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      return { available: false, reason: "No compatible WebGPU adapter was found" };
    }
    const info = adapter.info;
    const adapterName = [info.vendor, info.architecture, info.device]
      .map((value) => value.trim())
      .filter(Boolean)
      .join(" / ");
    return {
      available: true,
      reason: "GPU compute is available in the dedicated worker",
      ...(adapterName ? { adapter: adapterName } : {}),
    };
  } catch (error) {
    return {
      available: false,
      reason: error instanceof Error ? error.message : "Unable to inspect WebGPU support",
    };
  }
}

/**
 * Exact trap-domain WGS implementation whose phase field, arbitrary-frequency
 * Fourier samples, target weights, quantized codes, and accepted sequential
 * state stay in GPU buffers. A radix-2 FFT is used only once, after the solve,
 * for full-plane diagnostics. One compact final readback is performed.
 */
export class WebGpuSequentialWgsSolver implements SequentialHologramBackend {
  readonly backendId = WEBGPU_WGS_BACKEND_ID;
  readonly width: number;
  readonly height: number;

  private readonly device: GPUDevice;
  private readonly calibration: CalibrationPackage;
  private readonly config: Required<HologramConfig>;
  private readonly pixelCount: number;
  private readonly activePixelCount: number;
  private readonly partialCount: number;
  private readonly packedByteLength: number;
  private readonly logWidth: number;
  private readonly logHeight: number;
  private readonly buffers: Record<string, GPUBuffer>;
  private readonly pipelines = new Map<GeneralPipelineName, GPUComputePipeline>();
  private readonly bindGroups = new Map<GeneralPipelineName, GPUBindGroup>();
  private fftPipeline!: GPUComputePipeline;
  private bitReversePipeline!: GPUComputePipeline;
  private fftBindGroup!: GPUBindGroup;
  private fftParameterBuffer!: GPUBuffer;
  private readonly fftParameterOffsets = new Map<string, number>();
  private acceptedTargetCount = 0;
  private accepted: CandidateState | undefined;
  private candidate: CandidateState | undefined;
  private disposed = false;
  private deviceLossReason = "";

  private constructor(
    device: GPUDevice,
    calibration: CalibrationPackage,
    config: HologramConfig | Required<HologramConfig>,
  ) {
    this.device = device;
    this.calibration = calibration;
    this.config = normalizeConfig(calibration, config);
    this.width = this.config.width;
    this.height = this.config.height;
    assertGpuDimensions(calibration, this.width, this.height);
    this.pixelCount = this.width * this.height;
    this.activePixelCount = calibration.manifest.activeWidth * calibration.manifest.activeHeight;
    this.partialCount = Math.ceil(this.pixelCount / WORKGROUP_SIZE);
    const pixelsPerWord = this.config.format === "UINT8" ? 4 : 2;
    this.packedByteLength = Math.ceil(this.activePixelCount / pixelsPerWord) * 4;
    this.logWidth = Math.log2(this.width);
    this.logHeight = Math.log2(this.height);
    this.assertDeviceLimits();

    const storage = GPUBufferUsage.STORAGE;
    const copySource = GPUBufferUsage.COPY_SRC;
    const copyDestination = GPUBufferUsage.COPY_DST;
    const scalarBytes = this.pixelCount * 4;
    const complexBytes = this.pixelCount * 8;
    const targetBytes = MAX_TARGETS * TARGET_STRIDE;
    this.buffers = {
      frameParams: device.createBuffer({ label: "WGS frame parameters", size: 96, usage: GPUBufferUsage.UNIFORM | copyDestination }),
      targetInput: device.createBuffer({ label: "WGS target inputs", size: targetBytes, usage: storage | copySource | copyDestination }),
      targetState: device.createBuffer({ label: "WGS target state", size: targetBytes, usage: storage | copySource | copyDestination }),
      bestTargetState: device.createBuffer({ label: "WGS best target state", size: targetBytes, usage: storage }),
      acceptedTargetInput: device.createBuffer({ label: "WGS accepted target inputs", size: targetBytes, usage: storage | copySource | copyDestination }),
      acceptedTargetState: device.createBuffer({ label: "WGS accepted target state", size: targetBytes, usage: storage | copySource | copyDestination }),
      phase: device.createBuffer({ label: "WGS phase", size: scalarBytes, usage: storage | copySource | copyDestination }),
      bestPhase: device.createBuffer({ label: "WGS best phase", size: scalarBytes, usage: storage }),
      acceptedPhase: device.createBuffer({ label: "WGS accepted phase", size: scalarBytes, usage: storage | copySource | copyDestination }),
      amplitude: device.createBuffer({ label: "WGS incident amplitude", size: scalarBytes, usage: storage | copyDestination }),
      field: device.createBuffer({ label: "WGS Fourier field", size: complexBytes, usage: storage }),
      correction: device.createBuffer({ label: "WGS phase correction", size: scalarBytes, usage: storage | copyDestination }),
      inverseLut: device.createBuffer({ label: "WGS inverse phase LUT", size: INVERSE_LUT_SIZE * 4, usage: storage | copyDestination }),
      decodeLut: device.createBuffer({ label: "WGS phase response LUT", size: (this.config.format === "UINT8" ? 256 : 65536) * 4, usage: storage | copyDestination }),
      codes: device.createBuffer({ label: "WGS display codes", size: scalarBytes, usage: storage | copySource | copyDestination }),
      bestCodes: device.createBuffer({ label: "WGS best display codes", size: scalarBytes, usage: storage }),
      acceptedCodes: device.createBuffer({ label: "WGS accepted display codes", size: scalarBytes, usage: storage | copyDestination }),
      optimizer: device.createBuffer({ label: "WGS optimizer state", size: 32, usage: storage | copySource }),
      partialMetrics: device.createBuffer({ label: "WGS metric partials", size: this.partialCount * 16, usage: storage }),
      summary: device.createBuffer({ label: "WGS metric summary", size: 16, usage: storage | copySource }),
      packed: device.createBuffer({ label: "WGS packed active frame", size: this.packedByteLength, usage: storage | copySource }),
      supportMask: device.createBuffer({ label: "WGS target support mask", size: scalarBytes, usage: storage }),
    };

    const expanded = expandCalibration(calibration, this.config);
    device.queue.writeBuffer(this.buffers.amplitude!, 0, expanded.amplitude.buffer as ArrayBuffer);
    device.queue.writeBuffer(this.buffers.correction!, 0, expanded.correction.buffer as ArrayBuffer);
    device.queue.writeBuffer(this.buffers.inverseLut!, 0, expanded.inverseLut.buffer as ArrayBuffer);
    device.queue.writeBuffer(this.buffers.decodeLut!, 0, expanded.decodeLut.buffer as ArrayBuffer);
    void device.lost.then((info) => {
      this.deviceLossReason = info.message || info.reason;
    });
  }

  static async create(
    calibration: CalibrationPackage,
    config: HologramConfig | Required<HologramConfig> = {},
  ): Promise<WebGpuSequentialWgsSolver> {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new SlmError("INVALID_ARGUMENT", "WebGPU is unavailable in this browser worker", {
        stage: "SOLVING_SLM_FRAMES",
      });
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new SlmError("INVALID_ARGUMENT", "No compatible WebGPU adapter was found", {
        stage: "SOLVING_SLM_FRAMES",
      });
    }
    const device = await adapter.requestDevice();
    const solver = new WebGpuSequentialWgsSolver(device, calibration, config);
    try {
      await solver.initializePipelines();
      return solver;
    } catch (error) {
      solver.dispose();
      throw error;
    }
  }

  async solveSequentialFrame(frame: TrapFrame, iterationBudget?: number): Promise<HologramSolveResult> {
    this.assertUsable();
    validateFrame(frame);
    if (frame.traps.length > MAX_TARGETS) {
      throw new SlmError("INVALID_ARGUMENT", `WebGPU supports at most ${MAX_TARGETS} targets per frame`, {
        stage: "SOLVING_SLM_FRAMES",
      });
    }
    const mappedTargets = frame.traps.map((trap) => this.mapCoordinate(trap));

    const started = nowMs();
    const targetCount = frame.traps.length;
    const iterations = Math.min(
      Math.max(1, iterationBudget ?? (this.accepted ? this.config.subsequentFrameIterations : this.config.firstFrameIterations)),
      this.config.maxIterations,
    );
    const targetInput = serializeTargets(frame, mappedTargets);
    if (targetInput.byteLength > 0) this.device.queue.writeBuffer(this.buffers.targetInput!, 0, targetInput);
    this.device.queue.writeBuffer(this.buffers.frameParams!, 0, this.serializeFrameParameters(targetCount));

    const encoder = this.device.createCommandEncoder({ label: `WGS frame ${frame.frameIndex}` });
    this.dispatch(encoder, "initialize_targets", targetCount, TARGET_WORKGROUP_SIZE);
    this.dispatch(encoder, "initialize_optimizer", 1, 1);
    this.dispatch(encoder, "initialize_phase", this.pixelCount);
    this.dispatch(encoder, "quantize_codes", this.pixelCount);
    this.dispatch(encoder, "make_final_field", this.pixelCount);
    this.dispatch(encoder, "sample_targets", targetCount * EXACT_TARGET_WORKGROUP_SIZE, EXACT_TARGET_WORKGROUP_SIZE);
    this.dispatch(encoder, "evaluate_candidate", 1, 1);
    this.dispatch(encoder, "save_best_phase_codes", this.pixelCount);
    this.dispatch(encoder, "save_best_targets", targetCount, TARGET_WORKGROUP_SIZE);
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      this.dispatch(encoder, "update_controls", 1, 1);
      this.dispatch(encoder, "synthesize_phase", this.pixelCount);
      this.dispatch(encoder, "quantize_codes", this.pixelCount);
      this.dispatch(encoder, "make_final_field", this.pixelCount);
      this.dispatch(encoder, "sample_targets", targetCount * EXACT_TARGET_WORKGROUP_SIZE, EXACT_TARGET_WORKGROUP_SIZE);
      this.dispatch(encoder, "evaluate_candidate", 1, 1);
      this.dispatch(encoder, "save_best_phase_codes", this.pixelCount);
      this.dispatch(encoder, "save_best_targets", targetCount, TARGET_WORKGROUP_SIZE);
    }
    this.dispatch(encoder, "restore_best_phase_codes", this.pixelCount);
    this.dispatch(encoder, "restore_best_targets", targetCount, TARGET_WORKGROUP_SIZE);
    this.dispatch(encoder, "make_final_field", this.pixelCount);
    this.dispatch(encoder, "sample_targets", targetCount * EXACT_TARGET_WORKGROUP_SIZE, EXACT_TARGET_WORKGROUP_SIZE);
    this.dispatch(encoder, "clear_support", this.pixelCount);
    this.dispatch(encoder, "mark_support", 1, 1);
    this.encodeFft(encoder, this.fftBindGroup, false);
    this.dispatch(encoder, "reduce_field", this.pixelCount);
    this.dispatch(encoder, "finish_reduction", 1, 1);
    const packedWords = this.packedByteLength / 4;
    this.dispatch(encoder, "pack_active", packedWords);

    const targetStateBytes = targetCount * TARGET_STRIDE;
    const stateOffset = this.packedByteLength;
    const summaryOffset = alignTo(stateOffset + targetStateBytes, 4);
    const optimizerOffset = summaryOffset + 16;
    const readbackSize = optimizerOffset + 32;
    const readback = this.device.createBuffer({
      label: `WGS frame ${frame.frameIndex} readback`,
      size: readbackSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(this.buffers.packed!, 0, readback, 0, this.packedByteLength);
    if (targetStateBytes > 0) {
      encoder.copyBufferToBuffer(this.buffers.targetState!, 0, readback, stateOffset, targetStateBytes);
    }
    encoder.copyBufferToBuffer(this.buffers.summary!, 0, readback, summaryOffset, 16);
    encoder.copyBufferToBuffer(this.buffers.optimizer!, 0, readback, optimizerOffset, 32);
    this.device.queue.submit([encoder.finish()]);

    try {
      await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange();
      const snapshot = mapped.slice(0);
      readback.unmap();
      const pixels = unpackPixels(
        snapshot.slice(0, this.packedByteLength),
        this.activePixelCount,
        this.config.format,
      );
      const targetStates = parseTargetStates(snapshot, stateOffset, frame);
      const summary = new DataView(snapshot, summaryOffset, 16);
      const totalPower = summary.getFloat32(0, true);
      const maximumGhostIntensity = targetCount === 0 ? 0 : summary.getFloat32(4, true);
      const displayCodeChange = this.accepted ? summary.getFloat32(8, true) : 0;
      const optimizer = new DataView(snapshot, optimizerOffset, 32);
      const performedIterations = Math.max(1, optimizer.getUint32(20, true));
      const metrics = this.evaluateMetrics(
        frame,
        targetStates,
        totalPower,
        maximumGhostIntensity,
        displayCodeChange,
        performedIterations,
        started,
      );
      this.candidate = {
        frameIndex: frame.frameIndex,
        targetCount,
        measuredPhases: targetStates.measuredPhases,
        measuredIntensities: targetStates.measuredIntensities,
      };
      return { pixels, metrics };
    } finally {
      readback.destroy();
    }
  }

  /**
   * Return the target-plane phases that were already included in the latest
   * solve readback. This accessor performs no additional GPU work and keeps the
   * numerical WGS pipeline unchanged.
   */
  getCandidateMeasuredPhases(frame: TrapFrame): Float32Array {
    this.assertUsable();
    const candidate = this.candidate;
    if (!candidate || candidate.frameIndex !== frame.frameIndex || candidate.targetCount !== frame.traps.length) {
      throw new SlmError("INVALID_ARGUMENT", "No matching WebGPU hologram candidate is available", {
        stage: "SOLVING_SLM_FRAMES",
      });
    }
    return Float32Array.from(frame.traps, (trap) => {
      const phase = candidate.measuredPhases.get(trap.trapId);
      if (phase === undefined) {
        throw new SlmError("NUMERIC_ERROR", `The WebGPU result is missing measured phase for trap ${trap.trapId}`, {
          stage: "SOLVING_SLM_FRAMES",
        });
      }
      return phase;
    });
  }

  commitFrameState(): void {
    this.assertUsable();
    if (!this.candidate) {
      throw new SlmError("INVALID_ARGUMENT", "No WebGPU hologram candidate is available to commit", {
        stage: "SOLVING_SLM_FRAMES",
      });
    }
    const encoder = this.device.createCommandEncoder({ label: `Commit WGS frame ${this.candidate.frameIndex}` });
    encoder.copyBufferToBuffer(this.buffers.phase!, 0, this.buffers.acceptedPhase!, 0, this.pixelCount * 4);
    encoder.copyBufferToBuffer(this.buffers.codes!, 0, this.buffers.acceptedCodes!, 0, this.pixelCount * 4);
    if (this.candidate.targetCount > 0) {
      const targetBytes = this.candidate.targetCount * TARGET_STRIDE;
      encoder.copyBufferToBuffer(this.buffers.targetInput!, 0, this.buffers.acceptedTargetInput!, 0, targetBytes);
      encoder.copyBufferToBuffer(this.buffers.targetState!, 0, this.buffers.acceptedTargetState!, 0, targetBytes);
    }
    this.device.queue.submit([encoder.finish()]);
    this.acceptedTargetCount = this.candidate.targetCount;
    this.accepted = this.candidate;
    this.candidate = undefined;
  }

  rollbackToPreviousAcceptedFrame(): void {
    this.candidate = undefined;
  }

  beginSequence(): void {
    this.acceptedTargetCount = 0;
    this.accepted = undefined;
    this.candidate = undefined;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of Object.values(this.buffers)) buffer.destroy();
    this.fftParameterBuffer?.destroy();
    this.device.destroy();
  }

  private async initializePipelines(): Promise<void> {
    const module = this.device.createShaderModule({ label: "GPU-resident WGS", code: WGS_SHADER });
    const names: GeneralPipelineName[] = [
      "initialize_targets",
      "initialize_optimizer",
      "initialize_phase",
      "sample_targets",
      "evaluate_candidate",
      "save_best_phase_codes",
      "save_best_targets",
      "restore_best_phase_codes",
      "restore_best_targets",
      "update_controls",
      "synthesize_phase",
      "clear_support",
      "mark_support",
      "quantize_codes",
      "make_final_field",
      "reduce_field",
      "finish_reduction",
      "pack_active",
    ];
    const compiled = await Promise.all(names.map(async (name) => [name, await this.device.createComputePipelineAsync({
      label: `WGS ${name}`,
      layout: "auto",
      compute: { module, entryPoint: name },
    })] as const));
    for (const [name, pipeline] of compiled) this.pipelines.set(name, pipeline);

    const bufferByBinding: Record<number, GPUBuffer> = {
      0: this.buffers.frameParams!,
      1: this.buffers.targetInput!,
      2: this.buffers.targetState!,
      3: this.buffers.acceptedTargetInput!,
      4: this.buffers.acceptedTargetState!,
      5: this.buffers.phase!,
      6: this.buffers.acceptedPhase!,
      7: this.buffers.amplitude!,
      8: this.buffers.field!,
      10: this.buffers.correction!,
      11: this.buffers.inverseLut!,
      12: this.buffers.decodeLut!,
      13: this.buffers.codes!,
      14: this.buffers.acceptedCodes!,
      15: this.buffers.partialMetrics!,
      16: this.buffers.summary!,
      17: this.buffers.packed!,
      18: this.buffers.supportMask!,
      19: this.buffers.optimizer!,
      20: this.buffers.bestPhase!,
      21: this.buffers.bestCodes!,
      22: this.buffers.bestTargetState!,
    };
    for (const name of names) {
      const pipeline = this.pipelines.get(name)!;
      this.bindGroups.set(name, this.device.createBindGroup({
        label: `WGS ${name} bindings`,
        layout: pipeline.getBindGroupLayout(0),
        entries: WEBGPU_WGS_PIPELINE_BINDINGS[name].map((binding) => ({
          binding,
          resource: { buffer: bufferByBinding[binding]! },
        })),
      }));
    }
    await this.initializeFftPipelines();
  }

  private async initializeFftPipelines(): Promise<void> {
    const parameterAlignment = this.device.limits.minUniformBufferOffsetAlignment;
    const records: Array<{ key: string; values: number[] }> = [];
    const addRecord = (key: string, stage: number, inverse: number, axis: number): void => {
      records.push({ key, values: [this.width, this.height, stage, inverse, axis, 0, 0, 0] });
    };
    addRecord("reverse:h", this.logWidth, 0, 0);
    addRecord("reverse:v", this.logHeight, 0, 1);
    for (const inverse of [0, 1]) {
      for (let stage = 1; stage <= this.logWidth; stage += 1) addRecord(`butterfly:${inverse}:h:${stage}`, stage, inverse, 0);
      for (let stage = 1; stage <= this.logHeight; stage += 1) addRecord(`butterfly:${inverse}:v:${stage}`, stage, inverse, 1);
    }
    const stride = alignTo(32, parameterAlignment);
    const values = new Uint8Array(stride * records.length);
    records.forEach((record, index) => {
      const offset = index * stride;
      this.fftParameterOffsets.set(record.key, offset);
      const view = new DataView(values.buffer, offset, 32);
      record.values.forEach((value, valueIndex) => view.setUint32(valueIndex * 4, value, true));
    });
    this.fftParameterBuffer = this.device.createBuffer({
      label: "FFT stage parameters",
      size: values.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.fftParameterBuffer, 0, values);
    const layout = this.device.createBindGroupLayout({
      label: "FFT bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 } },
      ],
    });
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const module = this.device.createShaderModule({ label: "Radix-2 FFT", code: WEBGPU_RADIX2_FFT_SHADER });
    [this.bitReversePipeline, this.fftPipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: "FFT bit reversal",
        layout: pipelineLayout,
        compute: { module, entryPoint: "bit_reverse" },
      }),
      this.device.createComputePipelineAsync({
        label: "FFT butterfly",
        layout: pipelineLayout,
        compute: { module, entryPoint: "butterfly" },
      }),
    ]);
    const group = (buffer: GPUBuffer, label: string): GPUBindGroup => this.device.createBindGroup({
      label,
      layout,
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.fftParameterBuffer, size: 32 } },
      ],
    });
    this.fftBindGroup = group(this.buffers.field!, "FFT field bindings");
  }

  private dispatch(
    encoder: GPUCommandEncoder,
    name: GeneralPipelineName,
    invocationCount: number,
    workgroupSize = WORKGROUP_SIZE,
  ): void {
    if (invocationCount <= 0) return;
    const pass = encoder.beginComputePass({ label: name });
    pass.setPipeline(this.pipelines.get(name)!);
    pass.setBindGroup(0, this.bindGroups.get(name)!);
    pass.dispatchWorkgroups(Math.ceil(invocationCount / workgroupSize));
    pass.end();
  }

  private encodeFft(encoder: GPUCommandEncoder, bindGroup: GPUBindGroup, inverse: boolean): void {
    const pass = encoder.beginComputePass({ label: inverse ? "Inverse 2D FFT" : "Forward 2D FFT" });
    pass.setPipeline(this.bitReversePipeline);
    pass.setBindGroup(0, bindGroup, [this.fftParameterOffsets.get("reverse:h")!]);
    pass.dispatchWorkgroups(Math.ceil(this.pixelCount / WORKGROUP_SIZE));
    pass.setPipeline(this.fftPipeline);
    for (let stage = 1; stage <= this.logWidth; stage += 1) {
      pass.setBindGroup(0, bindGroup, [this.fftParameterOffsets.get(`butterfly:${inverse ? 1 : 0}:h:${stage}`)!]);
      pass.dispatchWorkgroups(Math.ceil((this.pixelCount / 2) / WORKGROUP_SIZE));
    }
    pass.setPipeline(this.bitReversePipeline);
    pass.setBindGroup(0, bindGroup, [this.fftParameterOffsets.get("reverse:v")!]);
    pass.dispatchWorkgroups(Math.ceil(this.pixelCount / WORKGROUP_SIZE));
    pass.setPipeline(this.fftPipeline);
    for (let stage = 1; stage <= this.logHeight; stage += 1) {
      pass.setBindGroup(0, bindGroup, [this.fftParameterOffsets.get(`butterfly:${inverse ? 1 : 0}:v:${stage}`)!]);
      pass.dispatchWorkgroups(Math.ceil((this.pixelCount / 2) / WORKGROUP_SIZE));
    }
    pass.end();
  }

  private serializeFrameParameters(targetCount: number): ArrayBuffer {
    const output = new ArrayBuffer(96);
    const view = new DataView(output);
    const { activeWidth, activeHeight } = this.calibration.manifest;
    const xStart = Math.floor((this.width - activeWidth) / 2);
    const yStart = Math.floor((this.height - activeHeight) / 2);
    const phaseMode = {
      REFERENCE_WGS: 0,
      PHASE_LOCKED_WGS: 1,
      SOFT_PHASE_LOCKED_WGS: 2,
      PHASE_INTERPOLATED_WGS: 3,
    }[this.config.targetPhaseMode];
    const integers = [
      this.width,
      this.height,
      targetCount,
      this.acceptedTargetCount,
      activeWidth,
      activeHeight,
      xStart,
      yStart,
      this.accepted ? 1 : 0,
      this.config.format === "UINT8" ? 8 : 16,
      phaseMode,
      this.config.backgroundPolicy === "PRESERVE" ? 1 : 0,
    ];
    integers.forEach((value, index) => view.setUint32(index * 4, value, true));
    view.setFloat32(48, this.config.gamma, true);
    view.setFloat32(52, this.config.epsilon, true);
    view.setFloat32(56, this.config.minWeight, true);
    view.setFloat32(60, this.config.maxWeight, true);
    view.setFloat32(64, this.config.convergenceTolerance, true);
    view.setUint32(68, INVERSE_LUT_SIZE, true);
    view.setUint32(72, this.config.format === "UINT8" ? 256 : 65536, true);
    view.setUint32(76, this.partialCount, true);
    view.setUint32(80, this.config.deterministicSeed >>> 0, true);
    return output;
  }

  private mapCoordinate(trap: TrapState): { x: number; y: number } {
    return mapPhysicalPointToDftFrequency(trap, this.calibration, this.width, this.height);
  }

  private evaluateMetrics(
    frame: TrapFrame,
    targetStates: ReturnType<typeof parseTargetStates>,
    totalPower: number,
    maximumGhostIntensity: number,
    displayCodeChange: number,
    iterations: number,
    started: number,
  ): FrameMetrics {
    const intensities = targetStates.intensities;
    const mean = meanOf(intensities);
    const std = standardDeviation(intensities, mean);
    const targetPower = intensities.reduce((sum, value) => sum + value, 0);
    const desired = frame.traps.map((trap) => Math.sqrt(Math.max(0, trap.intensity)));
    const amplitudes = intensities.map((value) => Math.sqrt(Math.max(0, value)));
    const scale = fitAmplitudeScale(amplitudes, desired, this.config.epsilon);
    const relativeError = maximumRelativeAmplitudeError(amplitudes, desired, scale, this.config.epsilon);
    const phaseError = maximumTargetPhaseError(frame, targetStates.measuredPhases, targetStates.targetPhases, this.config.targetPhaseMode);
    const converged = relativeError <= this.config.convergenceTolerance && phaseError <= phaseConvergenceTolerance(this.config);
    const phaseChange = this.accepted
      ? maximumMapPhaseChange(this.accepted.measuredPhases, targetStates.measuredPhases)
      : 0;
    const transitionMinimum = estimateTransitionMinimumIntensity(
      targetStates.measuredIntensities,
      this.accepted?.measuredIntensities,
      displayCodeChange,
      this.config.format,
    );
    const flags: string[] = [];
    if (!converged) flags.push("NOT_CONVERGED");
    if (frame.traps.some((trap) => trap.intensity > this.config.epsilon) && (mean <= this.config.epsilon || targetPower <= this.config.epsilon)) {
      flags.push("ZERO_TARGET_OUTPUT");
    }
    const numericalValues = [
      relativeError,
      totalPower,
      maximumGhostIntensity,
      mean,
      std,
      phaseError,
      phaseChange,
      displayCodeChange,
      ...targetStates.weights,
    ];
    if (!numericalValues.every(Number.isFinite)) flags.push("NUMERIC_ERROR");
    const metrics: FrameMetrics = {
      frameIndex: frame.frameIndex,
      timeUs: frame.timeUs,
      iterations,
      converged,
      maximumRelativeAmplitudeError: relativeError,
      amplitudeConvergenceTolerance: this.config.convergenceTolerance,
      phaseConvergenceToleranceRad: phaseConvergenceTolerance(this.config),
      targetIntensityMean: mean,
      targetIntensityStd: std,
      targetIntensityCoefficientOfVariation: mean > 0 ? std / mean : 0,
      minimumToMeanIntensityRatio: mean > 0 && intensities.length > 0 ? Math.min(...intensities) / mean : 1,
      diffractionEfficiency: totalPower > 0 ? targetPower / totalPower : 0,
      maximumGhostIntensity,
      maximumWgsWeight: Math.max(0, ...targetStates.weights),
      maximumTargetPhaseErrorRad: phaseError,
      targetPhaseChangeRad: phaseChange,
      displayCodeChange,
      estimatedTransitionMinimumIntensity: transitionMinimum,
      solveTimeMs: this.config.measureSolveTime ? nowMs() - started : 0,
      refinementCount: 0,
      numericalValid: !flags.includes("NUMERIC_ERROR"),
      accepted: false,
      flags,
    };
    metrics.accepted = passesQualityGates(metrics, this.config.qualityGates)
      && (!this.config.requireConvergence || metrics.converged);
    return metrics;
  }

  private assertDeviceLimits(): void {
    const largestStorageBuffer = this.pixelCount * 8;
    if (largestStorageBuffer > this.device.limits.maxStorageBufferBindingSize) {
      throw new SlmError("INVALID_ARGUMENT", "The selected FFT grid exceeds this GPU's storage-buffer limit", {
        stage: "SOLVING_SLM_FRAMES",
        details: { requiredBytes: largestStorageBuffer, maximumBytes: this.device.limits.maxStorageBufferBindingSize },
      });
    }
    if (Math.ceil(this.pixelCount / WORKGROUP_SIZE) > this.device.limits.maxComputeWorkgroupsPerDimension) {
      throw new SlmError("INVALID_ARGUMENT", "The selected FFT grid exceeds this GPU's dispatch limit", {
        stage: "SOLVING_SLM_FRAMES",
      });
    }
  }

  private assertUsable(): void {
    if (this.disposed) {
      throw new SlmError("INVALID_ARGUMENT", "The WebGPU solver has been disposed", { stage: "SOLVING_SLM_FRAMES" });
    }
    if (this.deviceLossReason) {
      throw new SlmError("NUMERIC_ERROR", `The WebGPU device was lost: ${this.deviceLossReason}`, {
        stage: "SOLVING_SLM_FRAMES",
        retryable: true,
      });
    }
  }
}

function normalizeConfig(
  calibration: CalibrationPackage,
  config: HologramConfig | Required<HologramConfig>,
): Required<HologramConfig> {
  const targetPhaseMode = config.targetPhaseMode ?? "PHASE_LOCKED_WGS";
  return {
    width: config.width ?? calibration.manifest.fftWidth ?? calibration.manifest.activeWidth,
    height: config.height ?? calibration.manifest.fftHeight ?? calibration.manifest.activeHeight,
    format: config.format ?? "UINT8",
    targetPhaseMode,
    firstFrameIterations: config.firstFrameIterations ?? 12,
    subsequentFrameIterations: config.subsequentFrameIterations ?? 4,
    maxIterations: config.maxIterations ?? 64,
    gamma: config.gamma ?? (targetPhaseMode === "REFERENCE_WGS" ? WGS_REFERENCE_TRAP_AMPLITUDE_GAIN : 0.7),
    epsilon: config.epsilon ?? 1e-8,
    minWeight: config.minWeight ?? 0.1,
    maxWeight: config.maxWeight ?? 10,
    convergenceTolerance: config.convergenceTolerance ?? 1e-4,
    backgroundPolicy: config.backgroundPolicy ?? "PRESERVE",
    oversampling: config.oversampling ?? 1,
    qualityGates: config.qualityGates ?? {},
    maxInsertedFrames: config.maxInsertedFrames ?? 32,
    deterministicSeed: config.deterministicSeed ?? 1,
    measureSolveTime: config.measureSolveTime ?? false,
    requireConvergence: config.requireConvergence ?? false,
  };
}

function assertGpuDimensions(calibration: CalibrationPackage, width: number, height: number): void {
  if (!isPowerOfTwo(width) || !isPowerOfTwo(height)) {
    throw new SlmError("INVALID_ARGUMENT", "WebGPU WGS requires power-of-two FFT dimensions", {
      stage: "SOLVING_SLM_FRAMES",
      details: { width, height },
    });
  }
  if (calibration.manifest.activeWidth > width || calibration.manifest.activeHeight > height) {
    throw new SlmError("INVALID_ARGUMENT", "The active SLM area must fit inside the WebGPU FFT grid", {
      stage: "SOLVING_SLM_FRAMES",
    });
  }
}

function validateFrame(frame: TrapFrame): void {
  if (frame.traps.some((trap) => !Number.isFinite(trap.xUm) || !Number.isFinite(trap.yUm) || !Number.isFinite(trap.intensity) || !Number.isFinite(trap.targetPhaseRad))) {
    throw new SlmError("NUMERIC_ERROR", "Trap frame contains a non-finite value", { stage: "SOLVING_SLM_FRAMES" });
  }
}

function serializeTargets(frame: TrapFrame, mapped: readonly { x: number; y: number }[]): ArrayBuffer {
  const output = new ArrayBuffer(frame.traps.length * TARGET_STRIDE);
  const view = new DataView(output);
  frame.traps.forEach((trap, index) => {
    const offset = index * TARGET_STRIDE;
    view.setFloat32(offset, mapped[index]!.x, true);
    view.setFloat32(offset + 4, mapped[index]!.y, true);
    view.setFloat32(offset + 8, Math.sqrt(Math.max(0, trap.intensity)), true);
    view.setFloat32(offset + 12, trap.targetPhaseRad, true);
    view.setUint32(offset + 16, trap.trapId >>> 0, true);
  });
  return output;
}

function parseTargetStates(snapshot: ArrayBuffer, offset: number, frame: TrapFrame): {
  intensities: number[];
  weights: number[];
  targetPhases: Map<number, number>;
  measuredPhases: Map<number, number>;
  measuredIntensities: Map<number, number>;
} {
  const view = new DataView(snapshot);
  const intensities: number[] = [];
  const weights: number[] = [];
  const targetPhases = new Map<number, number>();
  const measuredPhases = new Map<number, number>();
  const measuredIntensities = new Map<number, number>();
  frame.traps.forEach((trap, index) => {
    const base = offset + index * TARGET_STRIDE;
    const weight = view.getFloat32(base, true);
    const targetPhase = view.getFloat32(base + 4, true);
    const measuredReal = view.getFloat32(base + 8, true);
    const measuredImaginary = view.getFloat32(base + 12, true);
    const intensity = view.getFloat32(base + 16, true);
    weights.push(weight);
    intensities.push(intensity);
    targetPhases.set(trap.trapId, targetPhase);
    measuredPhases.set(trap.trapId, Math.atan2(measuredImaginary, measuredReal));
    measuredIntensities.set(trap.trapId, intensity);
  });
  return { intensities, weights, targetPhases, measuredPhases, measuredIntensities };
}

function unpackPixels(
  packed: ArrayBuffer,
  pixelCount: number,
  format: "UINT8" | "UINT16",
): Uint8Array | Uint16Array {
  const words = new Uint32Array(packed);
  if (format === "UINT8") {
    const output = new Uint8Array(pixelCount);
    for (let index = 0; index < pixelCount; index += 1) {
      output[index] = (words[index >>> 2]! >>> ((index & 3) * 8)) & 0xff;
    }
    return output;
  }
  const output = new Uint16Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    output[index] = (words[index >>> 1]! >>> ((index & 1) * 16)) & 0xffff;
  }
  return output;
}

function expandCalibration(calibration: CalibrationPackage, config: Required<HologramConfig>): ExpandedCalibration {
  const { width, height } = config;
  const { activeWidth, activeHeight } = calibration.manifest;
  const pixelCount = width * height;
  const activePixelCount = activeWidth * activeHeight;
  const xStart = Math.floor((width - activeWidth) / 2);
  const yStart = Math.floor((height - activeHeight) / 2);
  const amplitude = new Float32Array(pixelCount);
  const correction = new Float32Array(pixelCount);
  const signs = calibration.phaseSigns;
  for (let index = 0; index < pixelCount; index += 1) {
    const x = index % width;
    const y = Math.floor(index / width);
    const incident = amplitudeValue(calibration.incidentAmplitude, index, x, y, 1, width, height, activeWidth, activeHeight, activePixelCount, xStart, yStart);
    const aperture = amplitudeValue(calibration.apertureMask, index, x, y, 1, width, height, activeWidth, activeHeight, activePixelCount, xStart, yStart);
    amplitude[index] = incident * aperture;
    correction[index] =
      (signs?.aberration ?? 1) * mapValue(calibration.aberrationPhase, index, x, y, 0, pixelCount, activePixelCount, activeWidth, activeHeight, xStart, yStart)
      + (signs?.grating ?? 1) * mapValue(calibration.carrierGrating, index, x, y, 0, pixelCount, activePixelCount, activeWidth, activeHeight, xStart, yStart)
      + (signs?.lens ?? 1) * mapValue(calibration.digitalLens, index, x, y, 0, pixelCount, activePixelCount, activeWidth, activeHeight, xStart, yStart);
  }
  const maxCode = config.format === "UINT8" ? 255 : 65535;
  const inverseLut = new Float32Array(INVERSE_LUT_SIZE);
  for (let index = 0; index < inverseLut.length; index += 1) {
    const phase = index / (inverseLut.length - 1) * TAU - Math.PI;
    inverseLut[index] = clamp(inverseCode(phase, calibration, maxCode), 0, maxCode);
  }
  const decodeLut = new Float32Array(maxCode + 1);
  for (let code = 0; code <= maxCode; code += 1) {
    decodeLut[code] = decodeCode(code, calibration, maxCode);
  }
  return { amplitude, correction, inverseLut, decodeLut };
}

function amplitudeValue(
  value: ArrayLike<number> | undefined,
  index: number,
  x: number,
  y: number,
  fallback: number,
  width: number,
  height: number,
  activeWidth: number,
  activeHeight: number,
  activePixelCount: number,
  xStart: number,
  yStart: number,
): number {
  const active = x >= xStart && y >= yStart && x < xStart + activeWidth && y < yStart + activeHeight;
  if (value === undefined) return active ? fallback : 0;
  if (value.length === width * height) return value[index] ?? fallback;
  if (value.length === activePixelCount) {
    if (!active) return 0;
    return value[(y - yStart) * activeWidth + (x - xStart)] ?? fallback;
  }
  return active ? fallback : 0;
}

function mapValue(
  value: ArrayLike<number> | number | undefined,
  index: number,
  x: number,
  y: number,
  fallback: number,
  pixelCount: number,
  activePixelCount: number,
  activeWidth: number,
  activeHeight: number,
  xStart: number,
  yStart: number,
): number {
  if (value === undefined) return fallback;
  if (typeof value === "number") return value;
  if (value.length === pixelCount) return value[index] ?? fallback;
  if (value.length === activePixelCount) {
    if (x < xStart || y < yStart || x >= xStart + activeWidth || y >= yStart + activeHeight) return fallback;
    return value[(y - yStart) * activeWidth + (x - xStart)] ?? fallback;
  }
  return fallback;
}

function inverseCode(phase: number, calibration: CalibrationPackage, maxCode: number): number {
  const inverse = calibration.inversePhaseLut;
  if (inverse && inverse.length > 1) {
    const position = ((phase + Math.PI) / TAU) * (inverse.length - 1);
    const low = Math.floor(position);
    const high = Math.min(inverse.length - 1, low + 1);
    const fraction = position - low;
    return (inverse[low] ?? 0) * (1 - fraction) + (inverse[high] ?? 0) * fraction;
  }
  const response = calibration.phaseResponseLut;
  if (response && response.length > 1) {
    const phaseRange = calibration.manifest.phaseConvention === "ZERO_TO_TWO_PI"
      || (calibration.manifest.phaseConvention === undefined && response[0]! >= -1e-9 && response[response.length - 1]! > Math.PI)
      ? (phase < 0 ? phase + TAU : phase)
      : phase;
    const index = nearestMonotonicIndex(response, phaseRange);
    return index / Math.max(1, response.length - 1) * maxCode;
  }
  return ((phase + Math.PI) / TAU) * maxCode;
}

function decodeCode(code: number, calibration: CalibrationPackage, maxCode: number): number {
  const response = calibration.phaseResponseLut;
  if (response && response.length > 1) {
    const position = code / maxCode * (response.length - 1);
    const low = Math.floor(position);
    const high = Math.min(response.length - 1, low + 1);
    const fraction = position - low;
    return wrapPhase((response[low] ?? 0) * (1 - fraction) + (response[high] ?? 0) * fraction);
  }
  const inverse = calibration.inversePhaseLut;
  if (inverse && inverse.length > 1) {
    const index = nearestMonotonicIndex(inverse, code);
    return wrapPhase(index / (inverse.length - 1) * TAU - Math.PI);
  }
  return wrapPhase(code / maxCode * TAU - Math.PI);
}

function nearestMonotonicIndex(values: ArrayLike<number>, target: number): number {
  const increasing = values[0]! <= values[values.length - 1]!;
  let low = 0;
  let high = values.length - 1;
  while (high - low > 1) {
    const middle = (low + high) >>> 1;
    if (increasing ? values[middle]! < target : values[middle]! > target) low = middle;
    else high = middle;
  }
  return Math.abs(values[low]! - target) <= Math.abs(values[high]! - target) ? low : high;
}

function maximumTargetPhaseError(
  frame: TrapFrame,
  measured: Map<number, number>,
  targets: Map<number, number>,
  mode: HologramConfig["targetPhaseMode"],
): number {
  if (frame.traps.length === 0 || mode === "REFERENCE_WGS") return 0;
  return Math.max(...frame.traps.map((trap) => angularDistance(
    measured.get(trap.trapId) ?? 0,
    targets.get(trap.trapId) ?? trap.targetPhaseRad,
  )));
}

function maximumMapPhaseChange(previous: Map<number, number>, current: Map<number, number>): number {
  let maximum = 0;
  for (const [id, phase] of current) {
    const old = previous.get(id);
    if (old !== undefined) maximum = Math.max(maximum, angularDistance(old, phase));
  }
  return maximum;
}

function estimateTransitionMinimumIntensity(
  current: Map<number, number>,
  previous: Map<number, number> | undefined,
  codeChange: number,
  format: "UINT8" | "UINT16",
): number {
  const currentValues = [...current.values()];
  const currentMean = meanOf(currentValues);
  const currentMinimum = currentMean > 0 && currentValues.length > 0 ? Math.min(...currentValues) / currentMean : currentValues.length === 0 ? 1 : 0;
  if (!previous) return clamp(currentMinimum, 0, 1);
  const previousValues = [...previous.values()];
  const previousMean = meanOf(previousValues);
  const previousMinimum = previousMean > 0 && previousValues.length > 0 ? Math.min(...previousValues) / previousMean : 1;
  const maxCode = format === "UINT8" ? 255 : 65535;
  const codeFactor = 1 - clamp(codeChange / Math.max(1, maxCode), 0, 1);
  return clamp(Math.min(currentMinimum, previousMinimum, currentMinimum * codeFactor), 0, 1);
}

function fitAmplitudeScale(amplitudes: number[], desired: number[], epsilon: number): number {
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < desired.length; index += 1) {
    numerator += desired[index]! * amplitudes[index]!;
    denominator += desired[index]! * desired[index]!;
  }
  return denominator > epsilon ? numerator / denominator : 1;
}

function maximumRelativeAmplitudeError(amplitudes: number[], desired: number[], scale: number, epsilon: number): number {
  if (desired.length === 0) return 0;
  return Math.max(...desired.map((value, index) => Math.abs(amplitudes[index]! - scale * value) / (Math.abs(scale * value) + epsilon)));
}

function phaseConvergenceTolerance(config: Required<HologramConfig>): number {
  const codeCount = config.format === "UINT8" ? 256 : 65536;
  return Math.max(config.convergenceTolerance, 1e-3, TAU / codeCount * 1.5);
}

function passesQualityGates(metrics: FrameMetrics, gates: HologramConfig["qualityGates"]): boolean {
  if (!metrics.numericalValid || metrics.flags.includes("ZERO_TARGET_OUTPUT")) return false;
  if (gates?.maxIntensityCoefficientOfVariation !== undefined && metrics.targetIntensityCoefficientOfVariation > gates.maxIntensityCoefficientOfVariation) return false;
  if (gates?.minIntensityToMeanRatio !== undefined && metrics.minimumToMeanIntensityRatio < gates.minIntensityToMeanRatio) return false;
  if (gates?.minDiffractionEfficiency !== undefined && metrics.diffractionEfficiency < gates.minDiffractionEfficiency) return false;
  if (gates?.maxGhostIntensity !== undefined && metrics.maximumGhostIntensity > gates.maxGhostIntensity) return false;
  if (gates?.maxTargetPhaseErrorRad !== undefined && metrics.maximumTargetPhaseErrorRad > gates.maxTargetPhaseErrorRad) return false;
  if (gates?.maxPhaseChangeRad !== undefined && metrics.targetPhaseChangeRad > gates.maxPhaseChangeRad) return false;
  if (gates?.maxDisplayCodeChange !== undefined && metrics.displayCodeChange > gates.maxDisplayCodeChange) return false;
  return true;
}

function meanOf(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[], mean: number): number {
  return values.length === 0 ? 0 : Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

function isPowerOfTwo(value: number): boolean {
  return Number.isInteger(value) && value > 0 && (value & (value - 1)) === 0;
}

function alignTo(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

export const WEBGPU_RADIX2_FFT_SHADER = /* wgsl */ `
struct FftParameters {
  width: u32,
  height: u32,
  stage: u32,
  inverse: u32,
  axis: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
}

@group(0) @binding(0) var<storage, read_write> values: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: FftParameters;

fn reverse_bits(value: u32, count: u32) -> u32 {
  var source = value;
  var result = 0u;
  for (var bit = 0u; bit < count; bit = bit + 1u) {
    result = (result << 1u) | (source & 1u);
    source = source >> 1u;
  }
  return result;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn bit_reverse(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.width * params.height;
  let index = gid.x;
  if (index >= total) { return; }
  if (params.axis == 0u) {
    let x = index % params.width;
    let y = index / params.width;
    let reversed = reverse_bits(x, params.stage);
    if (reversed > x) {
      let other = y * params.width + reversed;
      let temporary = values[index];
      values[index] = values[other];
      values[other] = temporary;
    }
  } else {
    let x = index % params.width;
    let y = index / params.width;
    let reversed = reverse_bits(y, params.stage);
    if (reversed > y) {
      let other = reversed * params.width + x;
      let temporary = values[index];
      values[index] = values[other];
      values[other] = temporary;
    }
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pairCount = params.width * params.height / 2u;
  let pair = gid.x;
  if (pair >= pairCount) { return; }
  let span = 1u << params.stage;
  let halfSpan = span >> 1u;
  var first = 0u;
  var second = 0u;
  var offset = 0u;
  if (params.axis == 0u) {
    let row = pair / (params.width / 2u);
    let withinRow = pair % (params.width / 2u);
    let block = withinRow / halfSpan;
    offset = withinRow % halfSpan;
    first = row * params.width + block * span + offset;
    second = first + halfSpan;
  } else {
    let column = pair % params.width;
    let withinColumn = pair / params.width;
    let block = withinColumn / halfSpan;
    offset = withinColumn % halfSpan;
    first = (block * span + offset) * params.width + column;
    second = first + halfSpan * params.width;
  }
  let direction = select(-1.0, 1.0, params.inverse == 1u);
  let angle = direction * 6.283185307179586 * f32(offset) / f32(span);
  let twiddle = vec2<f32>(cos(angle), sin(angle));
  let even = values[first];
  let oddSource = values[second];
  let odd = vec2<f32>(
    oddSource.x * twiddle.x - oddSource.y * twiddle.y,
    oddSource.x * twiddle.y + oddSource.y * twiddle.x,
  );
  var outputA = even + odd;
  var outputB = even - odd;
  if (params.inverse == 1u && params.axis == 1u && span == params.height) {
    let scale = 1.0 / f32(params.width * params.height);
    outputA = outputA * scale;
    outputB = outputB * scale;
  }
  values[first] = outputA;
  values[second] = outputB;
}
`;

const WGS_SHADER = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;

struct FrameParameters {
  width: u32,
  height: u32,
  targetCount: u32,
  acceptedTargetCount: u32,
  activeWidth: u32,
  activeHeight: u32,
  xStart: u32,
  yStart: u32,
  hasAccepted: u32,
  formatBits: u32,
  phaseMode: u32,
  backgroundPreserve: u32,
  gamma: f32,
  epsilon: f32,
  minWeight: f32,
  maxWeight: f32,
  convergenceTolerance: f32,
  inverseLutSize: u32,
  decodeLutSize: u32,
  partialCount: u32,
  deterministicSeed: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
}

struct TargetInput {
  position: vec2<f32>,
  desired: f32,
  inputPhase: f32,
  id: u32,
  padding0: u32,
  padding1: u32,
  padding2: u32,
}

struct TargetState {
  weight: f32,
  targetPhase: f32,
  measured: vec2<f32>,
  intensity: f32,
  synthesisPhase: f32,
  padding1: f32,
  padding2: f32,
}

struct OptimizerState {
  currentScore: f32,
  bestScore: f32,
  currentRelativeError: f32,
  currentPhaseError: f32,
  saveCandidate: u32,
  performedIterations: u32,
  phaseCorrectionApplied: u32,
  updateActive: u32,
}

@group(0) @binding(0) var<uniform> params: FrameParameters;
@group(0) @binding(1) var<storage, read> targetInputs: array<TargetInput>;
@group(0) @binding(2) var<storage, read_write> targetStates: array<TargetState>;
@group(0) @binding(3) var<storage, read> acceptedTargetInputs: array<TargetInput>;
@group(0) @binding(4) var<storage, read> acceptedTargetStates: array<TargetState>;
@group(0) @binding(5) var<storage, read_write> phase: array<f32>;
@group(0) @binding(6) var<storage, read> acceptedPhase: array<f32>;
@group(0) @binding(7) var<storage, read> amplitude: array<f32>;
@group(0) @binding(8) var<storage, read_write> field: array<vec2<f32>>;
@group(0) @binding(10) var<storage, read> correction: array<f32>;
@group(0) @binding(11) var<storage, read> inverseLut: array<f32>;
@group(0) @binding(12) var<storage, read> decodeLut: array<f32>;
@group(0) @binding(13) var<storage, read_write> codes: array<u32>;
@group(0) @binding(14) var<storage, read> acceptedCodes: array<u32>;
@group(0) @binding(15) var<storage, read_write> partialMetrics: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> metricSummary: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read_write> packedOutput: array<u32>;
@group(0) @binding(18) var<storage, read_write> supportMask: array<u32>;
@group(0) @binding(19) var<storage, read_write> optimizer: OptimizerState;
@group(0) @binding(20) var<storage, read_write> bestPhase: array<f32>;
@group(0) @binding(21) var<storage, read_write> bestCodes: array<u32>;
@group(0) @binding(22) var<storage, read_write> bestTargetStates: array<TargetState>;

var<workgroup> reductionSum: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> reductionGhost: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> reductionCode: array<f32, ${WORKGROUP_SIZE}>;
var<workgroup> targetReductionReal: array<f32, ${EXACT_TARGET_WORKGROUP_SIZE}>;
var<workgroup> targetReductionImag: array<f32, ${EXACT_TARGET_WORKGROUP_SIZE}>;

fn wrap_phase(value: f32) -> f32 {
  let shifted = value + PI;
  return shifted - floor(shifted / TAU) * TAU - PI;
}

fn wrapped_dft_cycles(position: f32, coordinate: u32, extent: u32) -> f32 {
  let integral = i32(floor(position));
  let signedExtent = i32(extent);
  let product = integral * i32(coordinate);
  let modular = ((product % signedExtent) + signedExtent) % signedExtent;
  let fractional = position - f32(integral);
  return (f32(modular) + fractional * f32(coordinate)) / f32(extent);
}

fn deterministic_phase(key: u32) -> f32 {
  var value = (key ^ params.deterministicSeed) + 1u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  value = value ^ (value >> 16u);
  return f32(value >> 8u) / 16777216.0 * TAU - PI;
}

@compute @workgroup_size(${TARGET_WORKGROUP_SIZE})
fn initialize_targets(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.targetCount) { return; }
  var initialWeight = 1.0;
  var persistentPhase = targetInputs[index].inputPhase;
  if (params.phaseMode == 0u) {
    persistentPhase = deterministic_phase(targetInputs[index].id);
  }
  var synthesisPhase = persistentPhase;
  if (params.hasAccepted == 1u) {
    for (var previous = 0u; previous < params.acceptedTargetCount; previous = previous + 1u) {
      if (acceptedTargetInputs[previous].id == targetInputs[index].id) {
        initialWeight = acceptedTargetStates[previous].weight;
        let previousTargetPhase = acceptedTargetStates[previous].targetPhase;
        if (params.phaseMode != 3u) {
          persistentPhase = previousTargetPhase;
        }
        synthesisPhase = wrap_phase(
          acceptedTargetStates[previous].synthesisPhase
            + wrap_phase(persistentPhase - previousTargetPhase),
        );
        break;
      }
    }
  }
  targetStates[index].weight = initialWeight;
  targetStates[index].targetPhase = persistentPhase;
  targetStates[index].synthesisPhase = synthesisPhase;
  targetStates[index].measured = vec2<f32>(0.0);
  targetStates[index].intensity = 0.0;
}

@compute @workgroup_size(1)
fn initialize_optimizer(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  optimizer.currentScore = 3.402823466e+38;
  optimizer.bestScore = -1.0;
  optimizer.currentRelativeError = 3.402823466e+38;
  optimizer.currentPhaseError = 3.402823466e+38;
  optimizer.saveCandidate = 0u;
  optimizer.performedIterations = 0u;
  optimizer.phaseCorrectionApplied = 0u;
  optimizer.updateActive = 0u;
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn initialize_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  let total = params.width * params.height;
  if (index >= total) { return; }
  if (params.hasAccepted == 1u) {
    phase[index] = acceptedPhase[index];
    return;
  }
  var sum = vec2<f32>(0.0);
  var coherentAmplitude = 0.0;
  for (var targetIndex = 0u; targetIndex < params.targetCount; targetIndex = targetIndex + 1u) {
    let item = targetInputs[targetIndex];
    let cycles = wrapped_dft_cycles(item.position.x, index % params.width, params.width)
      + wrapped_dft_cycles(item.position.y, index / params.width, params.height);
    let angle = wrap_phase(TAU * cycles + targetStates[targetIndex].synthesisPhase);
    sum = sum + item.desired * vec2<f32>(cos(angle), sin(angle));
    coherentAmplitude = coherentAmplitude + item.desired;
  }
  let cancellationThreshold = max(params.epsilon, coherentAmplitude * ${WGS_INITIALIZATION_CANCELLATION_RATIO});
  phase[index] = select(atan2(sum.y, sum.x), deterministic_phase(index), length(sum) <= cancellationThreshold);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn make_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.width * params.height) { return; }
  field[index] = amplitude[index] * vec2<f32>(cos(phase[index]), sin(phase[index]));
}

@compute @workgroup_size(${EXACT_TARGET_WORKGROUP_SIZE})
fn sample_targets(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) workgroup: vec3<u32>,
) {
  let targetIndex = workgroup.x;
  if (targetIndex >= params.targetCount) { return; }
  let position = targetInputs[targetIndex].position;
  let total = params.width * params.height;
  var sum = vec2<f32>(0.0);
  var compensation = vec2<f32>(0.0);
  for (var index = lid.x; index < total; index = index + ${EXACT_TARGET_WORKGROUP_SIZE}u) {
    let cycles = wrapped_dft_cycles(position.x, index % params.width, params.width)
      + wrapped_dft_cycles(position.y, index / params.width, params.height);
    let angle = wrap_phase(-TAU * cycles);
    let factor = vec2<f32>(cos(angle), sin(angle));
    let value = field[index];
    let term = vec2<f32>(
      value.x * factor.x - value.y * factor.y,
      value.x * factor.y + value.y * factor.x,
    );
    let corrected = term - compensation;
    let next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
  }
  targetReductionReal[lid.x] = sum.x;
  targetReductionImag[lid.x] = sum.y;
  workgroupBarrier();
  var stride = ${EXACT_TARGET_WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      targetReductionReal[lid.x] = targetReductionReal[lid.x] + targetReductionReal[lid.x + stride];
      targetReductionImag[lid.x] = targetReductionImag[lid.x] + targetReductionImag[lid.x + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (lid.x == 0u) {
    let measured = vec2<f32>(targetReductionReal[0], targetReductionImag[0]);
    targetStates[targetIndex].measured = measured;
    targetStates[targetIndex].intensity = dot(measured, measured);
  }
}

fn phase_convergence_tolerance() -> f32 {
  let codeCount = select(65536.0, 256.0, params.formatBits == 8u);
  return max(max(params.convergenceTolerance, 0.001), TAU / codeCount * 1.5);
}

@compute @workgroup_size(1)
fn evaluate_candidate(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  var numerator = 0.0;
  var denominator = 0.0;
  for (var index = 0u; index < params.targetCount; index = index + 1u) {
    let desired = targetInputs[index].desired;
    let measuredAmplitude = sqrt(max(0.0, targetStates[index].intensity));
    numerator = numerator + desired * measuredAmplitude;
    denominator = denominator + desired * desired;
  }
  let scale = select(1.0, numerator / denominator, denominator > params.epsilon);
  var relativeError = 0.0;
  var phaseError = 0.0;
  for (var index = 0u; index < params.targetCount; index = index + 1u) {
    let desired = targetInputs[index].desired;
    let measuredAmplitude = sqrt(max(0.0, targetStates[index].intensity));
    relativeError = max(
      relativeError,
      abs(measuredAmplitude - scale * desired) / (abs(scale * desired) + params.epsilon),
    );
    if (params.phaseMode != 0u) {
      let measured = targetStates[index].measured;
      let measuredPhase = atan2(measured.y, measured.x);
      phaseError = max(phaseError, abs(wrap_phase(measuredPhase - targetStates[index].targetPhase)));
    }
  }
  var score = relativeError / params.convergenceTolerance;
  if (params.phaseMode != 0u) {
    score = max(score, phaseError / phase_convergence_tolerance());
  }
  optimizer.currentScore = score;
  optimizer.currentRelativeError = relativeError;
  optimizer.currentPhaseError = phaseError;
  optimizer.saveCandidate = select(0u, 1u, optimizer.bestScore < 0.0 || score < optimizer.bestScore);
  if (optimizer.saveCandidate == 1u) {
    optimizer.bestScore = score;
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn save_best_phase_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (optimizer.saveCandidate == 0u || index >= params.width * params.height) { return; }
  bestPhase[index] = phase[index];
  bestCodes[index] = codes[index];
}

@compute @workgroup_size(${TARGET_WORKGROUP_SIZE})
fn save_best_targets(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (optimizer.saveCandidate == 0u || index >= params.targetCount) { return; }
  bestTargetStates[index] = targetStates[index];
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn restore_best_phase_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.width * params.height) { return; }
  phase[index] = bestPhase[index];
  codes[index] = bestCodes[index];
}

@compute @workgroup_size(${TARGET_WORKGROUP_SIZE})
fn restore_best_targets(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.targetCount) { return; }
  targetStates[index] = bestTargetStates[index];
}

@compute @workgroup_size(1)
fn update_controls(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  if (optimizer.currentScore <= 1.0 || params.targetCount == 0u) {
    optimizer.updateActive = 0u;
    return;
  }
  optimizer.updateActive = 1u;
  optimizer.performedIterations = optimizer.performedIterations + 1u;
  var numerator = 0.0;
  var denominator = 0.0;
  for (var index = 0u; index < params.targetCount; index = index + 1u) {
    let desired = targetInputs[index].desired;
    let measuredAmplitude = sqrt(max(0.0, targetStates[index].intensity));
    numerator = numerator + desired * measuredAmplitude;
    denominator = denominator + desired * desired;
  }
  let scale = select(1.0, numerator / denominator, denominator > params.epsilon);
  let amplitudeGain = select(
    min(params.gamma, ${WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN}),
    params.gamma,
    params.phaseMode == 0u,
  );
  var weightSum = 0.0;
  for (var index = 0u; index < params.targetCount; index = index + 1u) {
    let desired = targetInputs[index].desired;
    var weight = targetStates[index].weight;
    if (desired > params.epsilon) {
      let measuredAmplitude = sqrt(max(0.0, targetStates[index].intensity));
      let ratio = scale * desired / (measuredAmplitude + params.epsilon);
      weight = clamp(
        weight * pow(ratio, amplitudeGain),
        params.minWeight,
        params.maxWeight,
      );
    }
    targetStates[index].weight = weight;
    weightSum = weightSum + weight;
  }
  let meanWeight = weightSum / f32(params.targetCount);
  if (meanWeight > 0.0) {
    for (var index = 0u; index < params.targetCount; index = index + 1u) {
      targetStates[index].weight = targetStates[index].weight / meanWeight;
    }
  }

  if (params.phaseMode == 0u) {
    for (var index = 0u; index < params.targetCount; index = index + 1u) {
      let measured = targetStates[index].measured;
      targetStates[index].synthesisPhase = atan2(measured.y, measured.x);
    }
  } else if (optimizer.phaseCorrectionApplied == 0u) {
    if (optimizer.currentPhaseError > phase_convergence_tolerance()) {
      let phaseGain = select(
        ${WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN},
        ${WGS_SOFT_PHASE_PRECOMPENSATION_GAIN},
        params.phaseMode == 2u,
      );
      for (var index = 0u; index < params.targetCount; index = index + 1u) {
        if (targetInputs[index].desired <= params.epsilon) { continue; }
        let measured = targetStates[index].measured;
        let measuredPhase = atan2(measured.y, measured.x);
        let phaseError = wrap_phase(targetStates[index].targetPhase - measuredPhase);
        targetStates[index].synthesisPhase = wrap_phase(
          targetStates[index].synthesisPhase + phaseGain * phaseError,
        );
      }
    }
    optimizer.phaseCorrectionApplied = 1u;
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn synthesize_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (optimizer.updateActive == 0u || index >= params.width * params.height) { return; }
  var sum = vec2<f32>(0.0);
  var compensation = vec2<f32>(0.0);
  for (var targetIndex = 0u; targetIndex < params.targetCount; targetIndex = targetIndex + 1u) {
    let item = targetInputs[targetIndex];
    let targetPhase = targetStates[targetIndex].synthesisPhase;
    let targetAmplitude = targetStates[targetIndex].weight * item.desired;
    let cycles = wrapped_dft_cycles(item.position.x, index % params.width, params.width)
      + wrapped_dft_cycles(item.position.y, index / params.width, params.height);
    let angle = wrap_phase(TAU * cycles + targetPhase);
    let term = targetAmplitude * vec2<f32>(cos(angle), sin(angle));
    let corrected = term - compensation;
    let next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
  }
  if (dot(sum, sum) > params.epsilon * params.epsilon) {
    phase[index] = atan2(sum.y, sum.x);
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn clear_support(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.width * params.height) { return; }
  supportMask[index] = 0u;
}

fn periodic_index(value: i32, extent: u32) -> u32 {
  let signedExtent = i32(extent);
  return u32(((value % signedExtent) + signedExtent) % signedExtent);
}

fn periodic_delta(pixel: f32, targetPosition: f32, extent: f32) -> f32 {
  let direct = abs(pixel - targetPosition);
  return min(direct, extent - direct);
}

fn periodic_position(value: f32, extent: f32) -> f32 {
  return value - floor(value / extent) * extent;
}

@compute @workgroup_size(1)
fn mark_support(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  for (var targetIndex = 0u; targetIndex < params.targetCount; targetIndex = targetIndex + 1u) {
    let signedPosition = targetInputs[targetIndex].position;
    let position = vec2<f32>(
      periodic_position(signedPosition.x, f32(params.width)),
      periodic_position(signedPosition.y, f32(params.height)),
    );
    let x0 = i32(floor(position.x));
    let y0 = i32(floor(position.y));
    for (var deltaY: i32 = -2; deltaY <= 2; deltaY = deltaY + 1) {
      for (var deltaX: i32 = -2; deltaX <= 2; deltaX = deltaX + 1) {
        let pixelX = periodic_index(x0 + deltaX, params.width);
        let pixelY = periodic_index(y0 + deltaY, params.height);
        let distance = vec2<f32>(
          periodic_delta(f32(pixelX), position.x, f32(params.width)),
          periodic_delta(f32(pixelY), position.y, f32(params.height)),
        );
        if (dot(distance, distance) <= 2.25) {
          supportMask[pixelY * params.width + pixelX] = 1u;
        }
      }
    }
  }
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn quantize_codes(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.width * params.height) { return; }
  let displayPhase = wrap_phase(phase[index] + correction[index]);
  let position = (displayPhase + PI) / TAU * f32(params.inverseLutSize - 1u);
  let low = u32(floor(position));
  let high = min(params.inverseLutSize - 1u, low + 1u);
  let code = mix(inverseLut[low], inverseLut[high], position - f32(low));
  let maximumCode = select(65535.0, 255.0, params.formatBits == 8u);
  codes[index] = u32(round(clamp(code, 0.0, maximumCode)));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn make_final_field(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.width * params.height) { return; }
  let decodedPhase = decodeLut[min(codes[index], params.decodeLutSize - 1u)];
  field[index] = amplitude[index] * vec2<f32>(cos(decodedPhase), sin(decodedPhase));
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn reduce_field(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) workgroup: vec3<u32>,
) {
  let index = gid.x;
  let local = lid.x;
  let total = params.width * params.height;
  var power = 0.0;
  var ghost = 0.0;
  var codeDifference = 0.0;
  if (index < total) {
    power = dot(field[index], field[index]);
    if (supportMask[index] == 0u) { ghost = power; }
    if (params.hasAccepted == 1u) {
      codeDifference = abs(f32(codes[index]) - f32(acceptedCodes[index]));
    }
  }
  reductionSum[local] = power;
  reductionGhost[local] = ghost;
  reductionCode[local] = codeDifference;
  workgroupBarrier();
  var stride = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (local < stride) {
      reductionSum[local] = reductionSum[local] + reductionSum[local + stride];
      reductionGhost[local] = max(reductionGhost[local], reductionGhost[local + stride]);
      reductionCode[local] = max(reductionCode[local], reductionCode[local + stride]);
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  if (local == 0u) {
    partialMetrics[workgroup.x] = vec4<f32>(reductionSum[0], reductionGhost[0], reductionCode[0], 0.0);
  }
}

@compute @workgroup_size(1)
fn finish_reduction(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x != 0u) { return; }
  var totalPower = 0.0;
  var maximumGhost = 0.0;
  var maximumCodeDifference = 0.0;
  for (var index = 0u; index < params.partialCount; index = index + 1u) {
    totalPower = totalPower + partialMetrics[index].x;
    maximumGhost = max(maximumGhost, partialMetrics[index].y);
    maximumCodeDifference = max(maximumCodeDifference, partialMetrics[index].z);
  }
  metricSummary[0] = vec4<f32>(totalPower, maximumGhost, maximumCodeDifference, 0.0);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn pack_active(@builtin(global_invocation_id) gid: vec3<u32>) {
  let wordIndex = gid.x;
  let pixelsPerWord = select(2u, 4u, params.formatBits == 8u);
  let activePixels = params.activeWidth * params.activeHeight;
  let firstPixel = wordIndex * pixelsPerWord;
  if (firstPixel >= activePixels) { return; }
  var word = 0u;
  for (var part = 0u; part < pixelsPerWord; part = part + 1u) {
    let activeIndex = firstPixel + part;
    if (activeIndex < activePixels) {
      let activeX = activeIndex % params.activeWidth;
      let activeY = activeIndex / params.activeWidth;
      let source = (params.yStart + activeY) * params.width + params.xStart + activeX;
      word = word | (codes[source] << (part * params.formatBits));
    }
  }
  packedOutput[wordIndex] = word;
}
`;
