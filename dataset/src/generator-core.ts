/** Headless Node/Dawn orchestration around the existing WebGPU WGS solver. */
import type { CalibrationPackage, FrameMetrics, HologramConfig, TrapFrame } from "../../src/types.js";
import { createOpticalCalibration, type GaussianIncidentBeamInput } from "../../web/src/lib/optical-calibration.js";
import { inspectWebGpu, WebGpuSequentialWgsSolver } from "../../web/src/lib/webgpu-wgs.js";
import { encodeSamplePayload } from "./protocol.js";
import {
  deriveSampleSeed,
  generateSampleLayout,
  sampleTrapCount,
  type DatasetSamplingConfig,
  type SampleLayout,
} from "./sampling.js";

interface CollectorProtocol {
  magic: string;
  version: number;
  headerBytes: number;
  littleEndian: boolean;
}

interface MeasuredLutConfig {
  values: number[];
  sourceSha256: string;
  valuesSha256: string;
  filename: string;
  direction: "INCREASING" | "DECREASING";
  phaseConvention: "NEGATIVE_PI_TO_PI" | "ZERO_TO_TWO_PI";
}

interface CollectorConfig {
  configHash: string;
  protocol: CollectorProtocol;
  backend: string;
  requestedSamples: number;
  acceptedSamples: number;
  nextSampleId: number;
  activeWidth: number;
  activeHeight: number;
  fftWidth: number;
  fftHeight: number;
  storageMaxTraps: number;
  dawn: {
    backend: string | null;
    adapter: string | null;
    options: string[];
  };
  output: {
    pixelFormat: "UINT8";
    frameMode: "SLMCONTROL3_LOGICAL" | "DEVICE_READY_LUT_BAKED";
    deviceReady: boolean;
    displayReady: boolean;
    lutApplication: "SLMCONTROL3" | "DAWN_NODE";
    slmControl3LutMustBeEnabled: boolean;
    slmControl3LutMustBeDisabled: boolean;
    frameSemantics: string;
  };
  lut: MeasuredLutConfig | null;
  optics: {
    wavelengthNm: number;
    focalLengthMm: number;
    pixelPitchUm: number;
    incidentBeam?: GaussianIncidentBeamInput;
  };
  solver: {
    targetPhaseMode: string;
    iterations: number;
    maxIterations: number;
    gamma: number;
    epsilon: number;
    minWeight: number;
    maxWeight: number;
    convergenceTolerance: number;
    deterministicSeed: number;
    requireConvergence: boolean;
    format: string;
    backgroundPolicy?: "PRESERVE" | "ZERO";
    qualityGates?: HologramConfig["qualityGates"];
  };
  sampling: {
    minTraps: number;
    maxTraps: number;
    distribution: string;
    datasetSeed: number;
    minSeparationUm: number;
    fieldFillFraction: number;
    xMinUm: number;
    xMaxUm: number;
    yMinUm: number;
    yMaxUm: number;
    zeroOrderGuardUm: number;
    maxAttemptsPerPoint: number;
    maxRetriesPerSample: number;
  };
}

interface ConfigSummary {
  requestedSamples: number;
  acceptedSamples: number;
  nextSampleId: number;
  activeWidth: number;
  activeHeight: number;
  fftWidth: number;
  fftHeight: number;
  minTraps: number;
  maxTraps: number;
  maxRetriesPerSample: number;
  iterations: number;
  convergenceTolerance: number;
  calibration: string;
  frameMode: CollectorConfig["output"]["frameMode"];
  lutEntries: number;
  lutSha256: string | null;
  rejectedSamples: number;
  rejectionCounts: Record<string, number>;
}

interface CollectorStatus {
  ok: true;
  complete: boolean;
  requestedSamples: number;
  acceptedSamples: number;
  remainingSamples: number;
  nextSampleId: number;
  rejectedSamples: number;
  currentSampleRejections: number;
  rejectionsByReason: Record<string, number>;
  completedShards: number;
  partialShard: unknown;
  configurationHash: string;
}

interface ProgressSnapshot {
  acceptedSamples: number;
  requestedSamples: number;
  nextSampleId: number;
  trapCount: number;
  attempt: number;
  sampleElapsedMs: number;
  runElapsedMs: number;
  acceptedPerHour: number;
  rejectedTotal: number;
  rejectionCounts: Record<string, number>;
  metrics?: Record<string, unknown>;
}

let solver: WebGpuSequentialWgsSolver | undefined;
let calibration: CalibrationPackage | undefined;
let runtimeFingerprint = "";
let running = false;
let cancelRequested = false;
let activeFetch: AbortController | undefined;
let lastProgress: ProgressSnapshot | null = null;
const jsonLineMessageSink = (message: unknown): void => {
  process.stdout.write(`${JSON.stringify(message)}\n`);
};
let messageSink: (message: unknown) => void = jsonLineMessageSink;

export async function runDatasetGeneration(collectorUrl: string): Promise<void> {
  if (running) throw new Error("Dawn dataset generation is already running");
  running = true;
  cancelRequested = false;
  const runStarted = performance.now();
  let rejectionCounts: Record<string, number> = {};
  let rejectedTotal = 0;
  let rejectedAtStart = 0;
  let acceptedThisRun = 0;
  let config: CollectorConfig | undefined;
  try {
    status("FETCHING_CONFIG", "读取最新 resume 状态。已存在的样本不会重新计算。");
    const [freshConfig, collectorStatus] = await Promise.all([
      fetchConfig(collectorUrl),
      fetchCollectorStatus(collectorUrl),
    ]);
    config = freshConfig;
    validateConfig(config);
    validateStatus(config, collectorStatus);
    rejectionCounts = { ...collectorStatus.rejectionsByReason };
    rejectedTotal = collectorStatus.rejectedSamples;
    rejectedAtStart = rejectedTotal;
    if (cancelRequested) return cancelled();

    const capability = await inspectWebGpu();
    if (!capability.available) throw new Error(`WebGPU unavailable: ${capability.reason}`);
    status("INITIALIZING_WEBGPU", "创建 calibration 和唯一的长驻 WebGPU solver；pipeline 只编译一次。");
    await ensureRuntime(config);
    if (cancelRequested) return cancelled();

    let acceptedSamples = config.acceptedSamples;
    let nextSampleId = config.nextSampleId;
    let retryOffset = collectorStatus.currentSampleRejections;
    lastProgress = makeProgress(
      config,
      acceptedSamples,
      nextSampleId,
      0,
      0,
      0,
      runStarted,
      acceptedThisRun,
      rejectedTotal,
      rejectionCounts,
    );
    post({ kind: "PROGRESS", progress: lastProgress });

    while (acceptedSamples < config.requestedSamples) {
      if (cancelRequested) return cancelled();
      const sampleStarted = performance.now();
      let fixedTrapCount: number | undefined;
      let accepted = false;
      // The limit is a per-run safety window, not a permanent poison pill.
      // On resume, continue with the next deterministic position seed even if
      // an earlier Node/Dawn run exhausted its window.
      const retryWindowEnd = retryOffset + config.sampling.maxRetriesPerSample;

      for (let retry = retryOffset; retry <= retryWindowEnd; retry += 1) {
        const attempt = retry + 1;
        if (cancelRequested) return cancelled();
        status("SAMPLING", `sample ${nextSampleId}：生成第 ${attempt} 组确定性随机位置。`);
        const samplingConfig = makeSamplingConfig(config);
        let layout: SampleLayout;
        try {
          layout = generateSampleLayout(nextSampleId, samplingConfig, retry);
        } catch (error) {
          const trapCount = sampleTrapCount(nextSampleId, samplingConfig);
          const samplingSeed = deriveSampleSeed(samplingConfig.masterSeed, nextSampleId, retry);
          const detail = error instanceof Error ? error.message : String(error);
          const reason = "SAMPLING_FAILED";
          rejectedTotal += 1;
          rejectionCounts[reason] = (rejectionCounts[reason] ?? 0) + 1;
          await postJson(collectorUrl, "/api/rejection", {
            sample_id: nextSampleId,
            sampling_seed: samplingSeed,
            trap_count: trapCount,
            attempt,
            reason: `${reason}: ${detail}`,
          });
          lastProgress = makeProgress(
            config,
            acceptedSamples,
            nextSampleId,
            trapCount,
            attempt,
            performance.now() - sampleStarted,
            runStarted,
            acceptedThisRun,
            rejectedTotal,
            rejectionCounts,
          );
          post({ kind: "PROGRESS", progress: lastProgress });
          post({ kind: "REJECTION", reason: `${reason}: ${detail}`, attempt, sampleId: nextSampleId, trapCount });
          if (retry >= retryWindowEnd) {
            throw new Error(
              `sample ${nextSampleId} (${trapCount} traps) exceeded `
              + `this run's ${config.sampling.maxRetriesPerSample} retry window; `
              + `restart to continue with the next deterministic positions. Last rejection: ${reason}: ${detail}`,
            );
          }
          continue;
        }
        fixedTrapCount ??= layout.trapCount;
        if (layout.trapCount !== fixedTrapCount) {
          throw new Error(`Sampling retry changed trap count from ${fixedTrapCount} to ${layout.trapCount}`);
        }

        status("SOLVING", `sample ${nextSampleId}：WebGPU WGS 正在求解 ${layout.trapCount} 个光镊。`);
        let solved: Awaited<ReturnType<typeof solveLayout>>;
        try {
          solved = await solveLayout(layout);
        } catch (error) {
          const detail = error instanceof Error ? error.message : String(error);
          // Candidate quality failures are returned as metrics and handled
          // below. An exception here indicates a WebGPU/runtime failure; do
          // not consume a data retry or mislabel it as non-convergence.
          throw new Error(
            `WebGPU solver failed for sample ${nextSampleId} attempt ${attempt}: ${detail}. `
            + "Reload the page to construct a fresh GPU device/solver.",
            { cause: error },
          );
        }
        if (cancelRequested) return cancelled();
        const rejectionReason = validateCandidate(
          solved.metrics,
          solved.measuredPhases,
          solved.frame,
          layout.trapCount,
          config.activeWidth * config.activeHeight,
        );

        if (rejectionReason) {
          rejectedTotal += 1;
          rejectionCounts[rejectionReason] = (rejectionCounts[rejectionReason] ?? 0) + 1;
          await postJson(collectorUrl, "/api/rejection", {
            sample_id: nextSampleId,
            sampling_seed: layout.samplingSeed,
            trap_count: layout.trapCount,
            attempt,
            reason: rejectionReason,
            metrics: solved.metrics,
          });
          lastProgress = makeProgress(
            config,
            acceptedSamples,
            nextSampleId,
            layout.trapCount,
            attempt,
            performance.now() - sampleStarted,
            runStarted,
            acceptedThisRun,
            rejectedTotal,
            rejectionCounts,
            solved.metrics,
          );
          post({ kind: "PROGRESS", progress: lastProgress });
          post({
            kind: "REJECTION",
            reason: rejectionReason,
            attempt,
            sampleId: nextSampleId,
            trapCount: layout.trapCount,
          });
          if (retry >= retryWindowEnd) {
            throw new Error(
              `sample ${nextSampleId} (${layout.trapCount} traps) exceeded `
              + `this run's ${config.sampling.maxRetriesPerSample} retry window; `
              + `restart to continue with the next deterministic positions. Last rejection: ${rejectionReason}`,
            );
          }
          continue;
        }

        status("UPLOADING", `sample ${nextSampleId}：上传 frame、位置、measuredPhases 和 metrics。`);
        const payload = encodeSamplePayload({
          sampleId: nextSampleId,
          samplingSeed: layout.samplingSeed,
          width: config.activeWidth,
          height: config.activeHeight,
          positions: positionsFor(layout),
          measuredPhases: solved.measuredPhases,
          trapIds: Uint32Array.from(layout.traps, (trap) => trap.trapId),
          frame: solved.frame,
          metrics: solved.metrics,
        });
        const response = await postBinary<SampleAcceptedResponse>(collectorUrl, "/api/sample", payload);
        validateAcceptedResponse(response, config, acceptedSamples, nextSampleId);
        acceptedSamples = response.acceptedSamples;
        nextSampleId = response.nextSampleId;
        acceptedThisRun += 1;
        accepted = true;
        retryOffset = 0;
        lastProgress = makeProgress(
          config,
          acceptedSamples,
          nextSampleId,
          layout.trapCount,
          attempt,
          performance.now() - sampleStarted,
          runStarted,
          acceptedThisRun,
          rejectedTotal,
          rejectionCounts,
          solved.metrics,
        );
        post({ kind: "PROGRESS", progress: lastProgress });
        break;
      }

      if (!accepted) throw new Error(`sample ${nextSampleId} stopped without acceptance`);
    }

    if (cancelRequested) return cancelled();
    status("FINALIZING", "请求 collector 封存最后一个 shard 并写入 manifest。");
    const completeResponse = await postJson(collectorUrl, "/api/complete", {
      summary: {
        backend: "DAWN_WEBGPU",
        backendId: solver?.backendId,
        acceptedThisRun,
        rejectedThisRun: rejectedTotal - rejectedAtStart,
        runElapsedMs: performance.now() - runStarted,
        rejectionCounts,
      },
    });
    if (!lastProgress) {
      lastProgress = makeProgress(
        config,
        config.requestedSamples,
        config.nextSampleId,
        0,
        0,
        0,
        runStarted,
        acceptedThisRun,
        rejectedTotal,
        rejectionCounts,
      );
    }
    post({ kind: "COMPLETE", response: completeResponse, progress: lastProgress });
  } finally {
    activeFetch = undefined;
    running = false;
  }
}

async function ensureRuntime(config: CollectorConfig): Promise<void> {
  const fingerprint = JSON.stringify({
    activeWidth: config.activeWidth,
    activeHeight: config.activeHeight,
    fftWidth: config.fftWidth,
    fftHeight: config.fftHeight,
    frameMode: config.output.frameMode,
    lutSha256: config.lut?.valuesSha256 ?? null,
    lut: config.lut?.values ?? null,
    phaseConvention: config.lut?.phaseConvention ?? "NEGATIVE_PI_TO_PI",
    optics: config.optics,
    solver: config.solver,
  });
  if (solver) {
    if (fingerprint !== runtimeFingerprint) {
      throw new Error("Collector calibration/solver config changed after Dawn initialization; restart the Node runner to create a new device safely");
    }
    return;
  }

  const lut = config.lut;
  calibration = createOpticalCalibration({
    activeWidth: config.activeWidth,
    activeHeight: config.activeHeight,
    fftWidth: config.fftWidth,
    fftHeight: config.fftHeight,
  }, {
    wavelengthNm: config.optics.wavelengthNm,
    focalLengthMm: config.optics.focalLengthMm,
    pixelPitchUm: config.optics.pixelPitchUm,
    ...(lut ? { phaseResponseLut: lut.values } : {}),
    ...(config.optics.incidentBeam ? { incidentBeam: config.optics.incidentBeam } : {}),
  }, `dataset-${lut?.valuesSha256 ?? "slmcontrol3-logical"}`);
  // Endpoint-only inference is ambiguous for a valid decreasing 2pi -> 0
  // response. The collector derives this convention from the complete LUT
  // range and persists it with every shard.
  if (lut) calibration.manifest.phaseConvention = lut.phaseConvention;

  const solverConfig: HologramConfig = {
    width: config.fftWidth,
    height: config.fftHeight,
    format: "UINT8",
    targetPhaseMode: "REFERENCE_WGS",
    firstFrameIterations: config.solver.iterations,
    subsequentFrameIterations: config.solver.iterations,
    maxIterations: config.solver.maxIterations,
    gamma: config.solver.gamma,
    epsilon: config.solver.epsilon,
    minWeight: config.solver.minWeight,
    maxWeight: config.solver.maxWeight,
    convergenceTolerance: config.solver.convergenceTolerance,
    deterministicSeed: config.solver.deterministicSeed,
    backgroundPolicy: config.solver.backgroundPolicy ?? "ZERO",
    requireConvergence: true,
    measureSolveTime: true,
    ...(config.solver.qualityGates ? { qualityGates: config.solver.qualityGates } : {}),
  };
  solver = await WebGpuSequentialWgsSolver.create(calibration, solverConfig);
  runtimeFingerprint = fingerprint;
}

function makeSamplingConfig(config: CollectorConfig): DatasetSamplingConfig {
  const distribution = config.sampling.distribution === "uniform" ? "uniform" : "log-uniform";
  return {
    totalSamples: config.requestedSamples,
    masterSeed: config.sampling.datasetSeed,
    minTrapCount: config.sampling.minTraps,
    maxTrapCount: config.sampling.maxTraps,
    distribution,
    bounds: {
      minXUm: config.sampling.xMinUm,
      maxXUm: config.sampling.xMaxUm,
      minYUm: config.sampling.yMinUm,
      maxYUm: config.sampling.yMaxUm,
    },
    minimumSpacingXUm: config.sampling.minSeparationUm,
    minimumSpacingYUm: config.sampling.minSeparationUm,
    maxAttemptsPerPoint: config.sampling.maxAttemptsPerPoint,
    ...(config.sampling.zeroOrderGuardUm > 0 ? {
      zeroOrderExclusion: {
        radiusXUm: config.sampling.zeroOrderGuardUm,
        radiusYUm: config.sampling.zeroOrderGuardUm,
      },
    } : {}),
  };
}

async function solveLayout(layout: SampleLayout): Promise<{
  frame: Uint8Array;
  measuredPhases: Float32Array;
  metrics: FrameMetrics;
}> {
  if (!solver) throw new Error("WebGPU solver was not initialized");
  const frame: TrapFrame = {
    frameIndex: layout.sampleId,
    timeUs: 0,
    traps: layout.traps.map((trap) => ({
      trapId: trap.trapId,
      atomId: null,
      xUm: trap.xUm,
      yUm: trap.yUm,
      intensity: 1,
      targetPhaseRad: 0,
      flags: 0,
    })),
  };
  try {
    const result = await solver.solveSequentialFrame(frame);
    if (!(result.pixels instanceof Uint8Array)) {
      throw new Error("WebGPU solver returned a non-UINT8 frame");
    }
    const measuredPhases = solver.getCandidateMeasuredPhases(frame);
    return {
      frame: result.pixels,
      measuredPhases,
      metrics: result.metrics,
    };
  } finally {
    // Dataset samples are independent. Never commit sequential WGS state.
    solver.rollbackToPreviousAcceptedFrame();
  }
}

function validateCandidate(
  metrics: FrameMetrics,
  measuredPhases: Float32Array,
  frame: Uint8Array,
  trapCount: number,
  expectedFrameBytes: number,
): string | null {
  if (!metrics.numericalValid || metrics.flags.includes("NUMERIC_ERROR")) return "NUMERICAL_INVALID";
  if (!metrics.converged || metrics.flags.includes("NOT_CONVERGED")) return "NOT_CONVERGED";
  if (!metrics.accepted) return "QUALITY_GATES_REJECTED";
  if (metrics.flags.includes("ZERO_TARGET_OUTPUT")) return "ZERO_TARGET_OUTPUT";
  if (!jsonNumbersFinite(metrics)) return "METRICS_NONFINITE";
  if (measuredPhases.length !== trapCount) return "MEASURED_PHASE_COUNT_MISMATCH";
  if (!allFinite(measuredPhases)) return "MEASURED_PHASE_NONFINITE";
  if (frame.length !== expectedFrameBytes) return "FRAME_SIZE_MISMATCH";
  return null;
}

function positionsFor(layout: SampleLayout): Float32Array {
  const result = new Float32Array(layout.trapCount * 2);
  layout.traps.forEach((trap, index) => {
    result[index * 2] = trap.xUm;
    result[index * 2 + 1] = trap.yUm;
  });
  return result;
}

function validateConfig(config: CollectorConfig): void {
  if (!config || typeof config !== "object") throw new Error("Collector returned an invalid config document");
  if (config.protocol?.magic !== "SLMD" || config.protocol.version !== 1 || config.protocol.headerBytes !== 64 || config.protocol.littleEndian !== true) {
    throw new Error("Collector protocol must be little-endian SLMD v1 with a 64-byte header");
  }
  if (config.backend !== "DAWN_WEBGPU") {
    throw new Error(`Collector requested unsupported backend ${String(config.backend)}; only DAWN_WEBGPU is allowed`);
  }
  if (!config.dawn || !Array.isArray(config.dawn.options) || !config.dawn.options.every((value) => typeof value === "string")) {
    throw new Error("Collector Dawn configuration is invalid");
  }
  if (!config.configHash) throw new Error("Collector configHash is required for safe resume");
  safeInteger(config.requestedSamples, "requestedSamples", 0);
  safeInteger(config.acceptedSamples, "acceptedSamples", 0);
  safeInteger(config.nextSampleId, "nextSampleId", 0);
  if (config.requestedSamples > 0xffff_ffff) throw new Error("requestedSamples must fit in uint32 for deterministic sampling");
  if (config.acceptedSamples > config.requestedSamples) throw new Error("acceptedSamples exceeds requestedSamples");
  if (config.nextSampleId !== config.acceptedSamples) throw new Error("Sequential dataset nextSampleId must equal acceptedSamples");
  positiveInteger(config.activeWidth, "activeWidth");
  positiveInteger(config.activeHeight, "activeHeight");
  positiveInteger(config.fftWidth, "fftWidth");
  positiveInteger(config.fftHeight, "fftHeight");
  if (!isPowerOfTwo(config.fftWidth) || !isPowerOfTwo(config.fftHeight)) throw new Error("WebGPU FFT dimensions must be powers of two");
  if (config.fftWidth < config.activeWidth || config.fftHeight < config.activeHeight) throw new Error("FFT grid cannot be smaller than the active SLM frame");
  validateOutputAndLut(config);
  if (config.solver.targetPhaseMode !== "REFERENCE_WGS") throw new Error("Dataset solver targetPhaseMode must be REFERENCE_WGS");
  if (config.solver.format !== "UINT8") throw new Error("Dataset solver format must be UINT8");
  if (config.solver.requireConvergence !== true) throw new Error("Dataset solver requireConvergence must be true");
  positiveInteger(config.solver.iterations, "solver.iterations");
  positiveInteger(config.solver.maxIterations, "solver.maxIterations");
  if (config.solver.iterations > config.solver.maxIterations) throw new Error("solver.iterations exceeds solver.maxIterations");
  positiveFinite(config.solver.gamma, "solver.gamma");
  positiveFinite(config.solver.epsilon, "solver.epsilon");
  positiveFinite(config.solver.minWeight, "solver.minWeight");
  positiveFinite(config.solver.maxWeight, "solver.maxWeight");
  if (config.solver.maxWeight < config.solver.minWeight) throw new Error("solver.maxWeight must be at least solver.minWeight");
  positiveFinite(config.solver.convergenceTolerance, "solver.convergenceTolerance");
  uint32(config.solver.deterministicSeed, "solver.deterministicSeed");
  if (config.solver.backgroundPolicy !== undefined
    && config.solver.backgroundPolicy !== "ZERO"
    && config.solver.backgroundPolicy !== "PRESERVE") {
    throw new Error("solver.backgroundPolicy must be ZERO or PRESERVE");
  }
  positiveInteger(config.sampling.minTraps, "sampling.minTraps");
  positiveInteger(config.sampling.maxTraps, "sampling.maxTraps");
  if (config.sampling.minTraps > config.sampling.maxTraps) throw new Error("sampling minTraps exceeds maxTraps");
  positiveInteger(config.storageMaxTraps, "storageMaxTraps");
  if (config.sampling.maxTraps > 2000 || config.sampling.maxTraps > config.storageMaxTraps) {
    throw new Error("sampling maxTraps exceeds the configured/WebGPU dataset limit");
  }
  if (config.sampling.distribution !== "log-uniform" && config.sampling.distribution !== "uniform") {
    throw new Error("sampling.distribution must be log-uniform or uniform");
  }
  uint32(config.sampling.datasetSeed, "sampling.datasetSeed");
  positiveFinite(config.sampling.minSeparationUm, "sampling.minSeparationUm");
  positiveFinite(config.sampling.fieldFillFraction, "sampling.fieldFillFraction");
  if (config.sampling.fieldFillFraction > 1) throw new Error("sampling.fieldFillFraction must be no greater than 1");
  finite(config.sampling.xMinUm, "sampling.xMinUm");
  finite(config.sampling.xMaxUm, "sampling.xMaxUm");
  finite(config.sampling.yMinUm, "sampling.yMinUm");
  finite(config.sampling.yMaxUm, "sampling.yMaxUm");
  if (config.sampling.xMinUm >= config.sampling.xMaxUm || config.sampling.yMinUm >= config.sampling.yMaxUm) {
    throw new Error("sampling bounds must have strictly increasing minima and maxima");
  }
  nonnegativeFinite(config.sampling.zeroOrderGuardUm, "sampling.zeroOrderGuardUm");
  positiveInteger(config.sampling.maxAttemptsPerPoint, "sampling.maxAttemptsPerPoint");
  safeInteger(config.sampling.maxRetriesPerSample, "sampling.maxRetriesPerSample", 0);
  positiveFinite(config.optics.wavelengthNm, "optics.wavelengthNm");
  positiveFinite(config.optics.focalLengthMm, "optics.focalLengthMm");
  positiveFinite(config.optics.pixelPitchUm, "optics.pixelPitchUm");
}

function validateOutputAndLut(config: CollectorConfig): void {
  const output = config.output;
  if (!output || output.pixelFormat !== "UINT8" || typeof output.frameSemantics !== "string" || !output.frameSemantics) {
    throw new Error("Collector output configuration is invalid");
  }
  if (output.displayReady !== true) throw new Error("Dataset frames must be marked displayReady");

  if (output.frameMode === "SLMCONTROL3_LOGICAL") {
    if (config.lut !== null) throw new Error("SLMCONTROL3_LOGICAL mode must not embed a phase-response LUT");
    if (output.deviceReady !== false
      || output.lutApplication !== "SLMCONTROL3"
      || output.slmControl3LutMustBeEnabled !== true
      || output.slmControl3LutMustBeDisabled !== false) {
      throw new Error("SLMCONTROL3_LOGICAL output flags are inconsistent");
    }
    return;
  }

  if (output.frameMode !== "DEVICE_READY_LUT_BAKED") {
    throw new Error(`Unsupported dataset frameMode ${String(output.frameMode)}`);
  }
  if (output.deviceReady !== true
    || output.lutApplication !== "DAWN_NODE"
    || output.slmControl3LutMustBeEnabled !== false
    || output.slmControl3LutMustBeDisabled !== true) {
    throw new Error("DEVICE_READY_LUT_BAKED output flags are inconsistent");
  }

  const lut = config.lut;
  if (!isRecord(lut) || !Array.isArray(lut.values) || lut.values.length < 2 || !lut.values.every(Number.isFinite)) {
    throw new Error("DEVICE_READY_LUT_BAKED mode requires a measured phase-response LUT with at least two finite entries");
  }
  if (typeof lut.sourceSha256 !== "string" || typeof lut.valuesSha256 !== "string" || typeof lut.filename !== "string"
    || !/^[0-9a-f]{64}$/.test(lut.sourceSha256) || !/^[0-9a-f]{64}$/.test(lut.valuesSha256) || !lut.filename) {
    throw new Error("Measured LUT provenance (filename, sourceSha256, and valuesSha256) is invalid");
  }
  if (lut.phaseConvention !== "NEGATIVE_PI_TO_PI" && lut.phaseConvention !== "ZERO_TO_TWO_PI") {
    throw new Error("Measured LUT phaseConvention must be NEGATIVE_PI_TO_PI or ZERO_TO_TWO_PI");
  }
  if (lut.direction !== "INCREASING" && lut.direction !== "DECREASING") {
    throw new Error("Measured LUT direction must be INCREASING or DECREASING");
  }
}

function validateStatus(config: CollectorConfig, statusValue: CollectorStatus): void {
  if (!statusValue || statusValue.ok !== true) throw new Error("Collector returned an invalid status document");
  if (statusValue.requestedSamples !== config.requestedSamples) {
    throw new Error("Collector status/config requestedSamples mismatch");
  }
  if (statusValue.acceptedSamples !== config.acceptedSamples || statusValue.nextSampleId !== config.nextSampleId) {
    throw new Error("Collector status/config resume position mismatch; retry configuration inspection");
  }
  if (statusValue.configurationHash !== config.configHash) throw new Error("Collector status/config hash mismatch");
  safeInteger(statusValue.rejectedSamples, "status.rejectedSamples", 0);
  safeInteger(statusValue.currentSampleRejections, "status.currentSampleRejections", 0);
  if (statusValue.currentSampleRejections > statusValue.rejectedSamples) {
    throw new Error("Collector currentSampleRejections exceeds rejectedSamples");
  }
  if (!isRecord(statusValue.rejectionsByReason)) throw new Error("Collector status rejectionsByReason must be an object");
  for (const [reason, count] of Object.entries(statusValue.rejectionsByReason)) {
    if (!reason) throw new Error("Collector status contains an empty rejection reason");
    safeInteger(count, `status.rejectionsByReason[${reason}]`, 0);
  }
}

function summarizeConfig(config: CollectorConfig, collectorStatus: CollectorStatus): ConfigSummary {
  return {
    requestedSamples: config.requestedSamples,
    acceptedSamples: config.acceptedSamples,
    nextSampleId: config.nextSampleId,
    activeWidth: config.activeWidth,
    activeHeight: config.activeHeight,
    fftWidth: config.fftWidth,
    fftHeight: config.fftHeight,
    minTraps: config.sampling.minTraps,
    maxTraps: config.sampling.maxTraps,
    maxRetriesPerSample: config.sampling.maxRetriesPerSample,
    iterations: config.solver.iterations,
    convergenceTolerance: config.solver.convergenceTolerance,
    calibration: `${config.optics.wavelengthNm} nm · ${config.optics.focalLengthMm} mm · ${config.optics.pixelPitchUm} µm/px`,
    frameMode: config.output.frameMode,
    lutEntries: config.lut?.values.length ?? 0,
    lutSha256: config.lut?.valuesSha256 ?? null,
    rejectedSamples: collectorStatus.rejectedSamples,
    rejectionCounts: { ...collectorStatus.rejectionsByReason },
  };
}

function makeProgress(
  config: CollectorConfig,
  acceptedSamples: number,
  nextSampleId: number,
  trapCount: number,
  attempt: number,
  sampleElapsedMs: number,
  runStarted: number,
  acceptedThisRun: number,
  rejectedTotal: number,
  rejectionCounts: Record<string, number>,
  metrics?: FrameMetrics,
): ProgressSnapshot {
  const runElapsedMs = performance.now() - runStarted;
  return {
    acceptedSamples,
    requestedSamples: config.requestedSamples,
    nextSampleId,
    trapCount,
    attempt,
    sampleElapsedMs,
    runElapsedMs,
    acceptedPerHour: runElapsedMs > 0 ? acceptedThisRun * 3_600_000 / runElapsedMs : 0,
    rejectedTotal,
    rejectionCounts: { ...rejectionCounts },
    ...(metrics ? { metrics: metrics as unknown as Record<string, unknown> } : {}),
  };
}

interface SampleAcceptedResponse {
  ok: true;
  acceptedSamples: number;
  nextSampleId: number;
  remainingSamples: number;
  shardIndex: number;
  rowIndex: number;
}

function validateAcceptedResponse(
  response: SampleAcceptedResponse,
  config: CollectorConfig,
  previousAccepted: number,
  previousSampleId: number,
): void {
  if (response.ok !== true) throw new Error("Collector did not acknowledge the sample");
  if (response.acceptedSamples !== previousAccepted + 1) throw new Error("Collector acceptedSamples did not advance by one");
  if (response.nextSampleId !== previousSampleId + 1) throw new Error("Collector nextSampleId did not advance by one");
  if (response.acceptedSamples > config.requestedSamples) throw new Error("Collector accepted too many samples");
}

async function fetchConfig(collectorUrl: string): Promise<CollectorConfig> {
  return requestJson<CollectorConfig>(`${collectorUrl}/api/config`, { method: "GET" });
}

async function fetchCollectorStatus(collectorUrl: string): Promise<CollectorStatus> {
  return requestJson<CollectorStatus>(`${collectorUrl}/api/status`, { method: "GET" });
}

async function postJson(collectorUrl: string, path: string, body: unknown): Promise<unknown> {
  return requestJson(`${collectorUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function postBinary<T>(collectorUrl: string, path: string, payload: Uint8Array): Promise<T> {
  const body = new Uint8Array(payload).buffer;
  return requestJson<T>(`${collectorUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body,
  });
}

async function requestJson<T>(url: string, init: RequestInit): Promise<T> {
  const controller = new AbortController();
  activeFetch = controller;
  try {
    const response = await fetch(url, { ...init, cache: "no-store", signal: controller.signal });
    const text = await response.text();
    let value: unknown = {};
    if (text) {
      try {
        value = JSON.parse(text);
      } catch {
        throw new Error(`Collector returned non-JSON HTTP ${response.status}: ${text.slice(0, 240)}`);
      }
    }
    if (!response.ok) {
      const detail = isRecord(value) && typeof value.error === "string" ? value.error : response.statusText;
      throw new Error(`Collector HTTP ${response.status}: ${detail}`);
    }
    if (isRecord(value) && value.ok === false) {
      throw new Error(typeof value.error === "string" ? value.error : "Collector rejected the request");
    }
    return value as T;
  } finally {
    if (activeFetch === controller) activeFetch = undefined;
  }
}

function status(phase: string, message: string): void {
  post({ kind: "STATUS", phase, message });
}

function cancelled(): void {
  post({ kind: "CANCELLED", progress: lastProgress });
}

function reportError(error: unknown): void {
  if (cancelRequested && isAbortError(error)) {
    cancelled();
    running = false;
    return;
  }
  const normalized = error instanceof Error ? error : new Error(String(error));
  post({
    kind: "ERROR",
    name: normalized.name,
    message: normalized.message,
    ...(normalized.stack ? { stack: normalized.stack } : {}),
  });
  running = false;
}

function post(message: unknown): void {
  messageSink(message);
}

function allFinite(values: ArrayLike<number>): boolean {
  for (let index = 0; index < values.length; index += 1) {
    if (!Number.isFinite(values[index])) return false;
  }
  return true;
}

function jsonNumbersFinite(value: unknown): boolean {
  if (typeof value === "number") return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(jsonNumbersFinite);
  if (isRecord(value)) return Object.values(value).every(jsonNumbersFinite);
  return true;
}

function safeInteger(value: number, name: string, minimum: number): void {
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} must be a safe integer >= ${minimum}`);
}

function positiveInteger(value: number, name: string): void {
  safeInteger(value, name, 1);
}

function uint32(value: number, name: string): void {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new Error(`${name} must be a uint32`);
}

function positiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be finite and > 0`);
}

function nonnegativeFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${name} must be finite and >= 0`);
}

function finite(value: number, name: string): void {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
}

function isPowerOfTwo(value: number): boolean {
  return value > 0 && (value & (value - 1)) === 0;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function requestDatasetCancellation(): void {
  cancelRequested = true;
  activeFetch?.abort();
}

/** Replace JSONL output with a host-specific reporter such as a TTY progress bar. */
export function setDatasetMessageSink(sink: (message: unknown) => void): void {
  messageSink = sink;
}

export function disposeDatasetRuntime(): void {
  solver?.dispose();
  solver = undefined;
  calibration = undefined;
  runtimeFingerprint = "";
  running = false;
  messageSink = jsonLineMessageSink;
}
