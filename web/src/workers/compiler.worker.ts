/// <reference lib="webworker" />

import {
  SequentialWgsSolver,
  SlmSequenceCompiler,
  crc32,
  type CalibrationPackage,
  type SequentialHologramBackend,
} from "../../../src/index.js";
import { opticalTweezersToFrame } from "../lib/tweezers.js";
import { createOpticalCalibration } from "../lib/optical-calibration.js";
import {
  inspectWebGpu,
  WebGpuSequentialWgsSolver,
} from "../lib/webgpu-wgs.js";
import type {
  CompilerWorkerRequest,
  CompilerWorkerResponse,
  SequenceWorkerInput,
  SerializedSlmFrame,
  TweezerFrameWorkerInput,
} from "./compiler-messages.js";

const scope = self as DedicatedWorkerGlobalScope;

scope.onmessage = (event: MessageEvent<CompilerWorkerRequest>): void => {
  const request = event.data;
  void runJob(request).catch((error: unknown) => {
    const response: CompilerWorkerResponse = {
      kind: "WORKER_ERROR",
      jobId: request.jobId,
      message: error instanceof Error ? error.message : "Worker computation failed",
      name: error instanceof Error ? error.name : "Error",
      ...(error instanceof Error && error.stack ? { stack: error.stack } : {}),
    };
    scope.postMessage(response);
  });
};

async function runJob(request: CompilerWorkerRequest): Promise<void> {
  if (request.kind === "CHECK_WEBGPU") {
    const capability = await inspectWebGpu();
    const response: CompilerWorkerResponse = {
      kind: "WEBGPU_CAPABILITY",
      jobId: request.jobId,
      ...capability,
    };
    scope.postMessage(response);
    return;
  }
  if (request.kind === "COMPILE_SEQUENCE") {
    await compileSequence(request.jobId, request.input);
    return;
  }
  await generateTweezerFrame(request.jobId, request.input);
}

async function compileSequence(jobId: number, input: SequenceWorkerInput): Promise<void> {
  const started = performance.now();
  const calibration = sequenceCalibration(input);
  const compiler = await SlmSequenceCompiler.create({
    simulationMode: true,
    calibration,
    hologram: {
      width: input.fftWidth,
      height: input.fftHeight,
      format: "UINT8",
      firstFrameIterations: input.iterations,
      subsequentFrameIterations: input.iterations,
      maxIterations: Math.max(8, input.iterations * 2),
      targetPhaseMode: input.targetPhaseMode,
      requireConvergence: false,
    },
    planner: {
      minimumSeparationUm: input.separationUm,
      geometricMarginUm: 0.1,
      gridResolutionUm: Math.max(0.5, input.separationUm / 2),
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
    initialAtoms: input.initialAtoms,
    targetSites: input.targetSites,
    calibrationId: calibration.manifest.calibrationId,
  }, {
    onProgress(progress) {
      const response: CompilerWorkerResponse = { kind: "SEQUENCE_PROGRESS", jobId, progress };
      scope.postMessage(response);
    },
    ...(input.backend === "webgpu" ? {
      hologramSolverFactory: (solverCalibration: CalibrationPackage, solverConfig: Parameters<typeof WebGpuSequentialWgsSolver.create>[1]) => (
        WebGpuSequentialWgsSolver.create(solverCalibration, solverConfig)
      ),
    } : {}),
  });
  const sourceFrames = await Promise.resolve(compiled.slmFrameStore.toArray());
  const slmFrames = sourceFrames.map(copySlmFrame);
  const response: CompilerWorkerResponse = {
    kind: "SEQUENCE_RESULT",
    jobId,
    elapsedMs: performance.now() - started,
    sequence: {
      manifest: compiled.manifest,
      assignment: compiled.assignment,
      trajectories: compiled.trajectories,
      trapFrames: await Promise.resolve(compiled.trapFrameStore.toArray()),
      slmFrames,
      slmFrameDescriptors: compiled.slmFrameDescriptors,
      frameMetrics: compiled.frameMetrics,
      validation: compiled.validation,
    },
  };
  scope.postMessage(response, slmFrames.map((frame) => frame.buffer));
}

async function generateTweezerFrame(jobId: number, input: TweezerFrameWorkerInput): Promise<void> {
  const started = performance.now();
  const calibration = tweezerCalibration(input);
  const config = {
    width: input.fftWidth,
    height: input.fftHeight,
    format: "UINT8",
    firstFrameIterations: input.iterations,
    subsequentFrameIterations: input.iterations,
    maxIterations: input.iterations,
    targetPhaseMode: "PHASE_LOCKED_WGS",
    backgroundPolicy: "ZERO",
    requireConvergence: false,
    measureSolveTime: true,
  } as const;
  const solver: SequentialHologramBackend = input.backend === "webgpu"
    ? await WebGpuSequentialWgsSolver.create(calibration, config)
    : new SequentialWgsSolver(calibration, config);
  try {
    const result = await solver.solveSequentialFrame(opticalTweezersToFrame(input.tweezers));
    const frame = copySlmFrame(result.pixels);
    const response: CompilerWorkerResponse = {
      kind: "TWEEZER_FRAME_RESULT",
      jobId,
      format: frame.format,
      buffer: frame.buffer,
      metrics: result.metrics,
      elapsedMs: performance.now() - started,
      checksum: crc32(bytesFor(result.pixels)),
      backendId: solver.backendId,
    };
    scope.postMessage(response, [frame.buffer]);
  } finally {
    await solver.dispose?.();
  }
}

function sequenceCalibration(input: SequenceWorkerInput): CalibrationPackage {
  return createOpticalCalibration({
    activeWidth: input.slmWidth,
    activeHeight: input.slmHeight,
    fftWidth: input.fftWidth,
    fftHeight: input.fftHeight,
  }, input.opticalCalibration, "browser-sequence-optical-calibration");
}

function tweezerCalibration(input: TweezerFrameWorkerInput): CalibrationPackage {
  return createOpticalCalibration({
    activeWidth: input.slmWidth,
    activeHeight: input.slmHeight,
    fftWidth: input.fftWidth,
    fftHeight: input.fftHeight,
  }, input.opticalCalibration, "browser-single-frame-optical-calibration");
}

function copySlmFrame(frame: Uint8Array | Uint16Array): SerializedSlmFrame {
  if (frame instanceof Uint16Array) {
    return { format: "UINT16", buffer: new Uint16Array(frame).buffer };
  }
  return { format: "UINT8", buffer: new Uint8Array(frame).buffer };
}

function bytesFor(frame: Uint8Array | Uint16Array): Uint8Array {
  return new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
}

export {};
