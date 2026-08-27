/// <reference lib="webworker" />

import { crc32 } from "../../../src/index.js";
import {
  solveContinuousFieldWasm,
  type ContinuousFieldInput,
} from "../lib/continuous-field.js";
import { solveContinuousFieldWebGpu } from "../lib/webgpu-continuous-field.js";
import type {
  ContinuousFieldWorkerRequest,
  ContinuousFieldWorkerResponse,
} from "./continuous-field-messages.js";

const scope = self as DedicatedWorkerGlobalScope;

scope.onmessage = (event: MessageEvent<ContinuousFieldWorkerRequest>): void => {
  const request = event.data;
  void generate(request).catch((error: unknown) => {
    const response: ContinuousFieldWorkerResponse = {
      kind: "WORKER_ERROR",
      jobId: request.jobId,
      message: error instanceof Error ? error.message : "Continuous-field computation failed",
      name: error instanceof Error ? error.name : "Error",
      ...(error instanceof Error && error.stack ? { stack: error.stack } : {}),
    };
    scope.postMessage(response);
  });
};

async function generate(request: ContinuousFieldWorkerRequest): Promise<void> {
  const source = request.input;
  const input: ContinuousFieldInput = {
    targetIntensity: new Float32Array(source.targetBuffer),
    targetWidth: source.targetWidth,
    targetHeight: source.targetHeight,
    slmWidth: source.slmWidth,
    slmHeight: source.slmHeight,
    fftWidth: source.fftWidth,
    fftHeight: source.fftHeight,
    fieldWidthUm: source.fieldWidthUm,
    fieldHeightUm: source.fieldHeightUm,
    fieldCenterXUm: source.fieldCenterXUm,
    fieldCenterYUm: source.fieldCenterYUm,
    iterations: source.iterations,
    mixingFactor: source.mixingFactor,
    deterministicSeed: source.deterministicSeed,
    opticalCalibration: source.opticalCalibration,
  };
  const result = source.backend === "webgpu"
    ? await solveContinuousFieldWebGpu(input)
    : solveContinuousFieldWasm(input);
  const pixels = new Uint8Array(result.pixels).buffer;
  const intensity = new Float32Array(result.intensity).buffer;
  const response: ContinuousFieldWorkerResponse = {
    kind: "CONTINUOUS_FIELD_RESULT",
    jobId: request.jobId,
    pixels,
    intensity,
    metrics: result.metrics,
    targetRegion: result.targetRegion,
    backendId: result.backendId,
    checksum: crc32(new Uint8Array(pixels)),
  };
  scope.postMessage(response, [pixels, intensity]);
}

export {};
