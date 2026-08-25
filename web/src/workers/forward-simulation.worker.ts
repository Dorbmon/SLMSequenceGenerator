/// <reference lib="webworker" />

import {
  simulateSlmFrameWasm,
  type ForwardSimulationInput,
} from "../lib/forward-simulation.js";
import { simulateSlmFrameWebGpu } from "../lib/webgpu-forward-simulation.js";
import type {
  ForwardSimulationWorkerRequest,
  ForwardSimulationWorkerResponse,
} from "./forward-simulation-messages.js";

const scope = self as DedicatedWorkerGlobalScope;

scope.onmessage = (event: MessageEvent<ForwardSimulationWorkerRequest>): void => {
  const request = event.data;
  void simulate(request).catch((error: unknown) => {
    const response: ForwardSimulationWorkerResponse = {
      kind: "FORWARD_SIMULATION_ERROR",
      jobId: request.jobId,
      message: error instanceof Error ? error.message : "Forward simulation failed",
      name: error instanceof Error ? error.name : "Error",
      ...(error instanceof Error && error.stack ? { stack: error.stack } : {}),
    };
    scope.postMessage(response);
  });
};

async function simulate(request: ForwardSimulationWorkerRequest): Promise<void> {
  const started = performance.now();
  const input: ForwardSimulationInput = {
    ...request.input,
    pixels: new Uint8Array(request.input.pixels),
  };
  const result = request.input.backend === "webgpu"
    ? await simulateSlmFrameWebGpu(input)
    : simulateSlmFrameWasm(input);
  const response: ForwardSimulationWorkerResponse = {
    kind: "FORWARD_SIMULATION_RESULT",
    jobId: request.jobId,
    intensity: result.intensity.buffer as ArrayBuffer,
    metrics: result.metrics,
    elapsedMs: performance.now() - started,
    backendId: result.backendId,
  };
  scope.postMessage(response, [response.intensity]);
}

export {};
