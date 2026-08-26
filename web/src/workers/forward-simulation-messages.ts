import type { ComputeBackend } from "./compiler-messages.js";
import type { ForwardSimulationMetrics } from "../lib/forward-simulation.js";

export interface ForwardSimulationWorkerInput {
  pixels: ArrayBuffer;
  width: number;
  height: number;
  fftWidth: number;
  fftHeight: number;
  backend: ComputeBackend;
  phaseResponseLut?: number[];
}

export type ForwardSimulationWorkerRequest = {
  kind: "SIMULATE_SLM_FRAME";
  jobId: number;
  input: ForwardSimulationWorkerInput;
};

export type ForwardSimulationWorkerResponse =
  | {
      kind: "FORWARD_SIMULATION_RESULT";
      jobId: number;
      intensity: ArrayBuffer;
      metrics: ForwardSimulationMetrics;
      elapsedMs: number;
      backendId: string;
    }
  | {
      kind: "FORWARD_SIMULATION_ERROR";
      jobId: number;
      message: string;
      name: string;
      stack?: string;
    };
