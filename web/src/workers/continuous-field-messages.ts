import type {
  ContinuousFieldMetrics,
  ContinuousFieldRegion,
} from "../lib/continuous-field.js";
import type { OpticalCalibrationInput } from "../lib/optical-calibration.js";
import type { ComputeBackend } from "./compiler-messages.js";

export interface ContinuousFieldWorkerInput {
  targetBuffer: ArrayBuffer;
  targetWidth: number;
  targetHeight: number;
  slmWidth: number;
  slmHeight: number;
  fftWidth: number;
  fftHeight: number;
  fieldWidthUm: number;
  fieldHeightUm: number;
  fieldCenterXUm: number;
  fieldCenterYUm: number;
  iterations: number;
  mixingFactor: number;
  deterministicSeed: number;
  backend: ComputeBackend;
  opticalCalibration: OpticalCalibrationInput;
}

export type ContinuousFieldWorkerRequest = {
  kind: "GENERATE_CONTINUOUS_FIELD";
  jobId: number;
  input: ContinuousFieldWorkerInput;
};

export type ContinuousFieldWorkerResponse =
  | {
      kind: "CONTINUOUS_FIELD_RESULT";
      jobId: number;
      pixels: ArrayBuffer;
      intensity: ArrayBuffer;
      metrics: ContinuousFieldMetrics;
      targetRegion: ContinuousFieldRegion;
      backendId: string;
      checksum: number;
    }
  | {
      kind: "WORKER_ERROR";
      jobId: number;
      message: string;
      name: string;
      stack?: string;
    };
