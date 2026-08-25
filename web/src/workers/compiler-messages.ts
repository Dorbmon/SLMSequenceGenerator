import type {
  AtomAssignment,
  AtomTrajectory,
  CompileProgress,
  FrameMetrics,
  InitialAtom,
  SequenceManifest,
  SequenceValidationReport,
  SlmFrameDescriptor,
  TargetSite,
  TrapFrame,
} from "../../../src/types.js";
import type { OpticalTweezerInput } from "../lib/tweezers.js";
import type { OpticalCalibrationInput } from "../lib/optical-calibration.js";

export type ComputeBackend = "wasm" | "webgpu";

export interface SequenceWorkerInput {
  initialAtoms: InitialAtom[];
  targetSites: TargetSite[];
  separationUm: number;
  iterations: number;
  slmWidth: number;
  slmHeight: number;
  fftWidth: number;
  fftHeight: number;
  targetPhaseMode: "PHASE_LOCKED_WGS" | "SOFT_PHASE_LOCKED_WGS";
  backend: ComputeBackend;
  opticalCalibration: OpticalCalibrationInput;
}

export interface TweezerFrameWorkerInput {
  tweezers: OpticalTweezerInput[];
  slmWidth: number;
  slmHeight: number;
  fftWidth: number;
  fftHeight: number;
  iterations: number;
  backend: ComputeBackend;
  opticalCalibration: OpticalCalibrationInput;
}

export type CompilerWorkerRequest =
  | { kind: "CHECK_WEBGPU"; jobId: number }
  | { kind: "COMPILE_SEQUENCE"; jobId: number; input: SequenceWorkerInput }
  | { kind: "GENERATE_TWEEZER_FRAME"; jobId: number; input: TweezerFrameWorkerInput };

export interface SerializedSlmFrame {
  format: "UINT8" | "UINT16";
  buffer: ArrayBuffer;
}

export interface SerializedSequence {
  manifest: SequenceManifest;
  assignment: AtomAssignment[];
  trajectories: AtomTrajectory[];
  trapFrames: TrapFrame[];
  slmFrames: SerializedSlmFrame[];
  slmFrameDescriptors: SlmFrameDescriptor[];
  frameMetrics: FrameMetrics[];
  validation: SequenceValidationReport;
}

export type CompilerWorkerResponse =
  | { kind: "WEBGPU_CAPABILITY"; jobId: number; available: boolean; reason: string; adapter?: string }
  | { kind: "SEQUENCE_PROGRESS"; jobId: number; progress: CompileProgress }
  | { kind: "SEQUENCE_RESULT"; jobId: number; sequence: SerializedSequence; elapsedMs: number }
  | {
      kind: "TWEEZER_FRAME_RESULT";
      jobId: number;
      format: "UINT8" | "UINT16";
      buffer: ArrayBuffer;
      metrics: FrameMetrics;
      elapsedMs: number;
      checksum: number;
      backendId: string;
    }
  | { kind: "WORKER_ERROR"; jobId: number; message: string; name: string; stack?: string };
