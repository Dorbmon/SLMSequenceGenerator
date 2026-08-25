import type {
  DetectedImagePoint,
  SpotDetectionOptions,
  SpotDetectionResult,
} from "../lib/image-points.js";

export type TargetImageWorkerRequest =
  | {
      kind: "LOAD_TARGET_IMAGE";
      jobId: number;
      width: number;
      height: number;
      rgba: ArrayBuffer;
      options: SpotDetectionOptions;
    }
  | {
      kind: "DETECT_TARGET_IMAGE";
      jobId: number;
      options: SpotDetectionOptions;
    };

export type TargetImageWorkerResponse =
  | {
      kind: "TARGET_IMAGE_RESULT";
      jobId: number;
      points: DetectedImagePoint[];
      thresholdSignal: number;
      minimumSignal: number;
      maximumSignal: number;
      discardedSmallComponents: number;
      discardedLargeComponents: number;
      discardedByLimit: number;
      elapsedMs: number;
    }
  | {
      kind: "TARGET_IMAGE_ERROR";
      jobId: number;
      message: string;
    };

export function resultMessage(
  jobId: number,
  result: SpotDetectionResult,
  elapsedMs: number,
): TargetImageWorkerResponse {
  return { kind: "TARGET_IMAGE_RESULT", jobId, ...result, elapsedMs };
}
