/// <reference lib="webworker" />

import { detectImagePoints } from "../lib/image-points.js";
import {
  resultMessage,
  type TargetImageWorkerRequest,
  type TargetImageWorkerResponse,
} from "./target-image-messages.js";

const scope = self as DedicatedWorkerGlobalScope;
let rgba: Uint8ClampedArray | null = null;
let width = 0;
let height = 0;

scope.onmessage = (event: MessageEvent<TargetImageWorkerRequest>): void => {
  const request = event.data;
  try {
    if (request.kind === "LOAD_TARGET_IMAGE") {
      rgba = new Uint8ClampedArray(request.rgba);
      width = request.width;
      height = request.height;
    }
    if (!rgba) throw new Error("Upload a target-field image before running detection");
    const started = performance.now();
    const result = detectImagePoints(rgba, width, height, request.options);
    scope.postMessage(resultMessage(request.jobId, result, performance.now() - started));
  } catch (error) {
    const response: TargetImageWorkerResponse = {
      kind: "TARGET_IMAGE_ERROR",
      jobId: request.jobId,
      message: error instanceof Error ? error.message : "Unable to detect points in the target image",
    };
    scope.postMessage(response);
  }
};

export {};
