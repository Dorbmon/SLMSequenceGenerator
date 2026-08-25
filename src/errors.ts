import type { ValidationIssue } from "./types.js";

export type SlmStatusCode =
  | "OK"
  | "INVALID_ARGUMENT"
  | "INSUFFICIENT_ATOMS"
  | "DUPLICATE_ID"
  | "OUT_OF_BOUNDS"
  | "INVALID_TARGET_GEOMETRY"
  | "CALIBRATION_MISMATCH"
  | "ASSIGNMENT_INFEASIBLE"
  | "PATH_NOT_FOUND"
  | "COLLISION_VALIDATION_FAILED"
  | "MOTION_LIMIT_VIOLATION"
  | "FRAME_LIMIT_EXCEEDED"
  | "NUMERIC_ERROR"
  | "WGS_NOT_CONVERGED"
  | "FRAME_QUALITY_REJECTED"
  | "STORAGE_ERROR"
  | "CANCELLED"
  | "INTERNAL_ERROR";

export class SlmError extends Error {
  readonly code: SlmStatusCode;
  readonly stage: string;
  readonly retryable: boolean;
  readonly details: Record<string, unknown>;

  constructor(
    code: SlmStatusCode,
    message: string,
    options: {
      stage?: string;
      retryable?: boolean;
      details?: Record<string, unknown>;
      cause?: unknown;
    } = {},
  ) {
    super(message, { cause: options.cause });
    this.name = "SlmError";
    this.code = code;
    this.stage = options.stage ?? "UNKNOWN";
    this.retryable = options.retryable ?? false;
    this.details = options.details ?? {};
  }
}

export class SlmCompilationError extends SlmError {
  readonly issues: ValidationIssue[];

  constructor(
    code: SlmStatusCode,
    message: string,
    issues: ValidationIssue[] = [],
    options: ConstructorParameters<typeof SlmError>[2] = {},
  ) {
    super(code, message, options);
    this.name = "SlmCompilationError";
    this.issues = issues;
  }
}

export function assertFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) {
    throw new SlmError("INVALID_ARGUMENT", `${name} must be finite`, {
      stage: "VALIDATING",
      details: { name, value },
    });
  }
}

export function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new SlmError("CANCELLED", "Compilation was cancelled", {
      stage: "CANCELLED",
      retryable: true,
    });
  }
}
