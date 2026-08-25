import { SlmError } from "./errors.js";

export const TAU = Math.PI * 2;

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

export function wrapPhase(value: number): number {
  const wrapped = ((value + Math.PI) % TAU + TAU) % TAU - Math.PI;
  return wrapped === -Math.PI ? Math.PI : wrapped;
}

export function angularDistance(a: number, b: number): number {
  return Math.abs(wrapPhase(a - b));
}

export function smoothstep5(s: number): number {
  const t = clamp(s, 0, 1);
  return t * t * t * (10 + t * (-15 + 6 * t));
}

export function stableStringify(value: unknown): string {
  return stringifyValue(value);
}

function stringifyValue(value: unknown): string {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return JSON.stringify(String(value));
    return JSON.stringify(Object.is(value, -0) ? 0 : value);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "bigint") return JSON.stringify(value.toString());
  if (typeof value === "undefined") return "null";
  if (ArrayBuffer.isView(value)) {
    return `[${Array.from(value as unknown as ArrayLike<number>, (item) => stringifyValue(item)).join(",")}]`;
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stringifyValue(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const keys = Object.keys(record).sort();
    return `{${keys
      .map((key) => `${JSON.stringify(key)}:${stringifyValue(record[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(String(value));
}

/** FNV-1a is used for stable manifests without requiring a platform crypto API. */
export function hashString(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

export function hashValue(value: unknown): string {
  return hashString(stableStringify(value));
}

export function crc32(bytes: ArrayLike<number>): number {
  let crc = 0xffffffff;
  for (let index = 0; index < bytes.length; index += 1) {
    crc ^= bytes[index]! & 0xff;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function asNumberArray(value: ArrayLike<number>, expected: number, name: string): number[] {
  if (value.length !== expected) {
    throw new SlmError("CALIBRATION_MISMATCH", `${name} must contain ${expected} values`, {
      stage: "VALIDATING",
      details: { expected, actual: value.length, name },
    });
  }
  const result = Array.from(value, Number);
  if (result.some((item) => !Number.isFinite(item))) {
    throw new SlmError("INVALID_ARGUMENT", `${name} contains a non-finite value`, {
      stage: "VALIDATING",
    });
  }
  return result;
}

export function numberOr(value: number | undefined, fallback: number): number {
  return value === undefined ? fallback : value;
}

export function integerOr(value: number | undefined, fallback: number): number {
  const result = numberOr(value, fallback);
  if (!Number.isInteger(result)) {
    throw new SlmError("INVALID_ARGUMENT", `Expected an integer, received ${result}`, {
      stage: "VALIDATING",
    });
  }
  return result;
}

export function cloneTyped<T extends Uint8Array | Uint16Array | Float32Array | Float64Array>(value: T): T {
  return new (value.constructor as { new (source: T): T })(value);
}
