import type { TrapFrame } from "../../../src/types.js";

export interface OpticalTweezerInput {
  trapId: number;
  xUm: number;
  yUm: number;
  phaseRad: number;
  intensity: number;
}

// The browser exports an 8-bit phase frame. A 0.01% amplitude certificate is
// below the quantized floor of the calibrated default four-trap frame
// (~0.0785%), so use a strict but attainable default for the UI.
export const DEFAULT_TWEEZER_AMPLITUDE_TOLERANCE_PERCENT = 0.1;

export const DEFAULT_OPTICAL_TWEEZERS: readonly OpticalTweezerInput[] = [
  { trapId: 1, xUm: -4, yUm: -4, phaseRad: 0, intensity: 1 },
  { trapId: 2, xUm: 4, yUm: -4, phaseRad: 1.570796, intensity: 1 },
  { trapId: 3, xUm: -4, yUm: 4, phaseRad: 3.141593, intensity: 1 },
  { trapId: 4, xUm: 4, yUm: 4, phaseRad: -1.570796, intensity: 1 },
];

export function cloneOpticalTweezers(tweezers: readonly OpticalTweezerInput[]): OpticalTweezerInput[] {
  return tweezers.map((tweezer) => ({ ...tweezer }));
}

export function serializeOpticalTweezers(tweezers: readonly OpticalTweezerInput[]): string {
  return JSON.stringify(tweezers, null, 2);
}

export function parseOpticalTweezers(raw: string): OpticalTweezerInput[] {
  const value: unknown = JSON.parse(raw);
  const list = Array.isArray(value)
    ? value
    : isRecord(value) && Array.isArray(value.tweezers)
      ? value.tweezers
      : isRecord(value) && Array.isArray(value.traps)
        ? value.traps
        : null;
  if (!list) throw new Error("Optical tweezers must be a JSON array or a tweezers/traps object");
  if (list.length === 0) throw new Error("Add at least one optical tweezer");

  const parsed = list.map((entry, index) => parseTweezer(entry, index));
  const identifiers = new Set<number>();
  for (const tweezer of parsed) {
    if (identifiers.has(tweezer.trapId)) throw new Error(`Trap ID ${tweezer.trapId} is duplicated`);
    identifiers.add(tweezer.trapId);
  }
  return parsed;
}

export function nextTweezerId(tweezers: readonly OpticalTweezerInput[]): number {
  return tweezers.length === 0 ? 1 : Math.max(...tweezers.map((tweezer) => tweezer.trapId)) + 1;
}

export function opticalTweezersToFrame(tweezers: readonly OpticalTweezerInput[]): TrapFrame {
  return {
    frameIndex: 0,
    timeUs: 0,
    traps: tweezers.map((tweezer) => ({
      trapId: tweezer.trapId,
      atomId: null,
      xUm: tweezer.xUm,
      yUm: tweezer.yUm,
      intensity: tweezer.intensity,
      targetPhaseRad: tweezer.phaseRad,
      flags: 0,
    })),
  };
}

function parseTweezer(entry: unknown, index: number): OpticalTweezerInput {
  const record = isRecord(entry) ? entry : {};
  const xUm: unknown = Array.isArray(entry) ? entry[0] : record.xUm ?? record.x;
  const yUm: unknown = Array.isArray(entry) ? entry[1] : record.yUm ?? record.y;
  const phaseRad: unknown = Array.isArray(entry)
    ? entry[2]
    : record.phaseRad ?? record.targetPhaseRad ?? record.phase;
  const intensity: unknown = Array.isArray(entry) ? (entry[3] ?? 1) : (record.intensity ?? 1);
  const trapId: unknown = Array.isArray(entry) ? index + 1 : (record.trapId ?? record.id ?? index + 1);

  if (typeof trapId !== "number" || !Number.isSafeInteger(trapId) || trapId < 0) {
    throw new Error(`Tweezer ${index + 1} has an invalid trap ID`);
  }
  if (!isFiniteNumber(xUm) || !isFiniteNumber(yUm)) {
    throw new Error(`Tweezer ${index + 1} has invalid coordinates`);
  }
  if (!isFiniteNumber(phaseRad)) {
    throw new Error(`Tweezer ${index + 1} has an invalid phase`);
  }
  if (!isFiniteNumber(intensity) || intensity <= 0) {
    throw new Error(`Tweezer ${index + 1} intensity must be greater than zero`);
  }
  return { trapId, xUm, yUm, phaseRad, intensity };
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
