import { SlmError } from "./errors.js";
import type { CalibrationPackage, Point2D } from "./types.js";

export interface FftCoordinate {
  x: number;
  y: number;
}

/**
 * Map a physical focal-plane point to its DFT frequency coordinate. Signed
 * optical frequencies stay signed so an f32 GPU buffer does not lose their low
 * bits by first adding a large FFT extent.
 */
export function mapPhysicalPointToDftFrequency(
  point: Point2D,
  calibration: CalibrationPackage,
  width: number,
  height: number,
): FftCoordinate {
  const transform = calibration.coordinateTransform;
  let mapped: FftCoordinate;
  if (transform?.physicalToFft) {
    const result = transform.physicalToFft(point);
    mapped = "x" in result ? result : { x: result.xUm, y: result.yUm };
  } else if (transform && [transform.a, transform.b, transform.c, transform.d].every((value) => value !== undefined)) {
    mapped = {
      x: transform.a! * point.xUm + transform.b! * point.yUm + (transform.offsetX ?? 0),
      y: transform.c! * point.xUm + transform.d! * point.yUm + (transform.offsetY ?? 0),
    };
  } else {
    const originX = transform?.originXUm ?? width / 2;
    const originY = transform?.originYUm ?? height / 2;
    const scaleX = transform?.scaleX ?? 1;
    const scaleY = transform?.scaleY ?? 1;
    const rotation = transform?.rotationRad ?? 0;
    const x = point.xUm * scaleX;
    const y = point.yUm * scaleY;
    mapped = {
      x: originX + Math.cos(rotation) * x - Math.sin(rotation) * y,
      y: originY - Math.sin(rotation) * x - Math.cos(rotation) * y,
    };
  }

  if (!Number.isFinite(mapped.x) || !Number.isFinite(mapped.y)) {
    throw coordinateError("A calibrated trap coordinate is not finite", mapped, width, height);
  }
  if (transform?.fftCoordinateSpace === "SIGNED_FREQUENCY") {
    const tolerance = 1e-9;
    if (mapped.x < -width / 2 - tolerance || mapped.x > width / 2 + tolerance ||
        mapped.y < -height / 2 - tolerance || mapped.y > height / 2 + tolerance) {
      throw coordinateError("A calibrated trap coordinate exceeds the optical Nyquist field of view", mapped, width, height);
    }
  } else if (mapped.x < -0.5 || mapped.x > width - 0.5 || mapped.y < -0.5 || mapped.y > height - 0.5) {
    throw coordinateError("A calibrated trap coordinate lies outside the FFT grid", mapped, width, height);
  }
  return mapped;
}

/** Map a physical focal-plane point to a raw, periodic FFT storage index. */
export function mapPhysicalPointToFft(
  point: Point2D,
  calibration: CalibrationPackage,
  width: number,
  height: number,
): FftCoordinate {
  const mapped = mapPhysicalPointToDftFrequency(point, calibration, width, height);
  return { x: wrapFftCoordinate(mapped.x, width), y: wrapFftCoordinate(mapped.y, height) };
}

export function wrapFftCoordinate(value: number, extent: number): number {
  const wrapped = ((value % extent) + extent) % extent;
  return wrapped === extent ? 0 : wrapped;
}

export function periodicFftDistance(first: number, second: number, extent: number): number {
  const direct = Math.abs(wrapFftCoordinate(first, extent) - wrapFftCoordinate(second, extent));
  return Math.min(direct, extent - direct);
}

function coordinateError(
  message: string,
  mapped: FftCoordinate,
  width: number,
  height: number,
): SlmError {
  return new SlmError("OUT_OF_BOUNDS", message, {
    stage: "SOLVING_SLM_FRAMES",
    details: { target: mapped, width, height },
  });
}
