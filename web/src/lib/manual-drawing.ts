export interface ManualDrawingPoint {
  xUm: number;
  yUm: number;
}

export interface ViewportRectangle {
  left: number;
  top: number;
  width: number;
  height: number;
}

const COORDINATE_PRECISION = 1_000_000;

/**
 * Resample a pointer polyline by travelled physical distance. The result does
 * not depend on how frequently the browser delivered pointer events.
 */
export function resampleDrawingPath(
  path: readonly ManualDrawingPoint[],
  spacingUm: number,
): ManualDrawingPoint[] {
  const spacing = positiveFinite(spacingUm, "Drawing point spacing");
  if (path.length === 0) return [];
  validatePoint(path[0]!, "Drawing point 1");

  const samples: ManualDrawingPoint[] = [roundPoint(path[0]!)];
  let distanceToNext = spacing;
  for (let index = 1; index < path.length; index += 1) {
    const segmentEnd = path[index]!;
    validatePoint(segmentEnd, `Drawing point ${index + 1}`);
    let segmentStart = path[index - 1]!;
    let remaining = pointDistance(segmentStart, segmentEnd);
    if (remaining === 0) continue;

    while (remaining + Number.EPSILON >= distanceToNext) {
      const fraction = distanceToNext / remaining;
      const sample = roundPoint({
        xUm: segmentStart.xUm + (segmentEnd.xUm - segmentStart.xUm) * fraction,
        yUm: segmentStart.yUm + (segmentEnd.yUm - segmentStart.yUm) * fraction,
      });
      samples.push(sample);
      segmentStart = sample;
      remaining = pointDistance(segmentStart, segmentEnd);
      distanceToNext = spacing;
    }
    distanceToNext -= remaining;
    if (distanceToNext <= spacing * 1e-12) distanceToNext = spacing;
  }
  return samples;
}

/** Keep candidates in path order while enforcing the focal-plane resolution. */
export function appendResolvableDrawingPoints(
  existing: readonly ManualDrawingPoint[],
  candidates: readonly ManualDrawingPoint[],
  minimumSeparationXUm: number,
  minimumSeparationYUm: number,
  maximumPoints: number,
): ManualDrawingPoint[] {
  const limit = Math.max(1, Math.floor(positiveFinite(maximumPoints, "Drawing point limit")));
  const separationX = nonNegativeFinite(minimumSeparationXUm, "Minimum X separation");
  const separationY = nonNegativeFinite(minimumSeparationYUm, "Minimum Y separation");
  const selected = existing.slice(0, limit).map((point, index) => {
    validatePoint(point, `Existing drawing point ${index + 1}`);
    return roundPoint(point);
  });

  for (const candidate of candidates) {
    if (selected.length >= limit) break;
    validatePoint(candidate, "Candidate drawing point");
    if (selected.every((point) => pointsAreSeparated(candidate, point, separationX, separationY))) {
      selected.push(roundPoint(candidate));
    }
  }
  return selected;
}

export function eraseDrawingPoints(
  points: readonly ManualDrawingPoint[],
  center: ManualDrawingPoint,
  radiusUm: number,
): ManualDrawingPoint[] {
  validatePoint(center, "Eraser center");
  const radius = positiveFinite(radiusUm, "Eraser radius");
  const radiusSquared = radius * radius;
  return points.filter((point, index) => {
    validatePoint(point, `Drawing point ${index + 1}`);
    const dx = point.xUm - center.xUm;
    const dy = point.yUm - center.yUm;
    return dx * dx + dy * dy > radiusSquared;
  });
}

export function mapViewportToDrawingField(
  clientX: number,
  clientY: number,
  rectangle: ViewportRectangle,
  fieldWidthUm: number,
  fieldHeightUm: number,
): ManualDrawingPoint {
  if (![rectangle.left, rectangle.top, rectangle.width, rectangle.height].every(Number.isFinite) ||
      rectangle.width <= 0 || rectangle.height <= 0) {
    throw new Error("Drawing viewport must have positive finite dimensions");
  }
  const width = positiveFinite(fieldWidthUm, "Drawing field width");
  const height = positiveFinite(fieldHeightUm, "Drawing field height");
  const normalizedX = clamp((clientX - rectangle.left) / rectangle.width, 0, 1);
  const normalizedY = clamp((clientY - rectangle.top) / rectangle.height, 0, 1);
  return roundPoint({
    xUm: (normalizedX - 0.5) * width,
    yUm: (0.5 - normalizedY) * height,
  });
}

export function pointDistance(first: ManualDrawingPoint, second: ManualDrawingPoint): number {
  return Math.hypot(first.xUm - second.xUm, first.yUm - second.yUm);
}

function pointsAreSeparated(
  first: ManualDrawingPoint,
  second: ManualDrawingPoint,
  separationX: number,
  separationY: number,
): boolean {
  if (separationX === 0 && separationY === 0) {
    return first.xUm !== second.xUm || first.yUm !== second.yUm;
  }
  const normalizedX = separationX === 0
    ? (first.xUm === second.xUm ? 0 : Number.POSITIVE_INFINITY)
    : (first.xUm - second.xUm) / separationX;
  const normalizedY = separationY === 0
    ? (first.yUm === second.yUm ? 0 : Number.POSITIVE_INFINITY)
    : (first.yUm - second.yUm) / separationY;
  return Math.hypot(normalizedX, normalizedY) >= 1;
}

function validatePoint(point: ManualDrawingPoint, label: string): void {
  if (!Number.isFinite(point.xUm) || !Number.isFinite(point.yUm)) {
    throw new Error(`${label} must contain finite coordinates`);
  }
}

function roundPoint(point: ManualDrawingPoint): ManualDrawingPoint {
  return {
    xUm: normalizeZero(Math.round(point.xUm * COORDINATE_PRECISION) / COORDINATE_PRECISION),
    yUm: normalizeZero(Math.round(point.yUm * COORDINATE_PRECISION) / COORDINATE_PRECISION),
  };
}

function positiveFinite(value: number, label: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be positive and finite`);
  return value;
}

function nonNegativeFinite(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) throw new Error(`${label} cannot be negative`);
  return value;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function normalizeZero(value: number): number {
  return Object.is(value, -0) ? 0 : value;
}
