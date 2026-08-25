export type SpotPolarity = "BRIGHT" | "DARK";
export type ImageExtractionMode = "CENTROIDS" | "PATTERN";

export interface SpotDetectionOptions {
  polarity: SpotPolarity;
  threshold: number;
  minimumAreaPx: number;
  maximumPoints: number;
  mode?: ImageExtractionMode;
  patternSpacingPx?: number;
}

export interface DetectedImagePoint {
  xPx: number;
  yPx: number;
  areaPx: number;
  peakSignal: number;
  integratedSignal: number;
}

export interface SpotDetectionResult {
  points: DetectedImagePoint[];
  mode: ImageExtractionMode;
  thresholdSignal: number;
  minimumSignal: number;
  maximumSignal: number;
  sourcePixelCount: number;
  effectiveSpacingPx: number;
  discardedSmallComponents: number;
  discardedLargeComponents: number;
  discardedByLimit: number;
}

export interface PhysicalImagePoint extends DetectedImagePoint {
  xUm: number;
  yUm: number;
}

const MAX_COMPONENT_FRACTION = 0.2;

export function detectImagePoints(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  options: SpotDetectionOptions,
): SpotDetectionResult {
  validateImage(rgba, width, height);
  const threshold = clamp(options.threshold, 0, 1);
  const minimumAreaPx = positiveInteger(options.minimumAreaPx, 1);
  const maximumPoints = positiveInteger(options.maximumPoints, 1);
  const pixelCount = width * height;
  const signal = new Uint8Array(pixelCount);
  let minimumSignal = 255;
  let maximumSignal = 0;

  for (let index = 0; index < pixelCount; index += 1) {
    const offset = index * 4;
    const alpha = rgba[offset + 3]! / 255;
    const luminance = (
      rgba[offset]! * 0.2126
      + rgba[offset + 1]! * 0.7152
      + rgba[offset + 2]! * 0.0722
    );
    const value = Math.round((options.polarity === "DARK" ? 255 - luminance : luminance) * alpha);
    signal[index] = value;
    minimumSignal = Math.min(minimumSignal, value);
    maximumSignal = Math.max(maximumSignal, value);
  }

  const thresholdSignal = minimumSignal + (maximumSignal - minimumSignal) * threshold;
  if (options.mode === "PATTERN") {
    return detectPatternPoints(
      signal,
      width,
      height,
      thresholdSignal,
      minimumSignal,
      maximumSignal,
      minimumAreaPx,
      maximumPoints,
      positiveInteger(options.patternSpacingPx ?? 2, 2),
    );
  }
  const visited = new Uint8Array(pixelCount);
  const stack = new Int32Array(pixelCount);
  const candidates: DetectedImagePoint[] = [];
  const maximumAreaPx = Math.max(minimumAreaPx, Math.floor(pixelCount * MAX_COMPONENT_FRACTION));
  let discardedSmallComponents = 0;
  let discardedLargeComponents = 0;

  for (let seed = 0; seed < pixelCount; seed += 1) {
    if (visited[seed] || signal[seed]! < thresholdSignal) continue;
    visited[seed] = 1;
    let stackSize = 1;
    stack[0] = seed;
    let areaPx = 0;
    let weightedX = 0;
    let weightedY = 0;
    let integratedSignal = 0;
    let peakSignal = 0;

    while (stackSize > 0) {
      const index = stack[--stackSize]!;
      const x = index % width;
      const y = Math.floor(index / width);
      const value = signal[index]!;
      const weight = value - thresholdSignal + 1;
      areaPx += 1;
      weightedX += x * weight;
      weightedY += y * weight;
      integratedSignal += weight;
      peakSignal = Math.max(peakSignal, value);

      const minY = Math.max(0, y - 1);
      const maxY = Math.min(height - 1, y + 1);
      const minX = Math.max(0, x - 1);
      const maxX = Math.min(width - 1, x + 1);
      for (let neighborY = minY; neighborY <= maxY; neighborY += 1) {
        for (let neighborX = minX; neighborX <= maxX; neighborX += 1) {
          const neighbor = neighborY * width + neighborX;
          if (neighbor === index || visited[neighbor] || signal[neighbor]! < thresholdSignal) continue;
          visited[neighbor] = 1;
          stack[stackSize++] = neighbor;
        }
      }
    }

    if (areaPx < minimumAreaPx) {
      discardedSmallComponents += 1;
      continue;
    }
    if (areaPx > maximumAreaPx) {
      discardedLargeComponents += 1;
      continue;
    }
    candidates.push({
      xPx: weightedX / integratedSignal,
      yPx: weightedY / integratedSignal,
      areaPx,
      peakSignal,
      integratedSignal,
    });
  }

  const discardedByLimit = Math.max(0, candidates.length - maximumPoints);
  const strongest = candidates
    .sort((left, right) => right.integratedSignal - left.integratedSignal || right.peakSignal - left.peakSignal)
    .slice(0, maximumPoints)
    .sort((left, right) => left.yPx - right.yPx || left.xPx - right.xPx);

  return {
    points: strongest,
    mode: "CENTROIDS",
    thresholdSignal,
    minimumSignal,
    maximumSignal,
    sourcePixelCount: candidates.reduce((sum, point) => sum + point.areaPx, 0),
    effectiveSpacingPx: 0,
    discardedSmallComponents,
    discardedLargeComponents,
    discardedByLimit,
  };
}

function detectPatternPoints(
  signal: Uint8Array,
  width: number,
  height: number,
  thresholdSignal: number,
  minimumSignal: number,
  maximumSignal: number,
  minimumAreaPx: number,
  maximumPoints: number,
  requestedSpacingPx: number,
): SpotDetectionResult {
  const pixelCount = width * height;
  const labels = new Int32Array(pixelCount);
  const componentAreas = [0];
  const stack = new Int32Array(pixelCount);
  let componentId = 0;

  for (let seed = 0; seed < pixelCount; seed += 1) {
    if (labels[seed] !== 0 || signal[seed]! < thresholdSignal) continue;
    componentId += 1;
    labels[seed] = componentId;
    let stackSize = 1;
    stack[0] = seed;
    let area = 0;
    while (stackSize > 0) {
      const index = stack[--stackSize]!;
      area += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      const minY = Math.max(0, y - 1);
      const maxY = Math.min(height - 1, y + 1);
      const minX = Math.max(0, x - 1);
      const maxX = Math.min(width - 1, x + 1);
      for (let neighborY = minY; neighborY <= maxY; neighborY += 1) {
        for (let neighborX = minX; neighborX <= maxX; neighborX += 1) {
          const neighbor = neighborY * width + neighborX;
          if (neighbor === index || labels[neighbor] !== 0 || signal[neighbor]! < thresholdSignal) continue;
          labels[neighbor] = componentId;
          stack[stackSize++] = neighbor;
        }
      }
    }
    componentAreas[componentId] = area;
  }

  let discardedSmallComponents = 0;
  for (let id = 1; id < componentAreas.length; id += 1) {
    if (componentAreas[id]! < minimumAreaPx) discardedSmallComponents += 1;
  }
  let sourcePixelCount = 0;
  for (let index = 0; index < pixelCount; index += 1) {
    const label = labels[index]!;
    if (label !== 0 && componentAreas[label]! >= minimumAreaPx) sourcePixelCount += 1;
  }

  let effectiveSpacingPx = requestedSpacingPx;
  let points = binPatternPixels(
    signal,
    labels,
    componentAreas,
    width,
    height,
    thresholdSignal,
    minimumAreaPx,
    effectiveSpacingPx,
  );
  const pointsAtRequestedSpacing = points.length;
  while (points.length > maximumPoints && effectiveSpacingPx < Math.max(width, height)) {
    const ratio = Math.sqrt(points.length / maximumPoints);
    effectiveSpacingPx = Math.max(effectiveSpacingPx + 1, Math.ceil(effectiveSpacingPx * ratio));
    points = binPatternPixels(
      signal,
      labels,
      componentAreas,
      width,
      height,
      thresholdSignal,
      minimumAreaPx,
      effectiveSpacingPx,
    );
  }
  if (points.length > maximumPoints) points = evenlyLimitPoints(points, maximumPoints);

  return {
    points,
    mode: "PATTERN",
    thresholdSignal,
    minimumSignal,
    maximumSignal,
    sourcePixelCount,
    effectiveSpacingPx,
    discardedSmallComponents,
    discardedLargeComponents: 0,
    discardedByLimit: Math.max(0, pointsAtRequestedSpacing - points.length),
  };
}

function binPatternPixels(
  signal: Uint8Array,
  labels: Int32Array,
  componentAreas: readonly number[],
  width: number,
  height: number,
  thresholdSignal: number,
  minimumAreaPx: number,
  spacingPx: number,
): DetectedImagePoint[] {
  interface PatternBin {
    weightedX: number;
    weightedY: number;
    integratedSignal: number;
    peakSignal: number;
    areaPx: number;
  }
  const columnCount = Math.ceil(width / spacingPx);
  const bins = new Map<number, PatternBin>();
  for (let index = 0; index < width * height; index += 1) {
    const label = labels[index]!;
    if (label === 0 || componentAreas[label]! < minimumAreaPx) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    const key = Math.floor(y / spacingPx) * columnCount + Math.floor(x / spacingPx);
    const value = signal[index]!;
    const weight = value - thresholdSignal + 1;
    const bin = bins.get(key) ?? {
      weightedX: 0,
      weightedY: 0,
      integratedSignal: 0,
      peakSignal: 0,
      areaPx: 0,
    };
    bin.weightedX += x * weight;
    bin.weightedY += y * weight;
    bin.integratedSignal += weight;
    bin.peakSignal = Math.max(bin.peakSignal, value);
    bin.areaPx += 1;
    bins.set(key, bin);
  }
  return [...bins.values()]
    .map((bin) => ({
      xPx: bin.weightedX / bin.integratedSignal,
      yPx: bin.weightedY / bin.integratedSignal,
      areaPx: bin.areaPx,
      peakSignal: bin.peakSignal,
      integratedSignal: bin.integratedSignal,
    }))
    .sort((left, right) => left.yPx - right.yPx || left.xPx - right.xPx);
}

function evenlyLimitPoints(points: DetectedImagePoint[], maximumPoints: number): DetectedImagePoint[] {
  if (points.length <= maximumPoints) return points;
  return Array.from({ length: maximumPoints }, (_, index) => (
    points[Math.min(points.length - 1, Math.floor((index + 0.5) * points.length / maximumPoints))]!
  ));
}

export function mapImagePointsToField(
  points: readonly DetectedImagePoint[],
  imageWidth: number,
  imageHeight: number,
  fieldWidthUm: number,
  fieldHeightUm: number,
): PhysicalImagePoint[] {
  if (!Number.isFinite(fieldWidthUm) || fieldWidthUm <= 0 || !Number.isFinite(fieldHeightUm) || fieldHeightUm <= 0) {
    throw new Error("Image field width and height must be positive finite values");
  }
  if (!Number.isInteger(imageWidth) || imageWidth <= 0 || !Number.isInteger(imageHeight) || imageHeight <= 0) {
    throw new Error("Image dimensions must be positive integers");
  }
  const xSpan = Math.max(1, imageWidth - 1);
  const ySpan = Math.max(1, imageHeight - 1);
  return points.map((point) => ({
    ...point,
    xUm: roundCoordinate((point.xPx / xSpan - 0.5) * fieldWidthUm),
    yUm: roundCoordinate((0.5 - point.yPx / ySpan) * fieldHeightUm),
  }));
}

function validateImage(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): void {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error("Image dimensions must be positive integers");
  }
  if (rgba.length !== width * height * 4) {
    throw new Error(`RGBA buffer length ${rgba.length} does not match ${width} x ${height}`);
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, Number.isFinite(value) ? value : minimum));
}

function positiveInteger(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(1, Math.round(value)) : fallback;
}

function roundCoordinate(value: number): number {
  const rounded = Math.round(value * 1e6) / 1e6;
  return Object.is(rounded, -0) ? 0 : rounded;
}
