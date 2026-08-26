import {
  mapPhysicalPointToDftFrequency,
  type CalibrationPackage,
  type Point2D,
} from "../../../src/index.js";

export const DEFAULT_WAVELENGTH_NM = 780;
export const DEFAULT_FOCAL_LENGTH_MM = 20;
export const DEFAULT_PIXEL_PITCH_UM = 8;

export interface OpticalCalibrationInput {
  wavelengthNm: number;
  focalLengthMm: number;
  pixelPitchUm: number;
  phaseResponseLut?: number[];
}

export interface OpticalCalibrationDimensions {
  activeWidth: number;
  activeHeight: number;
  fftWidth: number;
  fftHeight: number;
}

export interface OpticalFirstNullResolution {
  xUm: number;
  yUm: number;
}

export interface OpticalTrapResolutionPair {
  firstIndex: number;
  secondIndex: number;
  correlation: number;
  distanceUm: number;
}

export interface OpticalTrapResolutionAnalysis {
  maximumModeCorrelation: number;
  unresolvedPairCount: number;
  worstPair: OpticalTrapResolutionPair | null;
}

export function createOpticalCalibration(
  dimensions: OpticalCalibrationDimensions,
  input: OpticalCalibrationInput,
  calibrationId: string,
): CalibrationPackage {
  validateOpticalCalibration(input);
  const wavelengthUm = input.wavelengthNm / 1000;
  const focalLengthUm = input.focalLengthMm * 1000;
  const xBinsPerUm = dimensions.fftWidth * input.pixelPitchUm / (wavelengthUm * focalLengthUm);
  const yBinsPerUm = dimensions.fftHeight * input.pixelPitchUm / (wavelengthUm * focalLengthUm);
  const halfFieldOfViewUm = wavelengthUm * focalLengthUm / (2 * input.pixelPitchUm);
  return {
    manifest: {
      calibrationId,
      wavelengthNm: input.wavelengthNm,
      focalLengthMm: input.focalLengthMm,
      pixelPitchUm: input.pixelPitchUm,
      activeWidth: dimensions.activeWidth,
      activeHeight: dimensions.activeHeight,
      fftWidth: dimensions.fftWidth,
      fftHeight: dimensions.fftHeight,
      fieldOfViewUm: {
        xMinUm: -halfFieldOfViewUm,
        xMaxUm: halfFieldOfViewUm,
        yMinUm: -halfFieldOfViewUm,
        yMaxUm: halfFieldOfViewUm,
      },
      coordinateConvention: "Fraunhofer focal plane: +x right, +y up; raw unshifted DFT storage",
      phaseConvention: input.phaseResponseLut ? inferPhaseConvention(input.phaseResponseLut) : "NEGATIVE_PI_TO_PI",
    },
    coordinateTransform: {
      a: xBinsPerUm,
      b: 0,
      c: 0,
      d: -yBinsPerUm,
      offsetX: 0,
      offsetY: 0,
      fftCoordinateSpace: "SIGNED_FREQUENCY",
    },
    ...(input.phaseResponseLut ? { phaseResponseLut: Float64Array.from(input.phaseResponseLut) } : {}),
  };
}

export function validateOpticalCalibration(input: OpticalCalibrationInput): void {
  positiveFinite(input.wavelengthNm, "Wavelength");
  positiveFinite(input.focalLengthMm, "Focal length");
  positiveFinite(input.pixelPitchUm, "SLM pixel pitch");
  if (input.phaseResponseLut !== undefined) {
    if (input.phaseResponseLut.length < 2) throw new Error("The measured phase-response LUT needs at least two values");
    input.phaseResponseLut.forEach((value, index) => {
      if (!Number.isFinite(value)) throw new Error(`Phase-response LUT value ${index + 1} is not finite`);
    });
    const increasing = input.phaseResponseLut[0]! <= input.phaseResponseLut[input.phaseResponseLut.length - 1]!;
    for (let index = 1; index < input.phaseResponseLut.length; index += 1) {
      if (increasing ? input.phaseResponseLut[index]! < input.phaseResponseLut[index - 1]!
        : input.phaseResponseLut[index]! > input.phaseResponseLut[index - 1]!) {
        throw new Error("The measured phase-response LUT must be monotonic in display-code order");
      }
    }
  }
}

export function parsePhaseResponseLut(source: string): number[] {
  const trimmed = source.trim();
  if (!trimmed) throw new Error("The phase-response LUT file is empty");
  let candidate: unknown;
  try {
    candidate = JSON.parse(trimmed);
  } catch {
    candidate = trimmed.split(/[\s,;]+/).filter(Boolean).map(Number);
  }
  if (candidate && typeof candidate === "object" && !Array.isArray(candidate)) {
    candidate = (candidate as { phaseResponseLut?: unknown }).phaseResponseLut;
  }
  if (!Array.isArray(candidate)) throw new Error("Use a JSON number array, a phaseResponseLut object, CSV, or whitespace-separated phases");
  const values = candidate.map(Number);
  const input: OpticalCalibrationInput = {
    wavelengthNm: DEFAULT_WAVELENGTH_NM,
    focalLengthMm: DEFAULT_FOCAL_LENGTH_MM,
    pixelPitchUm: DEFAULT_PIXEL_PITCH_UM,
    phaseResponseLut: values,
  };
  validateOpticalCalibration(input);
  return values;
}

export function opticalFieldOfViewUm(input: OpticalCalibrationInput): number {
  validateOpticalCalibration(input);
  return (input.wavelengthNm / 1000) * (input.focalLengthMm * 1000) / input.pixelPitchUm;
}

/** First zero of the ideal rectangular active-aperture point-spread function. */
export function opticalFirstNullResolutionUm(
  dimensions: Pick<OpticalCalibrationDimensions, "activeWidth" | "activeHeight">,
  input: OpticalCalibrationInput,
): OpticalFirstNullResolution {
  validateOpticalCalibration(input);
  positiveFinite(dimensions.activeWidth, "Active SLM width");
  positiveFinite(dimensions.activeHeight, "Active SLM height");
  const wavelengthUm = input.wavelengthNm / 1000;
  const focalLengthUm = input.focalLengthMm * 1000;
  return {
    xUm: wavelengthUm * focalLengthUm / (dimensions.activeWidth * input.pixelPitchUm),
    yUm: wavelengthUm * focalLengthUm / (dimensions.activeHeight * input.pixelPitchUm),
  };
}

/**
 * Detect target pairs whose ideal rectangular-aperture steering vectors are
 * almost the same. Such pairs cannot be certified as independent traps.
 */
export function analyzeOpticalTrapResolution(
  points: readonly Point2D[],
  calibration: CalibrationPackage,
  width: number,
  height: number,
  unresolvedCorrelation = 0.98,
): OpticalTrapResolutionAnalysis {
  if (!Number.isFinite(unresolvedCorrelation) || unresolvedCorrelation <= 0 || unresolvedCorrelation >= 1) {
    throw new Error("Unresolved-mode correlation must be between zero and one");
  }
  const mapped = points.map((point) => mapPhysicalPointToDftFrequency(point, calibration, width, height));
  const activeWidth = calibration.manifest.activeWidth;
  const activeHeight = calibration.manifest.activeHeight;
  let maximumModeCorrelation = 0;
  let unresolvedPairCount = 0;
  let worstPair: OpticalTrapResolutionPair | null = null;
  for (let firstIndex = 0; firstIndex < mapped.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < mapped.length; secondIndex += 1) {
      const first = mapped[firstIndex]!;
      const second = mapped[secondIndex]!;
      const correlation = rectangularModeCorrelation(
        first.x - second.x,
        first.y - second.y,
        activeWidth,
        activeHeight,
        width,
        height,
      );
      if (correlation >= unresolvedCorrelation) unresolvedPairCount += 1;
      if (correlation > maximumModeCorrelation) {
        maximumModeCorrelation = correlation;
        worstPair = {
          firstIndex,
          secondIndex,
          correlation,
          distanceUm: Math.hypot(
            points[firstIndex]!.xUm - points[secondIndex]!.xUm,
            points[firstIndex]!.yUm - points[secondIndex]!.yUm,
          ),
        };
      }
    }
  }
  return { maximumModeCorrelation, unresolvedPairCount, worstPair };
}

function inferPhaseConvention(lut: readonly number[]): "NEGATIVE_PI_TO_PI" | "ZERO_TO_TWO_PI" {
  return lut[0]! >= -1e-9 && lut[lut.length - 1]! > Math.PI ? "ZERO_TO_TWO_PI" : "NEGATIVE_PI_TO_PI";
}

function rectangularModeCorrelation(
  deltaX: number,
  deltaY: number,
  activeWidth: number,
  activeHeight: number,
  width: number,
  height: number,
): number {
  return dirichletMagnitude(deltaX, activeWidth, width) * dirichletMagnitude(deltaY, activeHeight, height);
}

function dirichletMagnitude(delta: number, activeExtent: number, transformExtent: number): number {
  if (Math.abs(delta) < 1e-12) return 1;
  const denominator = activeExtent * Math.sin(Math.PI * delta / transformExtent);
  if (Math.abs(denominator) < 1e-12) return 1;
  return Math.min(1, Math.abs(Math.sin(Math.PI * activeExtent * delta / transformExtent) / denominator));
}

function positiveFinite(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive finite number`);
}
