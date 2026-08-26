import {
  mapPhysicalPointToDftFrequency,
  type CalibrationPackage,
  type Point2D,
} from "../../../src/index.js";

/** Hardware preset measured from the current Hamamatsu X15213-05 experiment. */
export const DEFAULT_WAVELENGTH_NM = 407;
export const DEFAULT_FOCAL_LENGTH_MM = 100;
export const DEFAULT_PIXEL_PITCH_UM = 12.5;
export const DEFAULT_INCIDENT_BEAM_DIAMETER_MM = 8;
export const DEFAULT_TWO_PI_SIGNAL_LEVEL = 217;

export interface GaussianIncidentBeamInput {
  profile: "GAUSSIAN";
  /** Full 1/e^2 intensity diameter at the SLM plane. */
  diameterXMm: number;
  /** Full 1/e^2 intensity diameter at the SLM plane. */
  diameterYMm: number;
  centerXMm?: number;
  centerYMm?: number;
}

export interface OpticalCalibrationInput {
  wavelengthNm: number;
  focalLengthMm: number;
  pixelPitchUm: number;
  incidentBeam?: GaussianIncidentBeamInput;
  phaseResponseLut?: number[];
}

export interface OpticalCalibrationCreationOptions {
  includeIncidentAmplitude?: boolean;
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
  options: OpticalCalibrationCreationOptions = {},
): CalibrationPackage {
  validateOpticalCalibration(input);
  const wavelengthUm = input.wavelengthNm / 1000;
  const focalLengthUm = input.focalLengthMm * 1000;
  const xBinsPerUm = dimensions.fftWidth * input.pixelPitchUm / (wavelengthUm * focalLengthUm);
  const yBinsPerUm = dimensions.fftHeight * input.pixelPitchUm / (wavelengthUm * focalLengthUm);
  const halfFieldOfViewUm = wavelengthUm * focalLengthUm / (2 * input.pixelPitchUm);
  const incidentAmplitude = options.includeIncidentAmplitude === false
    ? undefined
    : createGaussianIncidentAmplitude(dimensions, input);
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
    ...(incidentAmplitude ? { incidentAmplitude } : {}),
    ...(input.phaseResponseLut ? { phaseResponseLut: Float64Array.from(input.phaseResponseLut) } : {}),
  };
}

export function validateOpticalCalibration(input: OpticalCalibrationInput): void {
  positiveFinite(input.wavelengthNm, "Wavelength");
  positiveFinite(input.focalLengthMm, "Focal length");
  positiveFinite(input.pixelPitchUm, "SLM pixel pitch");
  if (input.incidentBeam !== undefined) {
    if (input.incidentBeam.profile !== "GAUSSIAN") throw new Error("Only Gaussian incident beams are supported");
    positiveFinite(input.incidentBeam.diameterXMm, "Incident beam X diameter");
    positiveFinite(input.incidentBeam.diameterYMm, "Incident beam Y diameter");
    finite(input.incidentBeam.centerXMm ?? 0, "Incident beam X centre");
    finite(input.incidentBeam.centerYMm ?? 0, "Incident beam Y centre");
  }
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

/**
 * Build the real, positive field amplitude of a Gaussian beam at the active
 * SLM plane. The supplied diameters are full 1/e^2 intensity diameters, so the
 * field amplitude is exp(-4 r^2 / D^2).
 */
export function createGaussianIncidentAmplitude(
  dimensions: Pick<OpticalCalibrationDimensions, "activeWidth" | "activeHeight">,
  input: OpticalCalibrationInput,
): Float64Array | undefined {
  validateOpticalCalibration(input);
  const beam = input.incidentBeam;
  if (!beam) return undefined;
  positiveFinite(dimensions.activeWidth, "Active SLM width");
  positiveFinite(dimensions.activeHeight, "Active SLM height");
  const diameterXUm = beam.diameterXMm * 1000;
  const diameterYUm = beam.diameterYMm * 1000;
  const centerXUm = (beam.centerXMm ?? 0) * 1000;
  const centerYUm = (beam.centerYMm ?? 0) * 1000;
  const centerPixelX = (dimensions.activeWidth - 1) / 2;
  const centerPixelY = (dimensions.activeHeight - 1) / 2;
  const result = new Float64Array(dimensions.activeWidth * dimensions.activeHeight);
  for (let y = 0; y < dimensions.activeHeight; y += 1) {
    const yUm = (y - centerPixelY) * input.pixelPitchUm - centerYUm;
    const normalizedY = yUm / diameterYUm;
    for (let x = 0; x < dimensions.activeWidth; x += 1) {
      const xUm = (x - centerPixelX) * input.pixelPitchUm - centerXUm;
      const normalizedX = xUm / diameterXUm;
      result[y * dimensions.activeWidth + x] = Math.exp(-4 * (
        normalizedX * normalizedX + normalizedY * normalizedY
      ));
    }
  }
  return result;
}

/** Linear device response whose indicated display code produces exactly 2pi. */
export function phaseResponseForTwoPiSignalLevel(signalLevel: number): number[] {
  if (!Number.isInteger(signalLevel) || signalLevel < 1 || signalLevel > 255) {
    throw new Error("The 2pi signal level must be an integer between 1 and 255");
  }
  return Array.from({ length: 256 }, (_, code) => code / signalLevel * 2 * Math.PI);
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
 * Conservative focal-spot scale for target spacing. A Gaussian pupil has no
 * first zero, so its 1/e^2 focal-intensity diameter is used and combined with
 * the finite rectangular active-aperture limit.
 */
export function opticalEffectiveResolutionUm(
  dimensions: Pick<OpticalCalibrationDimensions, "activeWidth" | "activeHeight">,
  input: OpticalCalibrationInput,
): OpticalFirstNullResolution {
  const aperture = opticalFirstNullResolutionUm(dimensions, input);
  const beam = input.incidentBeam;
  if (!beam) return aperture;
  const wavelengthUm = input.wavelengthNm / 1000;
  const focalLengthUm = input.focalLengthMm * 1000;
  const gaussianX = 4 * wavelengthUm * focalLengthUm / (Math.PI * beam.diameterXMm * 1000);
  const gaussianY = 4 * wavelengthUm * focalLengthUm / (Math.PI * beam.diameterYMm * 1000);
  return {
    xUm: Math.max(aperture.xUm, gaussianX),
    yUm: Math.max(aperture.yUm, gaussianY),
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
  opticalInput?: OpticalCalibrationInput,
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
      const rectangularCorrelation = rectangularModeCorrelation(
        first.x - second.x,
        first.y - second.y,
        activeWidth,
        activeHeight,
        width,
        height,
      );
      const correlation = opticalInput?.incidentBeam
        ? Math.max(rectangularCorrelation, gaussianModeCorrelation(
          points[firstIndex]!,
          points[secondIndex]!,
          opticalInput,
        ))
        : rectangularCorrelation;
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

function gaussianModeCorrelation(first: Point2D, second: Point2D, input: OpticalCalibrationInput): number {
  const beam = input.incidentBeam;
  if (!beam) return 0;
  const wavelengthUm = input.wavelengthNm / 1000;
  const focalLengthUm = input.focalLengthMm * 1000;
  const radiusXUm = beam.diameterXMm * 500;
  const radiusYUm = beam.diameterYMm * 500;
  const normalizedX = Math.PI * radiusXUm * (first.xUm - second.xUm) / (wavelengthUm * focalLengthUm);
  const normalizedY = Math.PI * radiusYUm * (first.yUm - second.yUm) / (wavelengthUm * focalLengthUm);
  return Math.exp(-0.5 * (normalizedX * normalizedX + normalizedY * normalizedY));
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

function finite(value: number, name: string): void {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
}
