import { createComplexField, fft2d, wrapPhase } from "../../../src/index.js";
import {
  createForwardPhaseTable,
  finalizeForwardIntensity,
  gaussianFieldAmplitudeAt,
  simulateSlmFrameWasm,
} from "./forward-simulation.js";
import {
  opticalFieldOfViewUm,
  validateOpticalCalibration,
  type OpticalCalibrationInput,
} from "./optical-calibration.js";

const TAU = Math.PI * 2;

export interface ContinuousFieldInput {
  targetIntensity: Float32Array;
  targetWidth: number;
  targetHeight: number;
  slmWidth: number;
  slmHeight: number;
  fftWidth: number;
  fftHeight: number;
  fieldWidthUm: number;
  fieldHeightUm: number;
  fieldCenterXUm: number;
  fieldCenterYUm: number;
  iterations: number;
  mixingFactor: number;
  deterministicSeed: number;
  opticalCalibration: OpticalCalibrationInput;
}

export interface ContinuousFieldRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ContinuousFieldMetrics {
  iterations: number;
  normalizedIntensityRmse: number;
  intensityCorrelation: number;
  diffractionEfficiency: number;
  outsideSignalPowerFraction: number;
  brightRegionSpeckleContrast: number;
  numericalValid: boolean;
  accepted: boolean;
  flags: string[];
  solveTimeMs: number;
}

export interface ContinuousFieldResult {
  pixels: Uint8Array;
  intensity: Float32Array;
  metrics: ContinuousFieldMetrics;
  targetRegion: ContinuousFieldRegion;
  backendId: string;
}

export interface ContinuousTargetGrid {
  amplitude: Float32Array;
  intensityShifted: Float32Array;
  signalMask: Uint32Array;
  targetPower: number;
  region: ContinuousFieldRegion;
}

export const DEFAULT_CONTINUOUS_FIELD_ITERATIONS = 24;
export const DEFAULT_CONTINUOUS_FIELD_MIXING = 0.4;
export const CONTINUOUS_FIELD_ACCEPTED_RMSE = 0.25;

export function createContinuousInversePhaseLut(
  phaseResponseLut?: readonly number[],
  size = 4096,
): Float32Array {
  if (!Number.isSafeInteger(size) || size < 2) throw new Error("Continuous inverse LUT size must be at least two");
  const phaseByCode = createForwardPhaseTable(phaseResponseLut);
  return Float32Array.from({ length: size }, (_, index) => {
    const phase = index / (size - 1) * TAU - Math.PI;
    if (!phaseResponseLut) return index / (size - 1) * 255;
    const first = phaseResponseLut[0]!;
    const last = phaseResponseLut[phaseResponseLut.length - 1]!;
    const usesPositivePhase = Math.min(first, last) >= -1e-9 && Math.max(first, last) > Math.PI;
    const target = usesPositivePhase && phase < 0
      ? phase + TAU
      : phase;
    let best = 0;
    let bestError = Number.POSITIVE_INFINITY;
    for (let code = 0; code < phaseByCode.length; code += 1) {
      const error = Math.abs(phaseByCode[code]! - target);
      if (error < bestError) {
        best = code;
        bestError = error;
      }
    }
    return best;
  });
}

export function validateContinuousFieldInput(input: ContinuousFieldInput): void {
  const integerDimensions = [
    input.targetWidth,
    input.targetHeight,
    input.slmWidth,
    input.slmHeight,
    input.fftWidth,
    input.fftHeight,
  ];
  if (!integerDimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Continuous-field dimensions must be positive integers");
  }
  if (input.targetIntensity.length !== input.targetWidth * input.targetHeight) {
    throw new Error("Continuous target pixels do not match its dimensions");
  }
  if (input.slmWidth > input.fftWidth || input.slmHeight > input.fftHeight) {
    throw new Error("The active SLM must fit inside the continuous-field FFT grid");
  }
  if (!isPowerOfTwo(input.fftWidth) || !isPowerOfTwo(input.fftHeight)) {
    throw new Error("Continuous-field FFT dimensions must be powers of two");
  }
  if (![input.fieldWidthUm, input.fieldHeightUm].every((value) => Number.isFinite(value) && value > 0)) {
    throw new Error("Continuous target field width and height must be positive");
  }
  if (![input.fieldCenterXUm, input.fieldCenterYUm].every(Number.isFinite)) {
    throw new Error("Continuous target field centre must be finite");
  }
  if (!Number.isSafeInteger(input.iterations) || input.iterations < 1 || input.iterations > 200) {
    throw new Error("Continuous-field iterations must be an integer from 1 to 200");
  }
  if (!Number.isFinite(input.mixingFactor) || input.mixingFactor <= 0 || input.mixingFactor > 1) {
    throw new Error("MRAF mixing must be greater than zero and no more than one");
  }
  if (!Number.isSafeInteger(input.deterministicSeed) || input.deterministicSeed < 0 || input.deterministicSeed > 0xffff_ffff) {
    throw new Error("Continuous-field seed must be an integer from 0 to 4294967295");
  }
  if (input.targetIntensity.some((value) => !Number.isFinite(value) || value < 0 || value > 1)) {
    throw new Error("Continuous target intensity must stay between zero and one");
  }
  validateOpticalCalibration(input.opticalCalibration);
  const fieldOfView = opticalFieldOfViewUm(input.opticalCalibration);
  if (input.fieldWidthUm > fieldOfView || input.fieldHeightUm > fieldOfView) {
    throw new Error(`Continuous target field must fit inside the ${fieldOfView.toFixed(3)} µm optical field of view`);
  }
}

export function buildContinuousTargetGrid(input: ContinuousFieldInput): ContinuousTargetGrid {
  validateContinuousFieldInput(input);
  const fieldOfView = opticalFieldOfViewUm(input.opticalCalibration);
  const regionWidth = Math.max(1, Math.min(input.fftWidth, Math.round(input.fieldWidthUm / fieldOfView * input.fftWidth)));
  const regionHeight = Math.max(1, Math.min(input.fftHeight, Math.round(input.fieldHeightUm / fieldOfView * input.fftHeight)));
  const centerX = input.fftWidth / 2 + input.fieldCenterXUm / fieldOfView * input.fftWidth;
  const centerY = input.fftHeight / 2 - input.fieldCenterYUm / fieldOfView * input.fftHeight;
  const region: ContinuousFieldRegion = {
    x: Math.round(centerX - regionWidth / 2),
    y: Math.round(centerY - regionHeight / 2),
    width: regionWidth,
    height: regionHeight,
  };
  if (region.x < 0 || region.y < 0 || region.x + region.width > input.fftWidth || region.y + region.height > input.fftHeight) {
    throw new Error("Continuous target field and centre extend beyond the optical field of view");
  }

  const pixelCount = input.fftWidth * input.fftHeight;
  const amplitude = new Float32Array(pixelCount);
  const intensityShifted = new Float32Array(pixelCount);
  const signalMask = new Uint32Array(pixelCount);
  let targetPower = 0;
  const halfWidth = input.fftWidth >>> 1;
  const halfHeight = input.fftHeight >>> 1;
  for (let localY = 0; localY < region.height; localY += 1) {
    const shiftedY = region.y + localY;
    const unshiftedY = (shiftedY + halfHeight) % input.fftHeight;
    const sourceY = (localY + 0.5) / region.height * input.targetHeight - 0.5;
    for (let localX = 0; localX < region.width; localX += 1) {
      const shiftedX = region.x + localX;
      const unshiftedX = (shiftedX + halfWidth) % input.fftWidth;
      const sourceX = (localX + 0.5) / region.width * input.targetWidth - 0.5;
      const value = bilinearSample(
        input.targetIntensity,
        input.targetWidth,
        input.targetHeight,
        sourceX,
        sourceY,
      );
      const unshiftedIndex = unshiftedY * input.fftWidth + unshiftedX;
      const shiftedIndex = shiftedY * input.fftWidth + shiftedX;
      amplitude[unshiftedIndex] = Math.sqrt(value);
      intensityShifted[shiftedIndex] = value;
      signalMask[unshiftedIndex] = 1;
      targetPower += value;
    }
  }
  if (!(targetPower > 1e-12)) throw new Error("Draw or upload a non-empty continuous target before generating");
  return { amplitude, intensityShifted, signalMask, targetPower, region };
}

/** CPU/Wasm MRAF solver. FFTs execute in the Rust WebAssembly core. */
export function solveContinuousFieldWasm(input: ContinuousFieldInput): ContinuousFieldResult {
  validateContinuousFieldInput(input);
  const started = nowMs();
  const target = buildContinuousTargetGrid(input);
  const pixelCount = input.fftWidth * input.fftHeight;
  const phase = initialContinuousPhase(input, target);
  const field = createComplexField(input.fftWidth, input.fftHeight);
  const xStart = Math.floor((input.fftWidth - input.slmWidth) / 2);
  const yStart = Math.floor((input.fftHeight - input.slmHeight) / 2);
  const targetScale = continuousTargetScale(input, target.targetPower);

  for (let iteration = 0; iteration < input.iterations; iteration += 1) {
    writeSpatialField(field.real, field.imag, phase, input, xStart, yStart);
    fft2d(field, false);
    for (let index = 0; index < pixelCount; index += 1) {
      const real = field.real[index]!;
      const imag = field.imag[index]!;
      const magnitude = Math.hypot(real, imag);
      const constrainedAmplitude = target.signalMask[index] !== 0
        ? input.mixingFactor * targetScale * target.amplitude[index]!
        : (1 - input.mixingFactor) * magnitude;
      if (magnitude > 1e-20) {
        field.real[index] = constrainedAmplitude * real / magnitude;
        field.imag[index] = constrainedAmplitude * imag / magnitude;
      } else {
        field.real[index] = constrainedAmplitude;
        field.imag[index] = 0;
      }
    }
    fft2d(field, true);
    for (let index = 0; index < pixelCount; index += 1) {
      if (field.real[index] !== 0 || field.imag[index] !== 0) {
        phase[index] = Math.atan2(field.imag[index]!, field.real[index]!);
      }
    }
  }

  const pixels = quantizeActivePhase(phase, input, xStart, yStart);
  const propagated = simulateSlmFrameWasm({
    pixels,
    width: input.slmWidth,
    height: input.slmHeight,
    fftWidth: input.fftWidth,
    fftHeight: input.fftHeight,
    pixelPitchUm: input.opticalCalibration.pixelPitchUm,
    ...(input.opticalCalibration.incidentBeam
      ? { incidentBeam: { ...input.opticalCalibration.incidentBeam } }
      : {}),
    ...(input.opticalCalibration.phaseResponseLut
      ? { phaseResponseLut: input.opticalCalibration.phaseResponseLut }
      : {}),
  });
  return finalizeContinuousFieldResult(
    input,
    target,
    pixels,
    propagated.intensity,
    "wasm-mraf-radix2",
    nowMs() - started,
  );
}

/**
 * Parseval-consistent scale for the unnormalised forward FFT. Keeping this
 * value fixed makes the CPU/Wasm and GPU MRAF constraints identical.
 */
export function continuousTargetScale(input: ContinuousFieldInput, targetPower: number): number {
  let inputPower = 0;
  for (let y = 0; y < input.slmHeight; y += 1) {
    for (let x = 0; x < input.slmWidth; x += 1) {
      const amplitude = gaussianFieldAmplitudeAt(
        x,
        y,
        input.slmWidth,
        input.slmHeight,
        input.opticalCalibration.pixelPitchUm,
        input.opticalCalibration.incidentBeam,
      );
      inputPower += amplitude * amplitude;
    }
  }
  return Math.sqrt(input.fftWidth * input.fftHeight * inputPower / Math.max(targetPower, 1e-20));
}

export function finalizeContinuousFieldResult(
  input: ContinuousFieldInput,
  target: ContinuousTargetGrid,
  pixels: Uint8Array,
  normalizedIntensity: Float32Array,
  backendId: string,
  solveTimeMs: number,
): ContinuousFieldResult {
  if (pixels.length !== input.slmWidth * input.slmHeight) throw new Error("Continuous-field output dimensions are inconsistent");
  if (normalizedIntensity.length !== input.fftWidth * input.fftHeight) throw new Error("Continuous-field reconstruction dimensions are inconsistent");
  // Reuse the forward validator's strict finite/non-zero checks without
  // changing the already normalized array.
  const checked = finalizeForwardIntensity({
    pixels,
    width: input.slmWidth,
    height: input.slmHeight,
    fftWidth: input.fftWidth,
    fftHeight: input.fftHeight,
  }, new Float32Array(normalizedIntensity), backendId);
  const intensity = checked.intensity;
  let targetSquared = 0;
  let targetMeasured = 0;
  let totalPower = 0;
  let signalPower = 0;
  for (let index = 0; index < intensity.length; index += 1) {
    const measured = intensity[index]!;
    const desired = target.intensityShifted[index]!;
    totalPower += measured;
    if (insideRegion(index, input.fftWidth, target.region)) signalPower += measured;
    targetSquared += desired * desired;
    targetMeasured += desired * measured;
  }
  const scale = targetSquared > 1e-20 ? targetMeasured / targetSquared : 1;
  let squaredError = 0;
  let scaledTargetPower = 0;
  const targetValues: number[] = [];
  const measuredValues: number[] = [];
  const brightRatios: number[] = [];
  for (let y = target.region.y; y < target.region.y + target.region.height; y += 1) {
    for (let x = target.region.x; x < target.region.x + target.region.width; x += 1) {
      const index = y * input.fftWidth + x;
      const desired = target.intensityShifted[index]!;
      const expected = scale * desired;
      const measured = intensity[index]!;
      squaredError += (measured - expected) ** 2;
      scaledTargetPower += expected ** 2;
      targetValues.push(desired);
      measuredValues.push(measured);
      if (desired >= 0.1) brightRatios.push(measured / Math.max(expected, 1e-12));
    }
  }
  const normalizedIntensityRmse = Math.sqrt(squaredError / Math.max(scaledTargetPower, 1e-20));
  const intensityCorrelation = pearsonCorrelation(targetValues, measuredValues);
  const brightMean = meanOf(brightRatios);
  const brightStd = standardDeviation(brightRatios, brightMean);
  const diffractionEfficiency = totalPower > 0 ? signalPower / totalPower : 0;
  const outsideSignalPowerFraction = totalPower > 0 ? Math.max(0, totalPower - signalPower) / totalPower : 1;
  const brightRegionSpeckleContrast = brightMean > 0 ? brightStd / brightMean : 0;
  const numericalValues = [
    normalizedIntensityRmse,
    intensityCorrelation,
    diffractionEfficiency,
    outsideSignalPowerFraction,
    brightRegionSpeckleContrast,
    solveTimeMs,
  ];
  const numericalValid = numericalValues.every(Number.isFinite);
  const flags: string[] = [];
  if (!numericalValid) flags.push("NUMERIC_ERROR");
  if (normalizedIntensityRmse > CONTINUOUS_FIELD_ACCEPTED_RMSE) flags.push("TARGET_RMSE_HIGH");
  return {
    pixels,
    intensity,
    targetRegion: target.region,
    backendId,
    metrics: {
      iterations: input.iterations,
      normalizedIntensityRmse,
      intensityCorrelation,
      diffractionEfficiency,
      outsideSignalPowerFraction,
      brightRegionSpeckleContrast,
      numericalValid,
      accepted: numericalValid && normalizedIntensityRmse <= CONTINUOUS_FIELD_ACCEPTED_RMSE,
      flags,
      solveTimeMs,
    },
  };
}

function initialContinuousPhase(input: ContinuousFieldInput, target: ContinuousTargetGrid): Float64Array {
  const seedPhase = deterministicPhase(0, input.deterministicSeed);
  const field = createComplexField(input.fftWidth, input.fftHeight);
  const halfWidth = input.fftWidth / 2;
  const halfHeight = input.fftHeight / 2;
  for (let index = 0; index < field.real.length; index += 1) {
    const amplitude = target.amplitude[index]!;
    if (amplitude === 0) continue;
    const x = index % input.fftWidth;
    const y = Math.floor(index / input.fftWidth);
    const signedX = x < halfWidth ? x : x - input.fftWidth;
    const signedY = y < halfHeight ? y : y - input.fftHeight;
    const normalizedX = signedX / Math.max(1, target.region.width);
    const normalizedY = signedY / Math.max(1, target.region.height);
    const angle = wrapPhase(seedPhase + TAU * 6 * (normalizedX * normalizedX + normalizedY * normalizedY));
    field.real[index] = amplitude * Math.cos(angle);
    field.imag[index] = amplitude * Math.sin(angle);
  }
  fft2d(field, true);
  const phase = new Float64Array(field.real.length);
  for (let index = 0; index < phase.length; index += 1) {
    phase[index] = field.real[index] !== 0 || field.imag[index] !== 0
      ? Math.atan2(field.imag[index]!, field.real[index]!)
      : deterministicPhase(index, input.deterministicSeed);
  }
  return phase;
}

function writeSpatialField(
  real: Float64Array,
  imag: Float64Array,
  phase: Float64Array,
  input: ContinuousFieldInput,
  xStart: number,
  yStart: number,
): void {
  real.fill(0);
  imag.fill(0);
  for (let activeY = 0; activeY < input.slmHeight; activeY += 1) {
    const y = yStart + activeY;
    for (let activeX = 0; activeX < input.slmWidth; activeX += 1) {
      const x = xStart + activeX;
      const index = y * input.fftWidth + x;
      const amplitude = gaussianFieldAmplitudeAt(
        activeX,
        activeY,
        input.slmWidth,
        input.slmHeight,
        input.opticalCalibration.pixelPitchUm,
        input.opticalCalibration.incidentBeam,
      );
      real[index] = amplitude * Math.cos(phase[index]!);
      imag[index] = amplitude * Math.sin(phase[index]!);
    }
  }
}

function quantizeActivePhase(
  phase: Float64Array,
  input: ContinuousFieldInput,
  xStart: number,
  yStart: number,
): Uint8Array {
  const pixels = new Uint8Array(input.slmWidth * input.slmHeight);
  const inverseLut = createContinuousInversePhaseLut(input.opticalCalibration.phaseResponseLut);
  for (let activeY = 0; activeY < input.slmHeight; activeY += 1) {
    const source = (yStart + activeY) * input.fftWidth + xStart;
    for (let activeX = 0; activeX < input.slmWidth; activeX += 1) {
      const wrapped = wrapPhase(phase[source + activeX]!);
      const position = (wrapped + Math.PI) / TAU * (inverseLut.length - 1);
      const low = Math.floor(position);
      const high = Math.min(inverseLut.length - 1, low + 1);
      const fraction = position - low;
      pixels[activeY * input.slmWidth + activeX] = Math.round(clamp(
        inverseLut[low]! * (1 - fraction) + inverseLut[high]! * fraction,
        0,
        255,
      ));
    }
  }
  return pixels;
}

function bilinearSample(values: Float32Array, width: number, height: number, x: number, y: number): number {
  const x0 = clamp(Math.floor(x), 0, width - 1);
  const y0 = clamp(Math.floor(y), 0, height - 1);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const tx = clamp(x - x0, 0, 1);
  const ty = clamp(y - y0, 0, 1);
  const top = values[y0 * width + x0]! * (1 - tx) + values[y0 * width + x1]! * tx;
  const bottom = values[y1 * width + x0]! * (1 - tx) + values[y1 * width + x1]! * tx;
  return clamp(top * (1 - ty) + bottom * ty, 0, 1);
}

function insideRegion(index: number, width: number, region: ContinuousFieldRegion): boolean {
  const x = index % width;
  const y = Math.floor(index / width);
  return x >= region.x && x < region.x + region.width && y >= region.y && y < region.y + region.height;
}

function pearsonCorrelation(left: number[], right: number[]): number {
  if (left.length === 0 || left.length !== right.length) return 0;
  const leftMean = meanOf(left);
  const rightMean = meanOf(right);
  let numerator = 0;
  let leftPower = 0;
  let rightPower = 0;
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index]! - leftMean;
    const b = right[index]! - rightMean;
    numerator += a * b;
    leftPower += a * a;
    rightPower += b * b;
  }
  return leftPower > 0 && rightPower > 0 ? numerator / Math.sqrt(leftPower * rightPower) : 0;
}

function meanOf(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[], mean: number): number {
  return values.length === 0 ? 0 : Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

function deterministicPhase(index: number, seed: number): number {
  let value = (((index >>> 0) ^ (seed >>> 0)) + 1) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
  value = (value ^ (value >>> 16)) >>> 0;
  return (value >>> 8) / 16_777_216 * TAU - Math.PI;
}

function isPowerOfTwo(value: number): boolean {
  return (value & (value - 1)) === 0;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
