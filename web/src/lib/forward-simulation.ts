import { createComplexField, fft2d } from "../../../src/index.js";

const TAU = Math.PI * 2;

export interface ForwardSimulationInput {
  pixels: Uint8Array;
  width: number;
  height: number;
  fftWidth: number;
  fftHeight: number;
  phaseResponseLut?: readonly number[];
}

export interface ForwardSimulationRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface SignedDftFrequency {
  x: number;
  y: number;
}

export interface ForwardSimulationMetrics {
  activeWidth: number;
  activeHeight: number;
  fftWidth: number;
  fftHeight: number;
  maximumIntensity: number;
  meanIntensity: number;
  peakToMeanRatio: number;
  zeroOrderRelativeIntensity: number;
  peakX: number;
  peakY: number;
  peakOffsetX: number;
  peakOffsetY: number;
}

export interface ForwardSimulationResult {
  /** FFT-shifted intensity normalized so its largest value is one. */
  intensity: Float32Array;
  metrics: ForwardSimulationMetrics;
  backendId: string;
}

export function validateForwardSimulationInput(input: ForwardSimulationInput): void {
  const dimensions = [input.width, input.height, input.fftWidth, input.fftHeight];
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new Error("Forward-simulation dimensions must be positive integers");
  }
  if (input.pixels.length !== input.width * input.height) {
    throw new Error(`SLM frame contains ${input.pixels.length} pixels; expected ${input.width} × ${input.height}`);
  }
  if (!isPowerOfTwo(input.fftWidth) || !isPowerOfTwo(input.fftHeight)) {
    throw new Error("Forward simulation requires power-of-two FFT dimensions");
  }
  if (input.width > input.fftWidth || input.height > input.fftHeight) {
    throw new Error("The SLM frame must fit inside the FFT grid");
  }
  if (input.phaseResponseLut !== undefined) validatePhaseResponseLut(input.phaseResponseLut);
}

/** Propagates a phase-code frame with the Rust/Wasm radix-2 FFT core. */
export function simulateSlmFrameWasm(input: ForwardSimulationInput): ForwardSimulationResult {
  validateForwardSimulationInput(input);
  const field = createComplexField(input.fftWidth, input.fftHeight);
  const phaseTable = createForwardPhaseTable(input.phaseResponseLut);
  const xStart = Math.floor((input.fftWidth - input.width) / 2);
  const yStart = Math.floor((input.fftHeight - input.height) / 2);
  for (let y = 0; y < input.height; y += 1) {
    for (let x = 0; x < input.width; x += 1) {
      const source = y * input.width + x;
      const destination = (yStart + y) * input.fftWidth + xStart + x;
      const phase = phaseTable[input.pixels[source]!]!;
      field.real[destination] = Math.cos(phase);
      field.imag[destination] = Math.sin(phase);
    }
  }
  fft2d(field, false);

  const intensity = new Float32Array(input.fftWidth * input.fftHeight);
  const halfWidth = input.fftWidth >>> 1;
  const halfHeight = input.fftHeight >>> 1;
  for (let y = 0; y < input.fftHeight; y += 1) {
    const sourceY = (y + halfHeight) % input.fftHeight;
    for (let x = 0; x < input.fftWidth; x += 1) {
      const sourceX = (x + halfWidth) % input.fftWidth;
      const source = sourceY * input.fftWidth + sourceX;
      intensity[y * input.fftWidth + x] = field.real[source]! ** 2 + field.imag[source]! ** 2;
    }
  }
  return finalizeForwardIntensity(input, intensity, "wasm-radix2-forward-fft");
}

export function decodeForwardPhaseCode(code: number, phaseResponseLut?: readonly number[]): number {
  if (!Number.isFinite(code) || code < 0 || code > 255) throw new Error("SLM phase code must be between 0 and 255");
  if (!phaseResponseLut) return code / 255 * TAU - Math.PI;
  validatePhaseResponseLut(phaseResponseLut);
  const position = code / 255 * (phaseResponseLut.length - 1);
  const low = Math.floor(position);
  const high = Math.min(phaseResponseLut.length - 1, low + 1);
  const fraction = position - low;
  return phaseResponseLut[low]! * (1 - fraction) + phaseResponseLut[high]! * fraction;
}

export function createForwardPhaseTable(phaseResponseLut?: readonly number[]): Float32Array {
  if (phaseResponseLut !== undefined) validatePhaseResponseLut(phaseResponseLut);
  return Float32Array.from({ length: 256 }, (_, code) => {
    if (!phaseResponseLut) return code / 255 * TAU - Math.PI;
    const position = code / 255 * (phaseResponseLut.length - 1);
    const low = Math.floor(position);
    const high = Math.min(phaseResponseLut.length - 1, low + 1);
    const fraction = position - low;
    return phaseResponseLut[low]! * (1 - fraction) + phaseResponseLut[high]! * fraction;
  });
}

/**
 * Build a compact FFT-shifted view around signed target frequencies. Padding
 * includes several aperture first-null widths so the reconstructed spots and
 * their local sidelobes remain visible.
 */
export function targetForwardSimulationRegion(
  frequencies: readonly SignedDftFrequency[],
  fftWidth: number,
  fftHeight: number,
  activeWidth: number,
  activeHeight: number,
): ForwardSimulationRegion | null {
  if (frequencies.length === 0) return null;
  validatePositiveInteger(fftWidth, "FFT width");
  validatePositiveInteger(fftHeight, "FFT height");
  validatePositiveInteger(activeWidth, "Active width");
  validatePositiveInteger(activeHeight, "Active height");
  const shifted = frequencies.map((frequency) => {
    if (!Number.isFinite(frequency.x) || !Number.isFinite(frequency.y)) {
      throw new Error("Target frequency must be finite");
    }
    return {
      x: shiftedForwardCoordinate(frequency.x, fftWidth),
      y: shiftedForwardCoordinate(frequency.y, fftHeight),
    };
  });
  const paddingX = Math.max(4, Math.ceil(4 * fftWidth / activeWidth));
  const paddingY = Math.max(4, Math.ceil(4 * fftHeight / activeHeight));
  const minimumX = Math.floor(Math.min(...shifted.map((point) => point.x)) - paddingX);
  const maximumX = Math.ceil(Math.max(...shifted.map((point) => point.x)) + paddingX);
  const minimumY = Math.floor(Math.min(...shifted.map((point) => point.y)) - paddingY);
  const maximumY = Math.ceil(Math.max(...shifted.map((point) => point.y)) + paddingY);
  const initial = centeredRegion(
    minimumX,
    maximumX,
    minimumY,
    maximumY,
    fftWidth,
    fftHeight,
    16,
    16,
  );
  return constrainPhysicalAspect(initial, fftWidth, fftHeight, 1.5, 3);
}

export function shiftedForwardCoordinate(frequency: number, extent: number): number {
  validatePositiveInteger(extent, "FFT extent");
  if (!Number.isFinite(frequency)) throw new Error("DFT frequency must be finite");
  const shifted = extent / 2 + frequency;
  return ((shifted % extent) + extent) % extent;
}

/** Physical aspect for square SLM pixels; the full FFT field is always square. */
export function forwardSimulationRegionAspect(
  region: ForwardSimulationRegion,
  fftWidth: number,
  fftHeight: number,
): number {
  validateRegion(region, fftWidth, fftHeight);
  return (region.width / fftWidth) / (region.height / fftHeight);
}

function centeredRegion(
  minimumX: number,
  maximumX: number,
  minimumY: number,
  maximumY: number,
  extentWidth: number,
  extentHeight: number,
  minimumWidth: number,
  minimumHeight: number,
): ForwardSimulationRegion {
  const centerX = (minimumX + maximumX) / 2;
  const centerY = (minimumY + maximumY) / 2;
  const width = Math.min(extentWidth, Math.max(minimumWidth, maximumX - minimumX + 1));
  const height = Math.min(extentHeight, Math.max(minimumHeight, maximumY - minimumY + 1));
  return placeRegion(centerX, centerY, width, height, extentWidth, extentHeight);
}

function constrainPhysicalAspect(
  region: ForwardSimulationRegion,
  fftWidth: number,
  fftHeight: number,
  minimumAspect: number,
  maximumAspect: number,
): ForwardSimulationRegion {
  let width = region.width;
  let height = region.height;
  const aspect = forwardSimulationRegionAspect(region, fftWidth, fftHeight);
  if (aspect > maximumAspect) {
    height = Math.min(fftHeight, Math.ceil(width * fftHeight / (maximumAspect * fftWidth)));
  } else if (aspect < minimumAspect) {
    width = Math.min(fftWidth, Math.ceil(minimumAspect * height * fftWidth / fftHeight));
  }
  return placeRegion(
    region.x + (region.width - 1) / 2,
    region.y + (region.height - 1) / 2,
    width,
    height,
    fftWidth,
    fftHeight,
  );
}

function placeRegion(
  centerX: number,
  centerY: number,
  requestedWidth: number,
  requestedHeight: number,
  extentWidth: number,
  extentHeight: number,
): ForwardSimulationRegion {
  const width = Math.max(1, Math.min(extentWidth, Math.ceil(requestedWidth)));
  const height = Math.max(1, Math.min(extentHeight, Math.ceil(requestedHeight)));
  const x = Math.max(0, Math.min(extentWidth - width, Math.round(centerX - (width - 1) / 2)));
  const y = Math.max(0, Math.min(extentHeight - height, Math.round(centerY - (height - 1) / 2)));
  return { x, y, width, height };
}

function validateRegion(region: ForwardSimulationRegion, fftWidth: number, fftHeight: number): void {
  validatePositiveInteger(fftWidth, "FFT width");
  validatePositiveInteger(fftHeight, "FFT height");
  if (![region.x, region.y, region.width, region.height].every(Number.isInteger) ||
      region.x < 0 || region.y < 0 || region.width <= 0 || region.height <= 0 ||
      region.x + region.width > fftWidth || region.y + region.height > fftHeight) {
    throw new Error("Forward-simulation view lies outside the FFT grid");
  }
}

function validatePositiveInteger(value: number, label: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`);
}

function validatePhaseResponseLut(lut: readonly number[]): void {
  if (lut.length < 2 || lut.some((phase) => !Number.isFinite(phase))) {
    throw new Error("Forward-simulation phase-response LUT must contain at least two finite values");
  }
}

export function finalizeForwardIntensity(
  input: ForwardSimulationInput,
  intensity: Float32Array,
  backendId: string,
): ForwardSimulationResult {
  if (intensity.length !== input.fftWidth * input.fftHeight) {
    throw new Error("Forward-simulation intensity dimensions are inconsistent");
  }
  let maximumIntensity = 0;
  let totalIntensity = 0;
  let peakIndex = 0;
  for (let index = 0; index < intensity.length; index += 1) {
    const value = intensity[index]!;
    if (!Number.isFinite(value) || value < 0) throw new Error("Forward simulation produced invalid intensity values");
    totalIntensity += value;
    if (value > maximumIntensity) {
      maximumIntensity = value;
      peakIndex = index;
    }
  }
  if (!(maximumIntensity > 0) || !Number.isFinite(totalIntensity)) {
    throw new Error("Forward simulation produced no optical power");
  }
  for (let index = 0; index < intensity.length; index += 1) intensity[index] = intensity[index]! / maximumIntensity;

  const peakX = peakIndex % input.fftWidth;
  const peakY = Math.floor(peakIndex / input.fftWidth);
  const centerIndex = (input.fftHeight >>> 1) * input.fftWidth + (input.fftWidth >>> 1);
  const meanIntensity = totalIntensity / intensity.length;
  return {
    intensity,
    backendId,
    metrics: {
      activeWidth: input.width,
      activeHeight: input.height,
      fftWidth: input.fftWidth,
      fftHeight: input.fftHeight,
      maximumIntensity,
      meanIntensity,
      peakToMeanRatio: maximumIntensity / meanIntensity,
      zeroOrderRelativeIntensity: intensity[centerIndex]!,
      peakX,
      peakY,
      peakOffsetX: peakX - (input.fftWidth >>> 1),
      peakOffsetY: (input.fftHeight >>> 1) - peakY,
    },
  };
}

function isPowerOfTwo(value: number): boolean {
  return (value & (value - 1)) === 0;
}
