import { createComplexField, fft2d } from "../../../src/index.js";

const TAU = Math.PI * 2;

export interface ForwardSimulationInput {
  pixels: Uint8Array;
  width: number;
  height: number;
  fftWidth: number;
  fftHeight: number;
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
}

/** Propagates a phase-code frame with the Rust/Wasm radix-2 FFT core. */
export function simulateSlmFrameWasm(input: ForwardSimulationInput): ForwardSimulationResult {
  validateForwardSimulationInput(input);
  const field = createComplexField(input.fftWidth, input.fftHeight);
  const xStart = Math.floor((input.fftWidth - input.width) / 2);
  const yStart = Math.floor((input.fftHeight - input.height) / 2);
  for (let y = 0; y < input.height; y += 1) {
    for (let x = 0; x < input.width; x += 1) {
      const source = y * input.width + x;
      const destination = (yStart + y) * input.fftWidth + xStart + x;
      const phase = input.pixels[source]! / 255 * TAU - Math.PI;
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
