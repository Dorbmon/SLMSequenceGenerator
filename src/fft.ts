import { SlmError } from "./errors.js";

export interface ComplexField {
  real: Float64Array;
  imag: Float64Array;
  width: number;
  height: number;
}

export function createComplexField(width: number, height: number): ComplexField {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new SlmError("INVALID_ARGUMENT", "Complex field dimensions must be positive integers", { stage: "SOLVING_SLM_FRAMES" });
  }
  return { real: new Float64Array(width * height), imag: new Float64Array(width * height), width, height };
}

export function fft1d(real: Float64Array, imag: Float64Array, inverse = false): void {
  if (real.length !== imag.length) throw new SlmError("INVALID_ARGUMENT", "FFT real and imaginary lengths differ", { stage: "SOLVING_SLM_FRAMES" });
  if (real.length === 0) return;
  if ((real.length & (real.length - 1)) !== 0) {
    dft1d(real, imag, inverse);
    return;
  }
  bitReversePermutation(real, imag);
  for (let size = 2; size <= real.length; size <<= 1) {
    const half = size >>> 1;
    const sign = inverse ? 1 : -1;
    const angle = sign * (2 * Math.PI / size);
    const rootReal = Math.cos(angle);
    const rootImag = Math.sin(angle);
    for (let offset = 0; offset < real.length; offset += size) {
      let twiddleReal = 1;
      let twiddleImag = 0;
      for (let index = 0; index < half; index += 1) {
        const even = offset + index;
        const odd = even + half;
        const productReal = real[odd]! * twiddleReal - imag[odd]! * twiddleImag;
        const productImag = real[odd]! * twiddleImag + imag[odd]! * twiddleReal;
        const evenReal = real[even]!;
        const evenImag = imag[even]!;
        real[even] = evenReal + productReal;
        imag[even] = evenImag + productImag;
        real[odd] = evenReal - productReal;
        imag[odd] = evenImag - productImag;
        const nextTwiddleReal = twiddleReal * rootReal - twiddleImag * rootImag;
        twiddleImag = twiddleReal * rootImag + twiddleImag * rootReal;
        twiddleReal = nextTwiddleReal;
      }
    }
  }
  if (inverse) {
    const scale = 1 / real.length;
    for (let index = 0; index < real.length; index += 1) {
      real[index] = real[index]! * scale;
      imag[index] = imag[index]! * scale;
    }
  }
}

export function fft2d(field: ComplexField, inverse = false): void {
  const { width, height, real, imag } = field;
  const rowReal = new Float64Array(width);
  const rowImag = new Float64Array(width);
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x += 1) {
      rowReal[x] = real[rowOffset + x]!;
      rowImag[x] = imag[rowOffset + x]!;
    }
    fft1d(rowReal, rowImag, inverse);
    for (let x = 0; x < width; x += 1) {
      real[rowOffset + x] = rowReal[x]!;
      imag[rowOffset + x] = rowImag[x]!;
    }
  }
  const columnReal = new Float64Array(height);
  const columnImag = new Float64Array(height);
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      columnReal[y] = real[y * width + x]!;
      columnImag[y] = imag[y * width + x]!;
    }
    fft1d(columnReal, columnImag, inverse);
    for (let y = 0; y < height; y += 1) {
      real[y * width + x] = columnReal[y]!;
      imag[y * width + x] = columnImag[y]!;
    }
  }
}

export function cloneComplexField(field: ComplexField): ComplexField {
  return { real: new Float64Array(field.real), imag: new Float64Array(field.imag), width: field.width, height: field.height };
}

export function sampleComplex(field: ComplexField, x: number, y: number, bilinear = true): { real: number; imag: number } {
  if (!bilinear) {
    const ix = clampIndex(Math.round(x), field.width);
    const iy = clampIndex(Math.round(y), field.height);
    const index = iy * field.width + ix;
    return { real: field.real[index]!, imag: field.imag[index]! };
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const tx = x - x0;
  const ty = y - y0;
  const p00 = fieldValue(field, x0, y0);
  const p10 = fieldValue(field, x1, y0);
  const p01 = fieldValue(field, x0, y1);
  const p11 = fieldValue(field, x1, y1);
  return {
    real: lerp(lerp(p00.real, p10.real, tx), lerp(p01.real, p11.real, tx), ty),
    imag: lerp(lerp(p00.imag, p10.imag, tx), lerp(p01.imag, p11.imag, tx), ty),
  };
}

export function scatterComplex(field: ComplexField, x: number, y: number, real: number, imag: number, bilinear = false): void {
  if (!bilinear) {
    const ix = clampIndex(Math.round(x), field.width);
    const iy = clampIndex(Math.round(y), field.height);
    const index = iy * field.width + ix;
    field.real[index] = real;
    field.imag[index] = imag;
    return;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  for (const [xOffset, xWeight] of [[0, 1 - tx], [1, tx]] as const) {
    for (const [yOffset, yWeight] of [[0, 1 - ty], [1, ty]] as const) {
      const ix = clampIndex(x0 + xOffset, field.width);
      const iy = clampIndex(y0 + yOffset, field.height);
      const index = iy * field.width + ix;
      const weight = xWeight * yWeight;
      field.real[index] = field.real[index]! + real * weight;
      field.imag[index] = field.imag[index]! + imag * weight;
    }
  }
}

/**
 * Adjoint bilinear scatter. Normalizing by the squared interpolation weights
 * makes a subsequent bilinear sample reproduce the requested complex value.
 */
export function scatterComplexAdjoint(field: ComplexField, x: number, y: number, real: number, imag: number, bilinear = true): void {
  if (!bilinear) {
    scatterComplex(field, x, y, real, imag, false);
    return;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const entries = new Map<number, number>();
  for (const [xOffset, xWeight] of [[0, 1 - tx], [1, tx]] as const) {
    for (const [yOffset, yWeight] of [[0, 1 - ty], [1, ty]] as const) {
      const ix = clampIndex(x0 + xOffset, field.width);
      const iy = clampIndex(y0 + yOffset, field.height);
      const index = iy * field.width + ix;
      entries.set(index, (entries.get(index) ?? 0) + xWeight * yWeight);
    }
  }
  const norm = [...entries.values()].reduce((sum, weight) => sum + weight * weight, 0);
  if (norm <= 0) return;
  for (const [index, weight] of entries) {
    field.real[index] = field.real[index]! + real * weight / norm;
    field.imag[index] = field.imag[index]! + imag * weight / norm;
  }
}

export function fieldPower(field: ComplexField): Float64Array {
  const power = new Float64Array(field.real.length);
  for (let index = 0; index < power.length; index += 1) power[index] = field.real[index]! ** 2 + field.imag[index]! ** 2;
  return power;
}

function bitReversePermutation(real: Float64Array, imag: Float64Array): void {
  let reversed = 0;
  for (let index = 1; index < real.length; index += 1) {
    let bit = real.length >>> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed]!, real[index]!];
      [imag[index], imag[reversed]] = [imag[reversed]!, imag[index]!];
    }
  }
}

function dft1d(real: Float64Array, imag: Float64Array, inverse: boolean): void {
  const inputReal = new Float64Array(real);
  const inputImag = new Float64Array(imag);
  const sign = inverse ? 1 : -1;
  const size = real.length;
  for (let output = 0; output < size; output += 1) {
    let sumReal = 0;
    let sumImag = 0;
    for (let input = 0; input < size; input += 1) {
      const angle = sign * 2 * Math.PI * output * input / size;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      sumReal += inputReal[input]! * cosine - inputImag[input]! * sine;
      sumImag += inputReal[input]! * sine + inputImag[input]! * cosine;
    }
    const scale = inverse ? 1 / size : 1;
    real[output] = sumReal * scale;
    imag[output] = sumImag * scale;
  }
}

function fieldValue(field: ComplexField, x: number, y: number): { real: number; imag: number } {
  const ix = clampIndex(x, field.width);
  const iy = clampIndex(y, field.height);
  const index = iy * field.width + ix;
  return { real: field.real[index]!, imag: field.imag[index]! };
}

function clampIndex(value: number, size: number): number {
  return Math.max(0, Math.min(size - 1, value));
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}
