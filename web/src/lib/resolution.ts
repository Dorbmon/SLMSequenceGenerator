export const DEFAULT_SLM_WIDTH = 1272;
export const DEFAULT_SLM_HEIGHT = 1024;
export const MIN_SLM_DIMENSION = 16;
export const MAX_SLM_DIMENSION = 2048;

export function normalizeSlmDimension(value: number, label: string): number {
  if (!Number.isInteger(value) || value < MIN_SLM_DIMENSION || value > MAX_SLM_DIMENSION) {
    throw new Error(`${label} must be an integer from ${MIN_SLM_DIMENSION} to ${MAX_SLM_DIMENSION} pixels`);
  }
  return value;
}

export function fftDimensionFor(activeDimension: number): number {
  return 2 ** Math.ceil(Math.log2(activeDimension));
}
