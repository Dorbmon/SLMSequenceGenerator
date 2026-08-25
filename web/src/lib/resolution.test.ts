import { describe, expect, it } from "vitest";
import {
  DEFAULT_SLM_HEIGHT,
  DEFAULT_SLM_WIDTH,
  fftDimensionFor,
  normalizeSlmDimension,
} from "./resolution.js";

describe("SLM resolution", () => {
  it("uses the requested physical SLM dimensions", () => {
    expect([DEFAULT_SLM_WIDTH, DEFAULT_SLM_HEIGHT]).toEqual([1272, 1024]);
  });

  it("pads non-power-of-two dimensions for the FFT grid", () => {
    expect(fftDimensionFor(1272)).toBe(2048);
    expect(fftDimensionFor(1024)).toBe(1024);
  });

  it("rejects unsafe or fractional dimensions", () => {
    expect(() => normalizeSlmDimension(15, "Width")).toThrow("16 to 2048");
    expect(() => normalizeSlmDimension(1272.5, "Width")).toThrow("integer");
  });
});
