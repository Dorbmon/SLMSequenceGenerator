import { describe, expect, it } from "vitest";
import {
  simulateSlmFrameWasm,
  validateForwardSimulationInput,
} from "./forward-simulation.js";

describe("SLM forward simulation", () => {
  it("places a uniform phase frame at the shifted zero order", () => {
    const result = simulateSlmFrameWasm({
      pixels: new Uint8Array(16 * 16),
      width: 16,
      height: 16,
      fftWidth: 16,
      fftHeight: 16,
    });
    expect(result.metrics.peakOffsetX).toBe(0);
    expect(result.metrics.peakOffsetY).toBe(0);
    expect(result.metrics.zeroOrderRelativeIntensity).toBe(1);
    expect(result.metrics.maximumIntensity).toBeCloseTo(16 ** 4, 2);
    expect(result.intensity[8 * 16 + 8]).toBe(1);
    expect([...result.intensity].filter((value) => value > 1e-10)).toHaveLength(1);
  });

  it("centres a non-power-of-two active frame inside the FFT aperture", () => {
    const result = simulateSlmFrameWasm({
      pixels: new Uint8Array(18 * 20),
      width: 18,
      height: 20,
      fftWidth: 32,
      fftHeight: 32,
    });
    expect(result.metrics.peakOffsetX).toBe(0);
    expect(result.metrics.peakOffsetY).toBe(0);
    expect(result.metrics.maximumIntensity).toBeCloseTo((18 * 20) ** 2, 1);
    expect(Math.max(...result.intensity)).toBe(1);
  });

  it("reports the signed FFT offset of a phase ramp", () => {
    const size = 32;
    const horizontalOrder = 5;
    const pixels = new Uint8Array(size * size);
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        const turns = ((horizontalOrder * x / size) % 1 + 1) % 1;
        pixels[y * size + x] = Math.round(turns * 255);
      }
    }
    const result = simulateSlmFrameWasm({
      pixels,
      width: size,
      height: size,
      fftWidth: size,
      fftHeight: size,
    });
    expect(result.metrics.peakOffsetX).toBe(horizontalOrder);
    expect(result.metrics.peakOffsetY).toBe(0);
  });

  it("rejects mismatched pixels and non-radix-2 grids", () => {
    expect(() => validateForwardSimulationInput({
      pixels: new Uint8Array(3), width: 2, height: 2, fftWidth: 2, fftHeight: 2,
    })).toThrow(/expected 2 × 2/i);
    expect(() => validateForwardSimulationInput({
      pixels: new Uint8Array(16), width: 4, height: 4, fftWidth: 6, fftHeight: 4,
    })).toThrow(/power-of-two/i);
  });
});
