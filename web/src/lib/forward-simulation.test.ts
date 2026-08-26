import { describe, expect, it } from "vitest";
import {
  decodeForwardPhaseCode,
  forwardSimulationRegionAspect,
  gaussianFieldAmplitudeAt,
  shiftedForwardCoordinate,
  simulateSlmFrameWasm,
  targetForwardSimulationRegion,
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

  it("propagates the measured Gaussian pupil instead of a uniform full aperture", () => {
    const size = 16;
    const beam = {
      profile: "GAUSSIAN" as const,
      diameterXMm: 0.08,
      diameterYMm: 0.08,
      centerXMm: 0,
      centerYMm: 0,
    };
    const result = simulateSlmFrameWasm({
      pixels: new Uint8Array(size * size),
      width: size,
      height: size,
      fftWidth: size,
      fftHeight: size,
      pixelPitchUm: 10,
      incidentBeam: beam,
    });
    let fieldSum = 0;
    for (let y = 0; y < size; y += 1) {
      for (let x = 0; x < size; x += 1) {
        fieldSum += gaussianFieldAmplitudeAt(x, y, size, size, 10, beam);
      }
    }
    expect(result.metrics.maximumIntensity).toBeCloseTo(fieldSum ** 2, 2);
    expect(result.intensity[8 * size + 9]).toBeGreaterThan(1e-3);
    expect([...result.intensity].filter((value) => value > 1e-6).length).toBeGreaterThan(1);
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

  it("decodes measured phase-response values instead of assuming a linear SLM", () => {
    expect(decodeForwardPhaseCode(0, [0, 1, 6])).toBe(0);
    expect(decodeForwardPhaseCode(255, [0, 1, 6])).toBe(6);
    expect(decodeForwardPhaseCode(127.5, [0, 1, 6])).toBeCloseTo(1, 12);
  });

  it("builds a calibrated target view without stretching the physical field", () => {
    const region = targetForwardSimulationRegion([
      { x: -42, y: -2.5 },
      { x: 42, y: 2.5 },
    ], 2048, 1024, 1272, 1024);

    expect(region).not.toBeNull();
    expect(region!.x).toBeLessThan(1024 - 42);
    expect(region!.x + region!.width).toBeGreaterThan(1024 + 42);
    expect(region!.y).toBeLessThan(512 - 2.5);
    expect(region!.y + region!.height).toBeGreaterThan(512 + 2.5);
    expect(forwardSimulationRegionAspect(region!, 2048, 1024)).toBeGreaterThanOrEqual(1.5);
    expect(forwardSimulationRegionAspect(region!, 2048, 1024)).toBeLessThanOrEqual(3);
    expect(forwardSimulationRegionAspect({ x: 0, y: 0, width: 2048, height: 1024 }, 2048, 1024)).toBe(1);
    expect(shiftedForwardCoordinate(1024, 2048)).toBe(0);
    expect(shiftedForwardCoordinate(-1024, 2048)).toBe(0);
  });
});
