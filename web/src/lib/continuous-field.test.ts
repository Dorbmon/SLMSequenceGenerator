import { describe, expect, it } from "vitest";
import {
  buildContinuousTargetGrid,
  createContinuousInversePhaseLut,
  solveContinuousFieldWasm,
  type ContinuousFieldInput,
} from "./continuous-field.js";
import { decodeGrayscaleBmp, encodeGrayscaleBmp } from "./bmp.js";
import { simulateSlmFrameWasm } from "./forward-simulation.js";

function input(): ContinuousFieldInput {
  const targetWidth = 48;
  const targetHeight = 32;
  const targetIntensity = new Float32Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y += 1) {
    for (let x = 0; x < targetWidth; x += 1) {
      const horizontal = x >= 7 && x <= 40 && Math.abs(y - 16) <= 2;
      const vertical = y >= 6 && y <= 25 && Math.abs(x - 24) <= 2;
      if (horizontal || vertical) targetIntensity[y * targetWidth + x] = 1;
    }
  }
  return {
    targetIntensity,
    targetWidth,
    targetHeight,
    slmWidth: 64,
    slmHeight: 64,
    fftWidth: 64,
    fftHeight: 64,
    fieldWidthUm: 1_000,
    fieldHeightUm: 800,
    fieldCenterXUm: 120,
    fieldCenterYUm: -90,
    iterations: 18,
    mixingFactor: 0.4,
    deterministicSeed: 7,
    opticalCalibration: {
      wavelengthNm: 780,
      focalLengthMm: 20,
      pixelPitchUm: 8,
      incidentBeam: {
        profile: "GAUSSIAN",
        diameterXMm: 0.5,
        diameterYMm: 0.5,
      },
    },
  };
}

describe("continuous-field MRAF", () => {
  it("maps a raster target into the calibrated shifted focal plane", () => {
    const request = input();
    const target = buildContinuousTargetGrid(request);
    const fov = 0.780 * 20_000 / 8;
    expect(target.region.width).toBe(Math.round(request.fieldWidthUm / fov * request.fftWidth));
    expect(target.region.height).toBe(Math.round(request.fieldHeightUm / fov * request.fftHeight));
    expect(target.targetPower).toBeGreaterThan(0);
    expect(target.signalMask.some((value) => value !== 0)).toBe(true);
  });

  it("generates a deterministic quantized frame and certifies its reconstruction", () => {
    const first = solveContinuousFieldWasm(input());
    const second = solveContinuousFieldWasm(input());
    expect(first.pixels).toEqual(second.pixels);
    expect(first.pixels.length).toBe(64 * 64);
    expect(first.intensity.length).toBe(64 * 64);
    expect(first.metrics.numericalValid).toBe(true);
    expect(first.metrics.accepted).toBe(true);
    expect(first.metrics.intensityCorrelation).toBeGreaterThan(0.95);
    expect(first.metrics.normalizedIntensityRmse).toBeLessThan(0.25);
    expect(first.metrics.diffractionEfficiency).toBeGreaterThan(0);
  });

  it("round-trips the exported BMP into the exact certified reconstruction", () => {
    const request = input();
    const result = solveContinuousFieldWasm(request);
    const decoded = decodeGrayscaleBmp(encodeGrayscaleBmp(
      result.pixels,
      request.slmWidth,
      request.slmHeight,
    ));
    const propagated = simulateSlmFrameWasm({
      pixels: decoded.pixels,
      width: request.slmWidth,
      height: request.slmHeight,
      fftWidth: request.fftWidth,
      fftHeight: request.fftHeight,
      pixelPitchUm: request.opticalCalibration.pixelPitchUm,
      ...(request.opticalCalibration.incidentBeam
        ? { incidentBeam: request.opticalCalibration.incidentBeam }
        : {}),
      ...(request.opticalCalibration.phaseResponseLut
        ? { phaseResponseLut: request.opticalCalibration.phaseResponseLut }
        : {}),
    });

    expect(decoded.pixels).toEqual(result.pixels);
    expect(propagated.intensity).toEqual(result.intensity);
  });

  it("rejects an empty continuous target", () => {
    const request = input();
    request.targetIntensity.fill(0);
    expect(() => solveContinuousFieldWasm(request)).toThrow(/non-empty continuous target/i);
  });

  it("inverts positive and descending device phase conventions", () => {
    const increasing = createContinuousInversePhaseLut([0, Math.PI, 2 * Math.PI], 5);
    const descending = createContinuousInversePhaseLut([2 * Math.PI, Math.PI, 0], 5);
    expect(increasing[0]).toBeGreaterThanOrEqual(127);
    expect(increasing[0]).toBeLessThanOrEqual(128);
    expect(descending[0]).toBeGreaterThanOrEqual(127);
    expect(descending[0]).toBeLessThanOrEqual(128);
    expect(increasing[1]).toBeGreaterThanOrEqual(190);
    expect(descending[1]).toBeLessThanOrEqual(65);
    expect(increasing[2]).toBe(0);
    expect(descending[2]).toBe(255);
  });
});
