import { describe, expect, it } from "vitest";
import {
  analyzeOpticalTrapResolution,
  createOpticalCalibration,
  opticalFirstNullResolutionUm,
} from "./optical-calibration.js";

const dimensions = {
  activeWidth: 1272,
  activeHeight: 1024,
  fftWidth: 2048,
  fftHeight: 1024,
};
const optics = { wavelengthNm: 780, focalLengthMm: 20, pixelPitchUm: 8 };

describe("optical target resolution", () => {
  it("derives the rectangular-aperture first-null spacing", () => {
    const resolution = opticalFirstNullResolutionUm(dimensions, optics);

    expect(resolution.xUm).toBeCloseTo(1.5330188679, 10);
    expect(resolution.yUm).toBeCloseTo(1.904296875, 10);
  });

  it("detects nearly identical trap modes before solving", () => {
    const calibration = createOpticalCalibration(dimensions, optics, "resolution-test");
    const analysis = analyzeOpticalTrapResolution([
      { xUm: 0, yUm: 0 },
      { xUm: 0, yUm: 0.078431 },
      { xUm: 10, yUm: 0 },
    ], calibration, dimensions.fftWidth, dimensions.fftHeight);

    expect(analysis.unresolvedPairCount).toBe(1);
    expect(analysis.worstPair?.firstIndex).toBe(0);
    expect(analysis.worstPair?.secondIndex).toBe(1);
    expect(analysis.worstPair?.correlation).toBeCloseTo(0.997212, 5);
  });

  it("treats first-null-separated traps as independent modes", () => {
    const calibration = createOpticalCalibration(dimensions, optics, "first-null-test");
    const resolution = opticalFirstNullResolutionUm(dimensions, optics);
    const analysis = analyzeOpticalTrapResolution([
      { xUm: 0, yUm: 0 },
      { xUm: resolution.xUm, yUm: 0 },
    ], calibration, dimensions.fftWidth, dimensions.fftHeight);

    expect(analysis.unresolvedPairCount).toBe(0);
    expect(analysis.maximumModeCorrelation).toBeLessThan(1e-10);
  });
});
