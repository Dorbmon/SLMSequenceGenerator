import { describe, expect, it } from "vitest";
import {
  analyzeOpticalTrapResolution,
  createGaussianIncidentAmplitude,
  createOpticalCalibration,
  DEFAULT_FOCAL_LENGTH_MM,
  DEFAULT_INCIDENT_BEAM_DIAMETER_MM,
  DEFAULT_PIXEL_PITCH_UM,
  DEFAULT_TWO_PI_SIGNAL_LEVEL,
  DEFAULT_WAVELENGTH_NM,
  opticalEffectiveResolutionUm,
  opticalFirstNullResolutionUm,
  phaseResponseForTwoPiSignalLevel,
} from "./optical-calibration.js";

const dimensions = {
  activeWidth: 1272,
  activeHeight: 1024,
  fftWidth: 2048,
  fftHeight: 1024,
};
const optics = { wavelengthNm: 780, focalLengthMm: 20, pixelPitchUm: 8 };

describe("optical target resolution", () => {
  it("uses the measured X15213-05 experiment as the browser hardware preset", () => {
    expect({
      wavelengthNm: DEFAULT_WAVELENGTH_NM,
      focalLengthMm: DEFAULT_FOCAL_LENGTH_MM,
      pixelPitchUm: DEFAULT_PIXEL_PITCH_UM,
      beamDiameterMm: DEFAULT_INCIDENT_BEAM_DIAMETER_MM,
      twoPiSignalLevel: DEFAULT_TWO_PI_SIGNAL_LEVEL,
    }).toEqual({
      wavelengthNm: 407,
      focalLengthMm: 100,
      pixelPitchUm: 12.5,
      beamDiameterMm: 8,
      twoPiSignalLevel: 217,
    });
  });

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

  it("models the measured underfilled Gaussian pupil and its wider focal spot", () => {
    const hardware = {
      wavelengthNm: 407,
      focalLengthMm: 100,
      pixelPitchUm: 12.5,
      incidentBeam: {
        profile: "GAUSSIAN" as const,
        diameterXMm: 8,
        diameterYMm: 8,
        centerXMm: 0,
        centerYMm: 0,
      },
    };
    const resolution = opticalEffectiveResolutionUm(dimensions, hardware);
    expect(resolution.xUm).toBeCloseTo(6.47760618, 7);
    expect(resolution.yUm).toBeCloseTo(6.47760618, 7);

    const amplitude = createGaussianIncidentAmplitude({ activeWidth: 3, activeHeight: 3 }, {
      ...hardware,
      pixelPitchUm: 1000,
      incidentBeam: { ...hardware.incidentBeam, diameterXMm: 2, diameterYMm: 2 },
    });
    expect(amplitude).toBeDefined();
    expect(amplitude![4]).toBe(1);
    expect(amplitude![5]).toBeCloseTo(Math.exp(-1), 12);
  });

  it("encodes a device-ready 2pi signal level without affecting SLMControl3 input mode", () => {
    const response = phaseResponseForTwoPiSignalLevel(217);
    expect(response).toHaveLength(256);
    expect(response[0]).toBe(0);
    expect(response[217]).toBeCloseTo(2 * Math.PI, 12);
    expect(response[255]).toBeGreaterThan(2 * Math.PI);
    expect(() => phaseResponseForTwoPiSignalLevel(0)).toThrow(/between 1 and 255/i);
  });
});
