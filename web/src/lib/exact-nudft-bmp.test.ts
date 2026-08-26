import { describe, expect, it } from "vitest";
import {
  SequentialWgsSolver,
  angularDistance,
  mapPhysicalPointToDftFrequency,
  mapPhysicalPointToFft,
  wasmNudftSampleTargets,
} from "../../../src/index.js";
import { decodeGrayscaleBmp, encodeGrayscaleBmp } from "./bmp.js";
import { simulateSlmFrameWasm } from "./forward-simulation.js";
import {
  createOpticalCalibration,
  parsePhaseResponseLut,
  phaseResponseForTwoPiSignalLevel,
} from "./optical-calibration.js";
import { DEFAULT_TWEEZER_AMPLITUDE_TOLERANCE_PERCENT } from "./tweezers.js";

const TAU = 2 * Math.PI;

describe("exact trap-domain Fourier regression", () => {
  it("matches an independent direct Fourier sum at fractional frequencies", () => {
    const width = 7;
    const height = 5;
    const real = Float64Array.from({ length: width * height }, (_, index) => Math.sin(index * 0.37) + 0.25);
    const imag = Float64Array.from({ length: width * height }, (_, index) => Math.cos(index * 0.19) - 0.4);
    const targetX = new Float64Array([0, 2, 3.25, width - 0.4]);
    const targetY = new Float64Array([0, 1, 2.6, height - 0.3]);
    const actual = wasmNudftSampleTargets(real, imag, width, height, targetX, targetY);

    for (let target = 0; target < targetX.length; target += 1) {
      const expected = directComplexDft(real, imag, width, height, targetX[target]!, targetY[target]!);
      expect(actual.real[target]).toBeCloseTo(expected.real, 10);
      expect(actual.imag[target]).toBeCloseTo(expected.imag, 10);
    }
  });

  it("reaches the requested off-grid trap in the exported BMP", () => {
    const width = 30;
    const height = 24;
    const fftWidth = 32;
    const fftHeight = 32;
    const optics = { wavelengthNm: 780, focalLengthMm: 20, pixelPitchUm: 8 };
    const calibration = createOpticalCalibration({
      activeWidth: width,
      activeHeight: height,
      fftWidth,
      fftHeight,
    }, optics, "bmp-analytical-regression");
    const scaleX = calibration.coordinateTransform!.a!;
    const scaleY = -calibration.coordinateTransform!.d!;
    const requestedFrequency = { x: 3.25, y: -2.4 };
    const targetPhase = 0.7;
    const trap = {
      trapId: 1,
      atomId: null,
      xUm: requestedFrequency.x / scaleX,
      // The optical transform flips +y physical coordinates to FFT row order.
      yUm: -requestedFrequency.y / scaleY,
      intensity: 1,
      targetPhaseRad: targetPhase,
      flags: 0,
    };
    const solver = new SequentialWgsSolver(calibration, {
      width: fftWidth,
      height: fftHeight,
      format: "UINT8",
      firstFrameIterations: 4,
      maxIterations: 4,
      targetPhaseMode: "PHASE_LOCKED_WGS",
      backgroundPolicy: "ZERO",
      requireConvergence: false,
    });
    const result = solver.solveSequentialFrame({ frameIndex: 0, timeUs: 0, traps: [trap] });

    const exported = encodeGrayscaleBmp(result.pixels, width, height);
    const decoded = decodeGrayscaleBmp(exported);
    const mapped = mapPhysicalPointToFft(trap, calibration, fftWidth, fftHeight);
    const measured = directCodeDft(decoded.pixels, width, height, mapped.x, mapped.y, fftWidth, fftHeight);
    const amplitude = Math.hypot(measured.real, measured.imag);
    const phaseError = angularDistance(Math.atan2(measured.imag, measured.real), targetPhase);

    expect(decoded.pixels).toEqual(result.pixels);
    expect(amplitude / (width * height)).toBeGreaterThan(0.999);
    expect(phaseError).toBeLessThan(0.015);
    expect(result.metrics.targetIntensityMean).toBeCloseTo(amplitude ** 2, 4);
    expect(result.metrics.maximumTargetPhaseErrorRad).toBeCloseTo(phaseError, 9);

    const propagated = simulateSlmFrameWasm({
      pixels: decoded.pixels,
      width,
      height,
      fftWidth,
      fftHeight,
    });
    const expectedX = Math.round(fftWidth / 2 + requestedFrequency.x);
    const expectedY = Math.round(fftHeight / 2 + requestedFrequency.y);
    const localPeak = maximumWindow(
      propagated.intensity,
      fftWidth,
      fftHeight,
      expectedX,
      expectedY,
      1,
    );
    expect(localPeak).toBeGreaterThan(0.95);
    expect(Math.abs(propagated.metrics.peakX - expectedX)).toBeLessThanOrEqual(1);
    expect(Math.abs(propagated.metrics.peakY - expectedY)).toBeLessThanOrEqual(1);
  });

  it("keeps device-ready Hamamatsu codes at or below the configured 2pi level", () => {
    const width = 32;
    const height = 16;
    const calibration = createOpticalCalibration({
      activeWidth: width,
      activeHeight: height,
      fftWidth: width,
      fftHeight: height,
    }, {
      wavelengthNm: 407,
      focalLengthMm: 100,
      pixelPitchUm: 12.5,
      incidentBeam: {
        profile: "GAUSSIAN",
        diameterXMm: 8,
        diameterYMm: 8,
      },
      phaseResponseLut: phaseResponseForTwoPiSignalLevel(217),
    }, "hamamatsu-device-ready-codes");
    const result = new SequentialWgsSolver(calibration, {
      width,
      height,
      format: "UINT8",
      firstFrameIterations: 2,
      maxIterations: 2,
      targetPhaseMode: "REFERENCE_WGS",
      backgroundPolicy: "ZERO",
      requireConvergence: false,
    }).solveSequentialFrame({
      frameIndex: 0,
      timeUs: 0,
      traps: [
        { trapId: 1, atomId: null, xUm: 10, yUm: 0, intensity: 1, targetPhaseRad: 0, flags: 0 },
        { trapId: 2, atomId: null, xUm: -10, yUm: 5, intensity: 1, targetPhaseRad: 0, flags: 0 },
      ],
    });
    expect(Math.max(...result.pixels)).toBeLessThanOrEqual(217);
  });

  it("certifies the calibrated default four traps at the attainable 8-bit tolerance", () => {
    const width = 1272;
    const height = 1024;
    const fftWidth = 2048;
    const calibration = createOpticalCalibration({
      activeWidth: width,
      activeHeight: height,
      fftWidth,
      fftHeight: height,
    }, {
      wavelengthNm: 407,
      focalLengthMm: 100,
      pixelPitchUm: 12.5,
      incidentBeam: {
        profile: "GAUSSIAN",
        diameterXMm: 8,
        diameterYMm: 8,
      },
    }, "default-four-trap-regression");
    const frame = {
      frameIndex: 0,
      timeUs: 0,
      traps: [
        { trapId: 1, atomId: null, xUm: -4, yUm: -4, intensity: 1, targetPhaseRad: 0, flags: 0 },
        { trapId: 2, atomId: null, xUm: 4, yUm: -4, intensity: 1, targetPhaseRad: Math.PI / 2, flags: 0 },
        { trapId: 3, atomId: null, xUm: -4, yUm: 4, intensity: 1, targetPhaseRad: Math.PI, flags: 0 },
        { trapId: 4, atomId: null, xUm: 4, yUm: 4, intensity: 1, targetPhaseRad: -Math.PI / 2, flags: 0 },
      ],
    };
    const solve = (iterations: number) => new SequentialWgsSolver(calibration, {
      width: fftWidth,
      height,
      format: "UINT8",
      firstFrameIterations: iterations,
      maxIterations: iterations,
      targetPhaseMode: "PHASE_LOCKED_WGS",
      backgroundPolicy: "ZERO",
      requireConvergence: false,
      convergenceTolerance: DEFAULT_TWEEZER_AMPLITUDE_TOLERANCE_PERCENT / 100,
    }).solveSequentialFrame(frame);

    const fourIterations = solve(4);
    const twelveIterations = solve(12);
    expect(fourIterations.metrics.converged).toBe(true);
    expect(fourIterations.metrics.maximumRelativeAmplitudeError).toBeGreaterThan(1e-4);
    expect(fourIterations.metrics.maximumRelativeAmplitudeError)
      .toBeLessThanOrEqual(fourIterations.metrics.amplitudeConvergenceTolerance);
    expect(fourIterations.metrics.maximumTargetPhaseErrorRad)
      .toBeLessThanOrEqual(fourIterations.metrics.phaseConvergenceToleranceRad);
    expect(twelveIterations.pixels).toEqual(fourIterations.pixels);
    expect(twelveIterations.metrics.maximumRelativeAmplitudeError)
      .toBe(fourIterations.metrics.maximumRelativeAmplitudeError);
    expect(twelveIterations.metrics.maximumTargetPhaseErrorRad)
      .toBe(fourIterations.metrics.maximumTargetPhaseErrorRad);

    const decoded = decodeGrayscaleBmp(encodeGrayscaleBmp(fourIterations.pixels, width, height));
    const measured = frame.traps.map((trap) => {
      const mapped = mapPhysicalPointToFft(trap, calibration, fftWidth, height);
      return directCodeDft(
        decoded.pixels,
        width,
        height,
        mapped.x,
        mapped.y,
        fftWidth,
        height,
        calibration.incidentAmplitude,
      );
    });
    const amplitudes = measured.map((value) => Math.hypot(value.real, value.imag));
    const scale = amplitudes.reduce((sum, value) => sum + value, 0) / amplitudes.length;
    const amplitudeError = Math.max(...amplitudes.map((value) => Math.abs(value - scale) / scale));
    const phaseError = Math.max(...measured.map((value, index) => angularDistance(
      Math.atan2(value.imag, value.real),
      frame.traps[index]!.targetPhaseRad,
    )));
    expect(amplitudeError).toBeCloseTo(fourIterations.metrics.maximumRelativeAmplitudeError, 9);
    expect(phaseError).toBeCloseTo(fourIterations.metrics.maximumTargetPhaseErrorRad, 9);
  }, 20_000);

  it("uses physical optics instead of fitting the loaded points", () => {
    const width = 128;
    const height = 64;
    const optics = { wavelengthNm: 780, focalLengthMm: 20, pixelPitchUm: 8 };
    const calibration = createOpticalCalibration({
      activeWidth: width,
      activeHeight: height,
      fftWidth: width,
      fftHeight: height,
    }, optics, "physical-map-regression");
    const first = mapPhysicalPointToFft({ xUm: 10, yUm: 6 }, calibration, width, height);
    const second = mapPhysicalPointToFft({ xUm: 20, yUm: 12 }, calibration, width, height);
    const signed = mapPhysicalPointToDftFrequency({ xUm: 10, yUm: 6 }, calibration, width, height);
    const expectedX = 10 * width * optics.pixelPitchUm /
      ((optics.wavelengthNm / 1000) * (optics.focalLengthMm * 1000));

    expect(first.x).toBeCloseTo(expectedX, 12);
    expect(signed.x).toBeCloseTo(expectedX, 12);
    expect(signed.y).toBeLessThan(0);
    expect(second.x).toBeCloseTo(first.x * 2, 12);
    expect(height - first.y).toBeCloseTo(6 * height * optics.pixelPitchUm /
      ((optics.wavelengthNm / 1000) * (optics.focalLengthMm * 1000)), 12);
    expect(second.x).not.toBeCloseTo(first.x, 6);
    expect(() => mapPhysicalPointToDftFrequency({ xUm: 976, yUm: 0 }, calibration, width, height))
      .toThrow(/Nyquist field of view/i);
  });

  it("loads a measured monotonic display-code phase response", () => {
    expect(parsePhaseResponseLut('{"phaseResponseLut":[0,1.1,3.2,6.1]}')).toEqual([0, 1.1, 3.2, 6.1]);
    expect(parsePhaseResponseLut("-3.14, -1.2, 0.4, 3.14")).toEqual([-3.14, -1.2, 0.4, 3.14]);
    expect(() => parsePhaseResponseLut("0, 2, 1, 6.2")).toThrow(/monotonic/i);
  });
});

function directCodeDft(
  pixels: Uint8Array,
  activeWidth: number,
  activeHeight: number,
  targetX: number,
  targetY: number,
  fftWidth = activeWidth,
  fftHeight = activeHeight,
  incidentAmplitude?: ArrayLike<number>,
): { real: number; imag: number } {
  const real = new Float64Array(pixels.length);
  const imag = new Float64Array(pixels.length);
  for (let index = 0; index < pixels.length; index += 1) {
    const phase = pixels[index]! / 255 * TAU - Math.PI;
    const amplitude = incidentAmplitude?.[index] ?? 1;
    real[index] = amplitude * Math.cos(phase);
    imag[index] = amplitude * Math.sin(phase);
  }
  const xStart = Math.floor((fftWidth - activeWidth) / 2);
  const yStart = Math.floor((fftHeight - activeHeight) / 2);
  let sumReal = 0;
  let sumImag = 0;
  for (let y = 0; y < activeHeight; y += 1) {
    for (let x = 0; x < activeWidth; x += 1) {
      const index = y * activeWidth + x;
      const angle = -TAU * (targetX * (x + xStart) / fftWidth + targetY * (y + yStart) / fftHeight);
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      sumReal += real[index]! * cosine - imag[index]! * sine;
      sumImag += real[index]! * sine + imag[index]! * cosine;
    }
  }
  return { real: sumReal, imag: sumImag };
}

function maximumWindow(
  values: Float32Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number,
): number {
  let maximum = 0;
  for (let y = Math.max(0, centerY - radius); y <= Math.min(height - 1, centerY + radius); y += 1) {
    for (let x = Math.max(0, centerX - radius); x <= Math.min(width - 1, centerX + radius); x += 1) {
      maximum = Math.max(maximum, values[y * width + x]!);
    }
  }
  return maximum;
}

function directComplexDft(
  real: Float64Array,
  imag: Float64Array,
  width: number,
  height: number,
  targetX: number,
  targetY: number,
): { real: number; imag: number } {
  let sumReal = 0;
  let sumImag = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const angle = -TAU * (targetX * x / width + targetY * y / height);
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      sumReal += real[index]! * cosine - imag[index]! * sine;
      sumImag += real[index]! * sine + imag[index]! * cosine;
    }
  }
  return { real: sumReal, imag: sumImag };
}
