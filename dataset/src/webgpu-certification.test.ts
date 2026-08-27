import { create, globals } from "webgpu";
import { describe, expect, it } from "vitest";
import { mapPhysicalPointToDftFrequency } from "../../src/coordinates.js";
import type { TrapFrame } from "../../src/types.js";
import { createOpticalCalibration } from "../../web/src/lib/optical-calibration.js";
import { WebGpuSequentialWgsSolver } from "../../web/src/lib/webgpu-wgs.js";

const TAU = Math.PI * 2;

describe("export-certified WebGPU WGS", () => {
  it("reports phases and metrics for the exact returned UINT8 frame", async () => {
    Object.assign(globalThis, globals);
    const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
    const dawnGpu = create([]);
    Object.defineProperty(globalThis, "navigator", {
      configurable: true,
      value: { gpu: dawnGpu },
    });

    const activeWidth = 30;
    const activeHeight = 24;
    const fftWidth = 32;
    const fftHeight = 32;
    const calibration = createOpticalCalibration({
      activeWidth,
      activeHeight,
      fftWidth,
      fftHeight,
    }, {
      wavelengthNm: 407,
      focalLengthMm: 20,
      pixelPitchUm: 8,
      incidentBeam: {
        profile: "GAUSSIAN",
        diameterXMm: 0.2,
        diameterYMm: 0.18,
      },
    }, "dawn-export-certification-test");
    const frequencies = [
      [-10.25, -8.4], [-6.6, -5.2], [-2.75, -7.1], [1.4, -4.3],
      [7.2, -6.8], [-8.8, 0.75], [-4.1, 3.6], [0.3, 1.8],
      [4.9, 5.25], [9.1, 2.4], [-1.7, 9.2], [6.35, 8.1],
    ] as const;
    const scaleX = calibration.coordinateTransform!.a!;
    const scaleY = calibration.coordinateTransform!.d!;
    const frame: TrapFrame = {
      frameIndex: 7,
      timeUs: 0,
      traps: frequencies.map(([x, y], index) => ({
        trapId: index + 1,
        atomId: null,
        xUm: x / scaleX,
        yUm: y / scaleY,
        intensity: 1,
        targetPhaseRad: 0,
        flags: 0,
      })),
    };

    let solver: WebGpuSequentialWgsSolver | undefined;
    try {
      solver = await WebGpuSequentialWgsSolver.create(calibration, {
        width: fftWidth,
        height: fftHeight,
        format: "UINT8",
        targetPhaseMode: "REFERENCE_WGS",
        firstFrameIterations: 4,
        maxIterations: 4,
        convergenceTolerance: 0.05,
        deterministicSeed: 123,
        backgroundPolicy: "ZERO",
        requireConvergence: false,
      });
      const result = await solver.solveSequentialFrame(frame);
      expect(result.pixels).toBeInstanceOf(Uint8Array);
      const pixels = result.pixels as Uint8Array;
      const measuredPhases = solver.getCandidateMeasuredPhases(frame);
      const measured = frame.traps.map((trap) => directTargetSample(
        pixels,
        activeWidth,
        activeHeight,
        fftWidth,
        fftHeight,
        mapPhysicalPointToDftFrequency(trap, calibration, fftWidth, fftHeight),
        calibration.incidentAmplitude!,
      ));

      measured.forEach((value, index) => {
        expect(phaseDistance(Math.atan2(value.imaginary, value.real), measuredPhases[index]!))
          .toBeLessThan(3e-4);
      });
      const amplitudes = measured.map((value) => Math.hypot(value.real, value.imaginary));
      const scale = amplitudes.reduce((sum, value) => sum + value, 0) / amplitudes.length;
      const maximumRelativeError = Math.max(...amplitudes.map((value) => Math.abs(value - scale) / scale));
      expect(result.metrics.maximumRelativeAmplitudeError).toBeCloseTo(maximumRelativeError, 4);

      const maximumGhost = directMaximumGhost(
        pixels,
        activeWidth,
        activeHeight,
        fftWidth,
        fftHeight,
        frame.traps.map((trap) => mapPhysicalPointToDftFrequency(trap, calibration, fftWidth, fftHeight)),
        calibration.incidentAmplitude!,
      );
      expect(result.metrics.maximumGhostIntensity / maximumGhost).toBeCloseTo(1, 4);
    } finally {
      solver?.dispose();
      if (originalNavigator) Object.defineProperty(globalThis, "navigator", originalNavigator);
      else delete (globalThis as { navigator?: unknown }).navigator;
    }
  }, 20_000);
});

function directTargetSample(
  pixels: Uint8Array,
  activeWidth: number,
  activeHeight: number,
  fftWidth: number,
  fftHeight: number,
  position: { x: number; y: number },
  amplitude: ArrayLike<number>,
): { real: number; imaginary: number } {
  const xStart = Math.floor((fftWidth - activeWidth) / 2);
  const yStart = Math.floor((fftHeight - activeHeight) / 2);
  let real = 0;
  let imaginary = 0;
  for (let y = 0; y < activeHeight; y += 1) {
    for (let x = 0; x < activeWidth; x += 1) {
      const activeIndex = y * activeWidth + x;
      const phase = pixels[activeIndex]! / 255 * TAU - Math.PI;
      const dftAngle = -TAU * (
        position.x * (xStart + x) / fftWidth
        + position.y * (yStart + y) / fftHeight
      );
      const magnitude = amplitude[activeIndex]!;
      real += magnitude * Math.cos(phase + dftAngle);
      imaginary += magnitude * Math.sin(phase + dftAngle);
    }
  }
  return { real, imaginary };
}

function directMaximumGhost(
  pixels: Uint8Array,
  activeWidth: number,
  activeHeight: number,
  fftWidth: number,
  fftHeight: number,
  targets: readonly { x: number; y: number }[],
  amplitude: ArrayLike<number>,
): number {
  const support = new Uint8Array(fftWidth * fftHeight);
  for (const target of targets) {
    const x = periodicPosition(target.x, fftWidth);
    const y = periodicPosition(target.y, fftHeight);
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) {
        const pixelX = periodicIndex(x0 + dx, fftWidth);
        const pixelY = periodicIndex(y0 + dy, fftHeight);
        const deltaX = Math.min(Math.abs(pixelX - x), fftWidth - Math.abs(pixelX - x));
        const deltaY = Math.min(Math.abs(pixelY - y), fftHeight - Math.abs(pixelY - y));
        if (deltaX * deltaX + deltaY * deltaY <= 2.25) support[pixelY * fftWidth + pixelX] = 1;
      }
    }
  }

  let maximum = 0;
  for (let y = 0; y < fftHeight; y += 1) {
    for (let x = 0; x < fftWidth; x += 1) {
      if (support[y * fftWidth + x] === 1) continue;
      const value = directTargetSample(
        pixels,
        activeWidth,
        activeHeight,
        fftWidth,
        fftHeight,
        { x, y },
        amplitude,
      );
      maximum = Math.max(maximum, value.real * value.real + value.imaginary * value.imaginary);
    }
  }
  return maximum;
}

function phaseDistance(first: number, second: number): number {
  return Math.abs(Math.atan2(Math.sin(first - second), Math.cos(first - second)));
}

function periodicPosition(value: number, extent: number): number {
  return value - Math.floor(value / extent) * extent;
}

function periodicIndex(value: number, extent: number): number {
  return ((value % extent) + extent) % extent;
}
