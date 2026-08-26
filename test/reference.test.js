import assert from "node:assert/strict";
import test from "node:test";
import {
  SequentialWgsSolver,
  SlmSequenceCompiler,
  createComplexField,
  createSequencePackage,
  decodeTrapFrames,
  fft1d,
  fft2d,
  getWasmCoreInfo,
  hungarianSolve,
  verifySequencePackage,
  verifySequencePackageFiles,
} from "../dist/index.js";

function calibration(width = 16, height = 16) {
  const phaseResponseLut = Float64Array.from({ length: 256 }, (_, index) => -Math.PI + (2 * Math.PI * index) / 255);
  return {
    manifest: {
      calibrationId: "test-calibration",
      wavelengthNm: 780,
      activeWidth: width,
      activeHeight: height,
      fftWidth: width,
      fftHeight: height,
    },
    phaseResponseLut,
    coordinateTransform: { originXUm: width / 2, originYUm: height / 2 },
  };
}

function compilerOptions() {
  return {
    calibration: calibration(),
    motion: {
      framePeriodUs: 100,
      preMoveDwellUs: 100,
      postMoveSettleUs: 100,
      maxVelocityUmPerUs: 1,
      maxAccelerationUmPerUs2: 1,
      maxJerkUmPerUs3: 1,
    },
    hologram: { firstFrameIterations: 1, subsequentFrameIterations: 1, maxIterations: 1, requireConvergence: false },
  };
}

test("Hungarian solver returns a known optimum", () => {
  const result = hungarianSolve([[4, 1, 3], [2, 0, 5], [3, 2, 2]]);
  assert.deepEqual(result.assignment, [1, 0, 2]);
  assert.equal(result.cost, 5);
  assert.deepEqual(hungarianSolve([[4, 1, 3], [2, 0, 5]]).assignment, [1, 0]);
  assert.equal(hungarianSolve([[1], [2]]).feasible, false);
});

test("Wasm FFT round trips power-of-two, non-power-of-two, and 2-D fields", () => {
  const core = getWasmCoreInfo();
  assert.equal(core.backend, "webassembly");
  assert.match(core.buildId, /^rust-wasm-core-abi2-[0-9a-f]{12}$/);
  assert.equal(core.abiVersion, 2);
  assert.ok(core.moduleBytes > 0);
  const real = new Float64Array([1, 2, 3, 4]);
  const imag = new Float64Array(4);
  const original = [...real];
  fft1d(real, imag);
  fft1d(real, imag, true);
  assert.deepEqual([...real].map((value) => Math.round(value * 1e10) / 1e10), original);
  assert.ok([...imag].every((value) => Math.abs(value) < 1e-9));

  const nonPowerReal = new Float64Array([1, -2, 3, -4, 5]);
  const nonPowerImag = new Float64Array([0.5, 0, -0.5, 1, -1]);
  const nonPowerOriginalReal = new Float64Array(nonPowerReal);
  const nonPowerOriginalImag = new Float64Array(nonPowerImag);
  fft1d(nonPowerReal, nonPowerImag);
  fft1d(nonPowerReal, nonPowerImag, true);
  assertArrayClose(nonPowerReal, nonPowerOriginalReal);
  assertArrayClose(nonPowerImag, nonPowerOriginalImag);

  const field = createComplexField(3, 5);
  field.real.set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
  field.imag.set([0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0]);
  const fieldOriginalReal = new Float64Array(field.real);
  const fieldOriginalImag = new Float64Array(field.imag);
  fft2d(field);
  fft2d(field, true);
  assertArrayClose(field.real, fieldOriginalReal);
  assertArrayClose(field.imag, fieldOriginalImag);
});

test("compiles an identity sequence and exports checksummed frames", async () => {
  const compiler = await SlmSequenceCompiler.create(compilerOptions());
  const sequence = await compiler.compileRearrangement({
    initialAtoms: [{ atomId: 1, xUm: 0, yUm: 0 }],
    targetSites: [{ siteId: 2, xUm: 0, yUm: 0 }],
    calibrationId: "test-calibration",
  });
  assert.equal(sequence.manifest.wasmBuildId, getWasmCoreInfo().buildId);
  assert.equal(sequence.validation.accepted, true);
  assert.equal(sequence.trapFrameStore.length, sequence.slmFrameStore.length);
  assert.equal(verifySequencePackage(sequence).valid, true);
  const packageFiles = createSequencePackage(sequence);
  assert.equal(verifySequencePackage(sequence, JSON.parse(packageFiles["slm-frames.index.json"])).valid, true);
  assert.equal(decodeTrapFrames(packageFiles["trap-frames.bin"]).length, sequence.trapFrameStore.length);
  assert.equal(verifySequencePackageFiles(sequence, packageFiles).valid, true);
  assert.ok(packageFiles["slm-frames.bin"].length > 0);
});

test("crops a padded FFT grid to the active SLM resolution", async () => {
  const options = compilerOptions();
  options.calibration = calibration(8, 8);
  options.calibration.manifest.activeWidth = 6;
  options.calibration.manifest.activeHeight = 4;
  const compiler = await SlmSequenceCompiler.create(options);
  const sequence = await compiler.compileRearrangement({
    initialAtoms: [{ atomId: 1, xUm: 0, yUm: 0 }],
    targetSites: [{ siteId: 2, xUm: 0, yUm: 0 }],
    calibrationId: "test-calibration",
  });
  const frames = sequence.slmFrameStore.toArray();
  assert.equal(sequence.manifest.outputWidth, 6);
  assert.equal(sequence.manifest.outputHeight, 4);
  assert.equal(sequence.manifest.wgsParameters.fftWidth, 8);
  assert.equal(sequence.manifest.wgsParameters.fftHeight, 8);
  assert.ok(frames.every((frame) => frame.length === 24));
  assert.ok(sequence.slmFrameDescriptors.every((descriptor) => descriptor.width === 6 && descriptor.height === 4));
  assert.equal(createSequencePackage(sequence)["slm-frames.bin"].length, frames.length * 24);
  assert.equal(verifySequencePackage(sequence).valid, true);
});

test("awaits an injected hologram backend and records its identity", async () => {
  const compiler = await SlmSequenceCompiler.create(compilerOptions());
  let disposed = false;
  const sequence = await compiler.compileRearrangement({
    initialAtoms: [{ atomId: 1, xUm: 0, yUm: 0 }],
    targetSites: [{ siteId: 2, xUm: 0, yUm: 0 }],
    calibrationId: "test-calibration",
  }, {
    hologramSolverFactory(solverCalibration, solverConfig) {
      const cpu = new SequentialWgsSolver(solverCalibration, solverConfig);
      return {
        backendId: "test-async-backend",
        async solveSequentialFrame(frame, budget) {
          await Promise.resolve();
          return cpu.solveSequentialFrame(frame, budget);
        },
        async commitFrameState() {
          cpu.commitFrameState();
        },
        async rollbackToPreviousAcceptedFrame() {
          cpu.rollbackToPreviousAcceptedFrame();
        },
        async dispose() {
          disposed = true;
        },
      };
    },
  });
  assert.equal(sequence.manifest.wgsBackend, "test-async-backend");
  assert.equal(disposed, true);
  assert.equal(sequence.validation.accepted, true);
});

test("generates a direct phase-locked optical tweezer frame", () => {
  const solver = new SequentialWgsSolver(calibration(), {
    width: 16,
    height: 16,
    firstFrameIterations: 4,
    maxIterations: 4,
    targetPhaseMode: "PHASE_LOCKED_WGS",
    backgroundPolicy: "ZERO",
    requireConvergence: false,
  });
  const result = solver.solveSequentialFrame({
    frameIndex: 0,
    timeUs: 0,
    traps: [
      { trapId: 1, atomId: null, xUm: -1, yUm: 0, intensity: 1, targetPhaseRad: 0, flags: 0 },
      { trapId: 2, atomId: null, xUm: 1, yUm: 0, intensity: 1, targetPhaseRad: 1.2, flags: 0 },
    ],
  });
  assert.equal(result.pixels.length, 16 * 16);
  assert.equal(result.metrics.accepted, true);
  assert.ok(result.metrics.maximumTargetPhaseErrorRad < 0.2);
});

test("uses full WGS amplitude feedback when target phases are free", () => {
  const width = 64;
  const referenceCalibration = calibration(width, width);
  referenceCalibration.coordinateTransform = {
    originXUm: width / 2,
    originYUm: width / 2,
    scaleX: 1,
    scaleY: 1,
  };
  const frame = {
    frameIndex: 0,
    timeUs: 0,
    traps: Array.from({ length: 8 }, (_, index) => ({
      trapId: index + 1,
      atomId: null,
      xUm: Math.cos(index * Math.PI / 4) * 14,
      yUm: Math.sin(index * Math.PI / 4) * 14,
      intensity: 0.2 + (index % 4) * 0.25,
      targetPhaseRad: 0,
      flags: 0,
    })),
  };
  const solve = (gamma) => new SequentialWgsSolver(referenceCalibration, {
    width,
    height: width,
    firstFrameIterations: 6,
    maxIterations: 6,
    targetPhaseMode: "REFERENCE_WGS",
    backgroundPolicy: "ZERO",
    convergenceTolerance: 0.01,
    ...(gamma === undefined ? {} : { gamma }),
  }).solveSequentialFrame(frame);

  const conservative = solve(0.1);
  const normal = solve(0.7);
  const defaultReference = solve(undefined);
  const tunedReference = solve(0.85);
  assert.ok(normal.metrics.maximumRelativeAmplitudeError < conservative.metrics.maximumRelativeAmplitudeError / 2);
  assert.equal(normal.metrics.maximumTargetPhaseErrorRad, 0);
  assert.deepEqual(defaultReference.pixels, tunedReference.pixels);
});

test("uses a seeded stable phase when the initial target superposition cancels", () => {
  const symmetricCalibration = calibration(64, 64);
  symmetricCalibration.coordinateTransform = {
    originXUm: 32,
    originYUm: 32,
    scaleX: 6.4,
    scaleY: 6.4,
  };
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
  const solve = (deterministicSeed) => new SequentialWgsSolver(symmetricCalibration, {
    width: 64,
    height: 64,
    firstFrameIterations: 1,
    maxIterations: 1,
    targetPhaseMode: "PHASE_LOCKED_WGS",
    backgroundPolicy: "ZERO",
    deterministicSeed,
    requireConvergence: false,
  }).solveSequentialFrame(frame);

  const first = solve(7);
  const repeated = solve(7);
  const differentSeed = solve(19);
  assert.deepEqual(first.pixels, repeated.pixels);
  assert.notDeepEqual(first.pixels, differentSeed.pixels);
  assert.equal(first.metrics.numericalValid, true);
});

test("a larger WGS budget cannot replace a better quantized candidate", () => {
  const width = 64;
  const candidateCalibration = calibration(width, width);
  candidateCalibration.coordinateTransform = {
    originXUm: width / 2,
    originYUm: width / 2,
    scaleX: 1,
    scaleY: 1,
  };
  const frame = {
    frameIndex: 0,
    timeUs: 0,
    traps: Array.from({ length: 6 }, (_, index) => ({
      trapId: index + 1,
      atomId: null,
      xUm: Math.cos(index * Math.PI / 3) * 10,
      yUm: Math.sin(index * Math.PI / 3) * 10,
      intensity: 0.5 + (index % 3) * 0.25,
      targetPhaseRad: -2.5 + index * 0.8,
      flags: 0,
    })),
  };
  const solve = (iterations) => new SequentialWgsSolver(candidateCalibration, {
    width,
    height: width,
    firstFrameIterations: iterations,
    maxIterations: iterations,
    targetPhaseMode: "PHASE_LOCKED_WGS",
    backgroundPolicy: "ZERO",
  }).solveSequentialFrame(frame);
  const score = (metrics) => Math.max(
    metrics.maximumRelativeAmplitudeError / metrics.amplitudeConvergenceTolerance,
    metrics.maximumTargetPhaseErrorRad / metrics.phaseConvergenceToleranceRad,
  );

  const fourIterations = solve(4);
  const twelveIterations = solve(12);
  assert.ok(score(twelveIterations.metrics) <= score(fourIterations.metrics));
  assert.equal(twelveIterations.metrics.numericalValid, true);
});

test("routes a constrained crossing without collision", async () => {
  const options = compilerOptions();
  options.planner = { minimumSeparationUm: 1, geometricMarginUm: 0.1, gridResolutionUm: 1, maxSearchTicks: 100, maxCbsNodes: 2000 };
  const compiler = await SlmSequenceCompiler.create(options);
  const sequence = await compiler.compileRearrangement({
    initialAtoms: [{ atomId: 1, xUm: -4, yUm: 0 }, { atomId: 2, xUm: 4, yUm: 0 }],
    targetSites: [
      { siteId: 10, xUm: 4, yUm: 0, requiredAtomId: 1 },
      { siteId: 11, xUm: -4, yUm: 0, requiredAtomId: 2 },
    ],
    calibrationId: "test-calibration",
  });
  assert.equal(sequence.validation.accepted, true);
  assert.ok(sequence.trajectories.some((trajectory) => trajectory.waypoints.length > 3));
});

test("detours around a static occupied trap", async () => {
  const options = compilerOptions();
  options.calibration.coordinateTransform = { originXUm: 8, originYUm: 8, scaleX: 0.5, scaleY: 0.5 };
  const compiler = await SlmSequenceCompiler.create(options);
  const sequence = await compiler.compileRearrangement({
    initialAtoms: [{ atomId: 1, xUm: 0, yUm: 0 }],
    targetSites: [{ siteId: 2, xUm: 10, yUm: 0 }],
    staticTraps: [{ trapId: 99, xUm: 5, yUm: 0, intensity: 1, containsAtom: true }],
    calibrationId: "test-calibration",
  });
  assert.equal(sequence.validation.accepted, true);
  assert.ok(sequence.trajectories[0].waypoints.some((point) => point.yUm !== 0));
});

test("parks and releases excess atoms when parking is configured", async () => {
  const options = compilerOptions();
  options.assignment = { parkingSites: [{ xUm: 0, yUm: 3 }] };
  const compiler = await SlmSequenceCompiler.create(options);
  const sequence = await compiler.compileRearrangement({
    initialAtoms: [{ atomId: 1, xUm: 0, yUm: 0 }, { atomId: 2, xUm: 4, yUm: 0 }],
    targetSites: [{ siteId: 2, xUm: 0, yUm: 0 }],
    calibrationId: "test-calibration",
  });
  assert.equal(sequence.assignment[1].disposition, "PARK");
  assert.equal(sequence.trapFrameStore.get(sequence.trapFrameStore.length - 1).traps[1].atomId, null);
});

test("rejects malformed geometry and calibration references", async () => {
  const compiler = await SlmSequenceCompiler.create(compilerOptions());
  await assert.rejects(
    compiler.compileRearrangement({
      initialAtoms: [{ xUm: 0, yUm: 0 }],
      targetSites: [{ xUm: 1, yUm: 0 }],
      forbiddenRegions: [{ type: "circle", coordinates: [0, 0, 1, 2] }],
      calibrationId: "test-calibration",
    }),
    (error) => error.code === "INVALID_ARGUMENT",
  );
  await assert.rejects(
    compiler.compileRearrangement({ initialAtoms: [], targetSites: [], calibrationId: "missing" }),
    (error) => error.code === "CALIBRATION_MISMATCH",
  );
});

function assertArrayClose(actual, expected, tolerance = 1e-9) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance, `index ${index}: ${actual[index]} != ${expected[index]}`);
  }
}
