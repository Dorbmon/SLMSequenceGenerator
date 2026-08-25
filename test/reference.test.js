import assert from "node:assert/strict";
import test from "node:test";
import {
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
  assert.match(core.buildId, /^rust-wasm-core-abi1-[0-9a-f]{12}$/);
  assert.equal(core.abiVersion, 1);
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
