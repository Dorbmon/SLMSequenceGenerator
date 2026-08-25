import { SlmCompilationError, SlmError, assertFinite } from "./errors.js";
import { distance, pointInForbiddenRegion } from "./geometry.js";
import type {
  AssignmentConfig,
  CalibrationPackage,
  CompilerConfig,
  ForbiddenRegion,
  HologramConfig,
  InitialAtom,
  MotionConfig,
  NormalizedInput,
  OccupiedLatticeInput,
  PlannerConfig,
  RearrangementRequest,
  StaticTrap,
  TargetSite,
} from "./types.js";
import { hashValue } from "./util.js";

export const DEFAULT_PLANNER_CONFIG: Required<PlannerConfig> = {
  minimumSeparationUm: 1,
  kSigma: 3,
  geometricMarginUm: 0.1,
  duplicatePointToleranceUm: 0,
  duplicatePointPolicy: "reject",
  gridResolutionUm: 0.5,
  planningTickUs: 1_000,
  maxSearchTicks: 512,
  maxAStarExpansions: 50_000,
  maxCbsNodes: 1_000,
  ecbsLimit: 12,
  maxPriorityRetries: 4,
};

export const DEFAULT_ASSIGNMENT_CONFIG: Required<AssignmentConfig> = {
  distanceWeight: 1,
  obstacleWeight: 1_000,
  staticObstacleWeight: 100,
  groupMismatchPenalty: 1_000,
  groupMismatchPolicy: "forbid",
  extraAtomPolicy: "KEEP",
  parkingSites: [],
  stayToleranceUm: 0.25,
  maxAssignmentRetries: 2,
  conflictPenalty: 1_000,
};

export const DEFAULT_MOTION_CONFIG: Required<MotionConfig> = {
  framePeriodUs: 1_000,
  preMoveDwellUs: 1_000,
  postMoveSettleUs: 1_000,
  minDwellBeforeMoveUs: 0,
  maxVelocityUmPerUs: 0.05,
  maxAccelerationUmPerUs2: 0.001,
  maxJerkUmPerUs3: 0.000001,
  maxPositionChangePerFrameUm: Number.POSITIVE_INFINITY,
  maxIntensityChangePerFrame: Number.POSITIVE_INFINITY,
  movingTrapIntensity: 1,
  defaultTrapIntensity: 1,
  maxValidationDepth: 12,
};

export const DEFAULT_HOLOGRAM_CONFIG: Required<HologramConfig> = {
  width: 32,
  height: 32,
  format: "UINT8",
  targetPhaseMode: "PHASE_LOCKED_WGS",
  firstFrameIterations: 12,
  subsequentFrameIterations: 4,
  maxIterations: 64,
  gamma: 0.7,
  epsilon: 1e-8,
  minWeight: 0.1,
  maxWeight: 10,
  convergenceTolerance: 1e-4,
  backgroundPolicy: "PRESERVE",
  oversampling: 1,
  qualityGates: {},
  maxInsertedFrames: 32,
  deterministicSeed: 1,
  measureSolveTime: false,
  requireConvergence: false,
};

export function normalizeAndValidate(
  request: RearrangementRequest,
  compilerConfig: CompilerConfig = {},
): NormalizedInput {
  if (!request || !Array.isArray(request.targetSites)) {
    throw new SlmError("INVALID_ARGUMENT", "targetSites must be an array", { stage: "VALIDATING" });
  }

  const plannerConfig = normalizePlannerConfig({
    ...DEFAULT_PLANNER_CONFIG,
    ...compilerConfig.planner,
    ...request.plannerConfig,
  });
  const assignmentOverrides = { ...compilerConfig.assignment, ...request.assignmentConfig };
  const assignmentConfig = normalizeAssignmentConfig({
    ...DEFAULT_ASSIGNMENT_CONFIG,
    ...assignmentOverrides,
    ...(assignmentOverrides.extraAtomPolicy === undefined && assignmentOverrides.parkingSites && assignmentOverrides.parkingSites.length > 0
      ? { extraAtomPolicy: "PARK_AND_RELEASE" as const }
      : {}),
  });
  const motionConfig = normalizeMotionConfig({
    ...DEFAULT_MOTION_CONFIG,
    ...compilerConfig.motion,
    ...request.motionConfig,
  });

  const calibration = resolveCalibration(request, compilerConfig, motionConfig, plannerConfig);
  const hologramConfig = normalizeHologramConfig({
    ...DEFAULT_HOLOGRAM_CONFIG,
    width: calibration.manifest.fftWidth ?? calibration.manifest.activeWidth,
    height: calibration.manifest.fftHeight ?? calibration.manifest.activeHeight,
    ...compilerConfig.hologram,
    ...request.hologramConfig,
  });
  validateCalibration(calibration, hologramConfig, compilerConfig.simulationMode === true, compilerConfig);
  const requestedWavelength = request.wavelengthNm ?? compilerConfig.wavelengthNm;
  if (requestedWavelength !== undefined && requestedWavelength !== calibration.manifest.wavelengthNm) {
    throw new SlmError("CALIBRATION_MISMATCH", "Calibration wavelength does not match the requested wavelength", {
      stage: "VALIDATING",
      details: { requestedWavelength, calibrationWavelength: calibration.manifest.wavelengthNm },
    });
  }

  const rawAtoms = normalizeLatticeOrAtoms(request.initialAtoms);
  const initialAtoms = normalizeAtoms(rawAtoms, plannerConfig);
  const targetSites = normalizeTargets(request.targetSites, plannerConfig);
  const staticTraps = normalizeStaticTraps(request.staticTraps ?? [], plannerConfig);
  const forbiddenRegions = normalizeForbiddenRegions(request.forbiddenRegions ?? []);
  if ((assignmentConfig.extraAtomPolicy === "PARK" || assignmentConfig.extraAtomPolicy === "PARK_AND_RELEASE") && assignmentConfig.parkingSites.length === 0) {
    assignmentConfig.parkingSites = makeAutomaticParkingSites(initialAtoms, targetSites, assignmentConfig, plannerConfig, calibration);
  }

  validateFieldOfView(initialAtoms, targetSites, staticTraps, calibration);
  validateForbiddenTargets(targetSites, forbiddenRegions);
  validateInitialAtomSeparation(initialAtoms, plannerConfig);
  validateRequiredTargetSeparation(targetSites, plannerConfig.minimumSeparationUm + plannerConfig.geometricMarginUm);
  validateStaticTrapIds(staticTraps);
  validateStaticTrapAtomIds(initialAtoms, staticTraps);
  validateTargetsAgainstStaticGeometry(targetSites, initialAtoms, staticTraps, plannerConfig);
  validateStaticGeometrySeparation(initialAtoms, staticTraps, plannerConfig);
  validateParkingSites(assignmentConfig.parkingSites, calibration);

  const movableCount = initialAtoms.filter((atom) => atom.movable).length;
  const requiredCount = targetSites.filter((site) => site.required).length;
  if (movableCount < requiredCount) {
    throw new SlmCompilationError(
      "INSUFFICIENT_ATOMS",
      `Need ${requiredCount} movable atoms but only ${movableCount} are available`,
      [
        {
          code: "INSUFFICIENT_ATOMS",
          stage: "VALIDATING",
          message: "There are fewer movable atoms than required targets",
          configured: requiredCount,
          measured: movableCount,
        },
      ],
      { stage: "VALIDATING" },
    );
  }

  return {
    initialAtoms,
    targetSites,
    staticTraps,
    forbiddenRegions,
    calibration,
    plannerConfig,
    assignmentConfig,
    motionConfig,
    hologramConfig,
  };
}

export function normalizeLatticeOrAtoms(input: RearrangementRequest["initialAtoms"]): InitialAtom[] {
  if (Array.isArray(input)) return input.map((atom) => ({ ...atom }));
  const lattice = input as OccupiedLatticeInput;
  if (!lattice || !Array.isArray(lattice.sites) || !(lattice.occupied instanceof Uint8Array)) {
    throw new SlmError("INVALID_ARGUMENT", "initialAtoms must be atoms or an occupied lattice", {
      stage: "VALIDATING",
    });
  }
  if (lattice.sites.length !== lattice.occupied.length) {
    throw new SlmError("INVALID_ARGUMENT", "lattice sites and occupancy lengths differ", {
      stage: "VALIDATING",
    });
  }
  return lattice.sites.flatMap((point, index) =>
    lattice.occupied[index] ? [{ xUm: point.xUm, yUm: point.yUm }] : [],
  );
}

function normalizeAtoms(atoms: InitialAtom[], planner: Required<PlannerConfig>): Required<InitialAtom>[] {
  const usedIds = new Set<number>();
  const result: Required<InitialAtom>[] = [];
  for (let index = 0; index < atoms.length; index += 1) {
    const atom = atoms[index]!;
    validatePoint(atom.xUm, atom.yUm, `initialAtoms[${index}]`);
    const atomId = atom.atomId ?? index;
    validateId(atomId, `initialAtoms[${index}].atomId`);
    if (usedIds.has(atomId)) duplicateId("atom", atomId);
    usedIds.add(atomId);
    const normalized: Required<InitialAtom> = {
      atomId,
      xUm: atom.xUm,
      yUm: atom.yUm,
      group: atom.group ?? 0,
      movable: atom.movable ?? true,
      initialTrapIntensity: atom.initialTrapIntensity ?? 1,
      localizationSigmaUm: atom.localizationSigmaUm ?? 0,
    };
    validateId(normalized.group, `initialAtoms[${index}].group`);
    if (typeof normalized.movable !== "boolean") throw new SlmError("INVALID_ARGUMENT", "movable must be boolean", { stage: "VALIDATING" });
    validateNonNegative(normalized.initialTrapIntensity, "initialTrapIntensity");
    validateNonNegative(normalized.localizationSigmaUm, "localizationSigmaUm");
    result.push(normalized);
  }
  return mergeDuplicatePoints(result, planner.duplicatePointToleranceUm, planner.duplicatePointPolicy);
}

function normalizeTargets(targets: TargetSite[], planner: Required<PlannerConfig>): Required<TargetSite>[] {
  const usedIds = new Set<number>();
  const result: Required<TargetSite>[] = [];
  for (let index = 0; index < targets.length; index += 1) {
    const target = targets[index]!;
    validatePoint(target.xUm, target.yUm, `targetSites[${index}]`);
    const siteId = target.siteId ?? index;
    validateId(siteId, `targetSites[${index}].siteId`);
    if (usedIds.has(siteId)) duplicateId("target site", siteId);
    usedIds.add(siteId);
    const normalized: Required<TargetSite> = {
      siteId,
      xUm: target.xUm,
      yUm: target.yUm,
      required: target.required ?? true,
      requiredAtomId: target.requiredAtomId ?? -1,
      requiredGroup: target.requiredGroup ?? -1,
      finalTrapIntensity: target.finalTrapIntensity ?? 1,
    };
    if (normalized.requiredAtomId !== -1) validateId(normalized.requiredAtomId, "requiredAtomId");
    if (normalized.requiredGroup !== -1) validateId(normalized.requiredGroup, "requiredGroup");
    if (typeof normalized.required !== "boolean") throw new SlmError("INVALID_ARGUMENT", "required must be boolean", { stage: "VALIDATING" });
    validateNonNegative(normalized.finalTrapIntensity, "finalTrapIntensity");
    result.push(normalized);
  }
  return mergeDuplicatePoints(result, planner.duplicatePointToleranceUm, planner.duplicatePointPolicy);
}

function normalizeStaticTraps(traps: StaticTrap[], planner: Required<PlannerConfig>): Required<StaticTrap>[] {
  const usedIds = new Set<number>();
  const result: Required<StaticTrap>[] = [];
  for (let index = 0; index < traps.length; index += 1) {
    const trap = traps[index]!;
    validatePoint(trap.xUm, trap.yUm, `staticTraps[${index}]`);
    validateId(trap.trapId, `staticTraps[${index}].trapId`);
    if (usedIds.has(trap.trapId)) duplicateId("static trap", trap.trapId);
    usedIds.add(trap.trapId);
    validateNonNegative(trap.intensity, "static trap intensity");
    if (typeof trap.containsAtom !== "boolean") throw new SlmError("INVALID_ARGUMENT", "containsAtom must be boolean", { stage: "VALIDATING" });
    result.push({
      trapId: trap.trapId,
      xUm: trap.xUm,
      yUm: trap.yUm,
      intensity: trap.intensity,
      containsAtom: trap.containsAtom,
      atomId: trap.atomId ?? -1,
    });
    if (trap.atomId !== undefined) validateId(trap.atomId, `staticTraps[${index}].atomId`);
  }
  return mergeDuplicatePoints(result, planner.duplicatePointToleranceUm, planner.duplicatePointPolicy);
}

function normalizeForbiddenRegions(regions: ForbiddenRegion[]): ForbiddenRegion[] {
  return regions.map((region, index) => {
    if (!region || !["circle", "axisAlignedBox", "polygon"].includes(region.type) || !Array.isArray(region.coordinates)) {
      throw new SlmError("INVALID_ARGUMENT", `Invalid forbidden region at index ${index}`, {
        stage: "VALIDATING",
      });
    }
    const expected = region.type === "circle" ? 3 : region.type === "axisAlignedBox" ? 4 : 6;
    if (region.type === "polygon" && (region.coordinates.length < 6 || region.coordinates.length % 2 !== 0)) {
      throw new SlmError("INVALID_ARGUMENT", `Polygon ${index} needs at least three points`, {
        stage: "VALIDATING",
      });
    }
    if (region.type !== "polygon" && region.coordinates.length !== expected) {
      throw new SlmError("INVALID_ARGUMENT", `Forbidden region ${index} has invalid coordinates`, {
        stage: "VALIDATING",
      });
    }
    region.coordinates.forEach((value, coordinateIndex) =>
      assertFinite(value, `forbiddenRegions[${index}].coordinates[${coordinateIndex}]`),
    );
    if (region.type === "circle" && region.coordinates[2]! < 0) {
      throw new SlmError("INVALID_ARGUMENT", "Circle radius cannot be negative", { stage: "VALIDATING" });
    }
    if (region.type === "axisAlignedBox" &&
        (region.coordinates[0]! > region.coordinates[2]! || region.coordinates[1]! > region.coordinates[3]!)) {
      throw new SlmError("INVALID_ARGUMENT", "Axis-aligned box bounds are inverted", { stage: "VALIDATING" });
    }
    if (region.clearanceUm !== undefined) validateNonNegative(region.clearanceUm, "region clearanceUm");
    return { type: region.type, coordinates: [...region.coordinates], ...(region.clearanceUm === undefined ? {} : { clearanceUm: region.clearanceUm }) };
  });
}

function mergeDuplicatePoints<T extends { xUm: number; yUm: number }>(
  points: T[],
  tolerance: number,
  policy: "reject" | "merge",
): T[] {
  if (tolerance <= 0 || points.length < 2) return points;
  const result: T[] = [];
  for (const point of points) {
    const duplicateIndex = result.findIndex((other) => distance(point, other) <= tolerance);
    if (duplicateIndex < 0) {
      result.push(point);
    } else if (policy === "reject") {
      throw new SlmError("INVALID_TARGET_GEOMETRY", "Duplicate points violate duplicatePointToleranceUm", {
        stage: "VALIDATING",
        details: { tolerance },
      });
    }
  }
  return result;
}

function normalizePlannerConfig(config: PlannerConfig): Required<PlannerConfig> {
  const result = config as Required<PlannerConfig>;
  positive(result.minimumSeparationUm, "minimumSeparationUm");
  nonNegative(result.kSigma, "kSigma");
  nonNegative(result.geometricMarginUm, "geometricMarginUm");
  nonNegative(result.duplicatePointToleranceUm, "duplicatePointToleranceUm");
  if (result.duplicatePointPolicy !== "reject" && result.duplicatePointPolicy !== "merge") {
    throw new SlmError("INVALID_ARGUMENT", "Unknown duplicatePointPolicy", { stage: "VALIDATING" });
  }
  positive(result.gridResolutionUm, "gridResolutionUm");
  positiveInteger(result.planningTickUs, "planningTickUs");
  positiveInteger(result.maxSearchTicks, "maxSearchTicks");
  positiveInteger(result.maxAStarExpansions, "maxAStarExpansions");
  positiveInteger(result.maxCbsNodes, "maxCbsNodes");
  positiveInteger(result.ecbsLimit, "ecbsLimit");
  positiveInteger(result.maxPriorityRetries, "maxPriorityRetries");
  return result;
}

function normalizeAssignmentConfig(config: AssignmentConfig): Required<AssignmentConfig> {
  const result = config as Required<AssignmentConfig>;
  positive(result.distanceWeight, "distanceWeight");
  nonNegative(result.obstacleWeight, "obstacleWeight");
  nonNegative(result.staticObstacleWeight, "staticObstacleWeight");
  nonNegative(result.groupMismatchPenalty, "groupMismatchPenalty");
  nonNegative(result.stayToleranceUm, "stayToleranceUm");
  positiveInteger(result.maxAssignmentRetries + 1, "maxAssignmentRetries");
  nonNegative(result.conflictPenalty, "conflictPenalty");
  if (![
    "KEEP",
    "PARK",
    "PARK_AND_RELEASE",
    "RELEASE_IN_PLACE",
  ].includes(result.extraAtomPolicy)) {
    throw new SlmError("INVALID_ARGUMENT", "Unknown extraAtomPolicy", { stage: "VALIDATING" });
  }
  if (result.groupMismatchPolicy !== "forbid" && result.groupMismatchPolicy !== "penalize") {
    throw new SlmError("INVALID_ARGUMENT", "Unknown groupMismatchPolicy", { stage: "VALIDATING" });
  }
  if (!Array.isArray(result.parkingSites)) {
    throw new SlmError("INVALID_ARGUMENT", "parkingSites must be an array", { stage: "VALIDATING" });
  }
  result.parkingSites.forEach((point, index) => {
    if (!point) throw new SlmError("INVALID_ARGUMENT", `parkingSites[${index}] is invalid`, { stage: "VALIDATING" });
    validatePoint(point.xUm, point.yUm, `parkingSites[${index}]`);
  });
  return result;
}

function normalizeMotionConfig(config: MotionConfig): Required<MotionConfig> {
  const result = config as Required<MotionConfig>;
  positiveInteger(result.framePeriodUs, "framePeriodUs");
  nonNegativeInteger(result.preMoveDwellUs, "preMoveDwellUs");
  nonNegativeInteger(result.postMoveSettleUs, "postMoveSettleUs");
  nonNegativeInteger(result.minDwellBeforeMoveUs, "minDwellBeforeMoveUs");
  positive(result.maxVelocityUmPerUs, "maxVelocityUmPerUs");
  positive(result.maxAccelerationUmPerUs2, "maxAccelerationUmPerUs2");
  positive(result.maxJerkUmPerUs3, "maxJerkUmPerUs3");
  if (result.maxPositionChangePerFrameUm !== Number.POSITIVE_INFINITY) positive(result.maxPositionChangePerFrameUm, "maxPositionChangePerFrameUm");
  if (result.maxIntensityChangePerFrame !== Number.POSITIVE_INFINITY) positive(result.maxIntensityChangePerFrame, "maxIntensityChangePerFrame");
  nonNegative(result.movingTrapIntensity, "movingTrapIntensity");
  nonNegative(result.defaultTrapIntensity, "defaultTrapIntensity");
  positiveInteger(result.maxValidationDepth, "maxValidationDepth");
  return result;
}

function normalizeHologramConfig(config: HologramConfig): Required<HologramConfig> {
  const result = config as Required<HologramConfig>;
  positiveInteger(result.width, "hologram width");
  positiveInteger(result.height, "hologram height");
  if (result.format !== "UINT8" && result.format !== "UINT16") {
    throw new SlmError("INVALID_ARGUMENT", "hologram format must be UINT8 or UINT16", { stage: "VALIDATING" });
  }
  positiveInteger(result.firstFrameIterations, "firstFrameIterations");
  positiveInteger(result.subsequentFrameIterations, "subsequentFrameIterations");
  positiveInteger(result.maxIterations, "maxIterations");
  positive(result.gamma, "gamma");
  positive(result.epsilon, "epsilon");
  positive(result.minWeight, "minWeight");
  if (result.maxWeight < result.minWeight) {
    throw new SlmError("INVALID_ARGUMENT", "maxWeight must be >= minWeight", { stage: "VALIDATING" });
  }
  positive(result.maxWeight, "maxWeight");
  if (result.targetPhaseMode !== "REFERENCE_WGS" && result.targetPhaseMode !== "PHASE_LOCKED_WGS" && result.targetPhaseMode !== "SOFT_PHASE_LOCKED_WGS" && result.targetPhaseMode !== "PHASE_INTERPOLATED_WGS") {
    throw new SlmError("INVALID_ARGUMENT", "Unknown target phase mode", { stage: "VALIDATING" });
  }
  if (result.backgroundPolicy !== "PRESERVE" && result.backgroundPolicy !== "ZERO") {
    throw new SlmError("INVALID_ARGUMENT", "Unknown WGS background policy", { stage: "VALIDATING" });
  }
  validateQualityGates(result.qualityGates);
  positiveInteger(result.oversampling, "oversampling");
  positiveInteger(result.maxInsertedFrames, "maxInsertedFrames");
  if (!Number.isInteger(result.deterministicSeed) || !Number.isFinite(result.deterministicSeed)) {
    throw new SlmError("INVALID_ARGUMENT", "deterministicSeed must be an integer", { stage: "VALIDATING" });
  }
  if (typeof result.measureSolveTime !== "boolean") {
    throw new SlmError("INVALID_ARGUMENT", "measureSolveTime must be boolean", { stage: "VALIDATING" });
  }
  if (typeof result.requireConvergence !== "boolean") {
    throw new SlmError("INVALID_ARGUMENT", "requireConvergence must be boolean", { stage: "VALIDATING" });
  }
  return result;
}

function resolveCalibration(
  request: RearrangementRequest,
  config: CompilerConfig,
  motion: Required<MotionConfig>,
  planner: Required<PlannerConfig>,
): CalibrationPackage {
  const calibration = request.calibration ??
    (request.calibrationId && config.calibrations?.[request.calibrationId]) ??
    config.calibration;
  if (calibration) {
    if (!calibration.manifest || (request.calibrationId && calibration.manifest.calibrationId !== request.calibrationId)) {
      throw new SlmError("CALIBRATION_MISMATCH", "Requested calibrationId does not match calibration package", {
        stage: "VALIDATING",
      });
    }
    validateCalibrationManifest(calibration);
    return calibration;
  }

  if (config.simulationMode !== true) {
    throw new SlmError("CALIBRATION_MISMATCH", "A measured calibration package is required", { stage: "VALIDATING" });
  }
  // A small identity package is available only for explicit reference simulation.
  const width = request.hologramConfig?.width ?? config.hologram?.width ?? 32;
  const height = request.hologramConfig?.height ?? config.hologram?.height ?? 32;
  positiveInteger(width, "default calibration width");
  positiveInteger(height, "default calibration height");
  void motion;
  void planner;
  return {
    manifest: {
      calibrationId: request.calibrationId ?? "synthetic-identity",
      wavelengthNm: 1,
      activeWidth: width,
      activeHeight: height,
      fftWidth: width,
      fftHeight: height,
      pixelPitchUm: 1,
      coordinateConvention: "+x right, +y up",
    },
    coordinateTransform: { originXUm: width / 2, originYUm: height / 2, scaleX: 1, scaleY: 1 },
  };
}

export function validateCalibrationManifest(calibration: CalibrationPackage): void {
  const manifest = calibration?.manifest;
  if (!manifest || typeof manifest.calibrationId !== "string" || manifest.calibrationId.length === 0) {
    throw new SlmError("CALIBRATION_MISMATCH", "Calibration manifest needs a calibrationId", { stage: "VALIDATING" });
  }
  positiveInteger(manifest.activeWidth, "calibration activeWidth");
  positiveInteger(manifest.activeHeight, "calibration activeHeight");
  if (manifest.fftWidth !== undefined) positiveInteger(manifest.fftWidth, "calibration fftWidth");
  if (manifest.fftHeight !== undefined) positiveInteger(manifest.fftHeight, "calibration fftHeight");
  positive(manifest.wavelengthNm, "calibration wavelengthNm");
  if (manifest.pixelPitchUm !== undefined) positive(manifest.pixelPitchUm, "calibration pixelPitchUm");
  if (manifest.focalLengthMm !== undefined) positive(manifest.focalLengthMm, "calibration focalLengthMm");
  if (manifest.fftWidth !== undefined && manifest.fftHeight === undefined ||
      manifest.fftHeight !== undefined && manifest.fftWidth === undefined) {
    throw new SlmError("CALIBRATION_MISMATCH", "Both FFT dimensions must be specified", { stage: "VALIDATING" });
  }
  if ((manifest.fftWidth ?? manifest.activeWidth) < manifest.activeWidth ||
      (manifest.fftHeight ?? manifest.activeHeight) < manifest.activeHeight) {
    throw new SlmError("CALIBRATION_MISMATCH", "Active calibration dimensions exceed the FFT grid", { stage: "VALIDATING" });
  }
  if (manifest.fieldOfViewUm) {
    const fov = manifest.fieldOfViewUm;
    [fov.xMinUm, fov.xMaxUm, fov.yMinUm, fov.yMaxUm].forEach((value) => assertFinite(value, "field of view"));
    if (fov.xMinUm > fov.xMaxUm || fov.yMinUm > fov.yMaxUm) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration field of view bounds are inverted", { stage: "VALIDATING" });
    }
  }
}

function validateCalibration(
  calibration: CalibrationPackage,
  hologram: Required<HologramConfig>,
  simulationMode: boolean,
  compilerConfig: CompilerConfig = {},
): void {
  validateCalibrationManifest(calibration);
  const manifest = calibration.manifest;
  const expectedWidth = manifest.fftWidth ?? manifest.activeWidth;
  const expectedHeight = manifest.fftHeight ?? manifest.activeHeight;
  if (hologram.width !== expectedWidth || hologram.height !== expectedHeight) {
    throw new SlmError("CALIBRATION_MISMATCH", "Hologram dimensions do not match the calibration FFT grid", {
      stage: "VALIDATING",
      details: { expectedWidth, expectedHeight, width: hologram.width, height: hologram.height },
    });
  }
  const pixels = expectedWidth * expectedHeight;
  validateCalibrationArray(calibration.incidentAmplitude, pixels, "incidentAmplitude", manifest.activeWidth * manifest.activeHeight);
  validateCalibrationArray(calibration.apertureMask, pixels, "apertureMask", manifest.activeWidth * manifest.activeHeight);
  validateCalibrationNonNegative(calibration.incidentAmplitude, "incidentAmplitude");
  validateCalibrationNonNegative(calibration.apertureMask, "apertureMask");
  validateCalibrationArray(calibration.aberrationPhase, pixels, "aberrationPhase");
  validateCalibrationArray(calibration.carrierGrating, pixels, "carrierGrating");
  validateCalibrationArray(calibration.digitalLens, pixels, "digitalLens");
  if (calibration.phaseResponseLut && calibration.phaseResponseLut.length < 2) {
    throw new SlmError("CALIBRATION_MISMATCH", "phaseResponseLut must contain at least two codes", { stage: "VALIDATING" });
  }
  if (calibration.inversePhaseLut && calibration.inversePhaseLut.length < 2) {
    throw new SlmError("CALIBRATION_MISMATCH", "inversePhaseLut must contain at least two codes", { stage: "VALIDATING" });
  }
  validateCalibrationLut(calibration.phaseResponseLut, "phaseResponseLut");
  validateCalibrationLut(calibration.inversePhaseLut, "inversePhaseLut");
  if (!simulationMode && !calibration.phaseResponseLut && !calibration.inversePhaseLut) {
    throw new SlmError("CALIBRATION_MISMATCH", "Calibration must provide a measured phase-response LUT", { stage: "VALIDATING" });
  }
  const requirements = compilerConfig.calibrationRequirements;
  if (requirements) {
    if (requirements.slmModel !== undefined && requirements.slmModel !== manifest.slmModel) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration SLM model does not match the requested device", { stage: "VALIDATING" });
    }
    if (requirements.serialNumber !== undefined && requirements.serialNumber !== manifest.serialNumber) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration serial number does not match the requested device", { stage: "VALIDATING" });
    }
    if (requirements.wavelengthNm !== undefined && requirements.wavelengthNm !== manifest.wavelengthNm) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration wavelength does not match the requested device", { stage: "VALIDATING" });
    }
  }
  validateCalibrationChecksums(calibration, compilerConfig.strictCalibration === true);
  const transform = calibration.coordinateTransform;
  if (transform) {
    for (const value of [transform.originXUm, transform.originYUm, transform.scaleX, transform.scaleY, transform.rotationRad, transform.offsetX, transform.offsetY, transform.a, transform.b, transform.c, transform.d]) {
      if (value !== undefined) assertFinite(value, "coordinate transform");
    }
    if (transform.fftCoordinateSpace !== undefined &&
        transform.fftCoordinateSpace !== "RAW_INDEX" &&
        transform.fftCoordinateSpace !== "SIGNED_FREQUENCY") {
      throw new SlmError("CALIBRATION_MISMATCH", "Unknown FFT coordinate space", { stage: "VALIDATING" });
    }
  }
  if (calibration.phaseSigns) {
    for (const sign of [calibration.phaseSigns.aberration, calibration.phaseSigns.grating, calibration.phaseSigns.lens]) {
      if (sign !== undefined && sign !== 1 && sign !== -1) {
        throw new SlmError("CALIBRATION_MISMATCH", "Phase correction signs must be +1 or -1", { stage: "VALIDATING" });
      }
    }
  }
}

export function validateCalibrationPackage(
  calibration: CalibrationPackage,
  options: { width?: number; height?: number; simulationMode?: boolean } = {},
): void {
  validateCalibrationManifest(calibration);
  const hologram = normalizeHologramConfig({
    ...DEFAULT_HOLOGRAM_CONFIG,
    width: options.width ?? calibration.manifest.fftWidth ?? calibration.manifest.activeWidth,
    height: options.height ?? calibration.manifest.fftHeight ?? calibration.manifest.activeHeight,
  });
  validateCalibration(calibration, hologram, options.simulationMode === true);
}

function validateCalibrationChecksums(calibration: CalibrationPackage, strict: boolean): void {
  const checksums = calibration.manifest.checksums;
  const arrays: [string, ArrayLike<number> | number | undefined][] = [
    ["incidentAmplitude", calibration.incidentAmplitude],
    ["apertureMask", calibration.apertureMask],
    ["aberrationPhase", calibration.aberrationPhase],
    ["phaseResponseLut", calibration.phaseResponseLut],
    ["inversePhaseLut", calibration.inversePhaseLut],
    ["carrierGrating", calibration.carrierGrating],
    ["digitalLens", calibration.digitalLens],
  ];
  for (const [name, value] of arrays) {
    if (value === undefined) continue;
    const filename = name.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
    const extension = name === "apertureMask" ? ".u8" : name.toLowerCase().includes("lut") ? ".u16" : ".f32";
    const expected = checksums?.[name] ?? checksums?.[filename] ?? checksums?.[`${filename}${extension}`];
    if (!expected) {
      if (strict) throw new SlmError("CALIBRATION_MISMATCH", `Calibration checksum is missing for ${name}`, { stage: "VALIDATING" });
      continue;
    }
    const actual = hashValue(value);
    if (actual !== expected) throw new SlmError("CALIBRATION_MISMATCH", `Calibration checksum mismatch for ${name}`, { stage: "VALIDATING", details: { name, expected, actual } });
  }
}

function validateCalibrationLut(value: ArrayLike<number> | undefined, name: string): void {
  if (!value) return;
  for (let index = 0; index < value.length; index += 1) assertFinite(value[index]!, `${name}[${index}]`);
}

function validateCalibrationNonNegative(value: ArrayLike<number> | number | undefined, name: string): void {
  if (value === undefined) return;
  if (typeof value === "number") {
    if (value < 0) throw new SlmError("CALIBRATION_MISMATCH", `${name} cannot be negative`, { stage: "VALIDATING" });
    return;
  }
  for (let index = 0; index < value.length; index += 1) {
    if (value[index]! < 0) throw new SlmError("CALIBRATION_MISMATCH", `${name} cannot be negative`, { stage: "VALIDATING" });
  }
}

function validateQualityGates(gates: HologramConfig["qualityGates"]): void {
  if (!gates) throw new SlmError("INVALID_ARGUMENT", "qualityGates must be an object", { stage: "VALIDATING" });
  const values: [string, number | undefined][] = [
    ["maxIntensityCoefficientOfVariation", gates.maxIntensityCoefficientOfVariation],
    ["minIntensityToMeanRatio", gates.minIntensityToMeanRatio],
    ["minDiffractionEfficiency", gates.minDiffractionEfficiency],
    ["maxGhostIntensity", gates.maxGhostIntensity],
    ["maxTargetPhaseErrorRad", gates.maxTargetPhaseErrorRad],
    ["maxPhaseChangeRad", gates.maxPhaseChangeRad],
    ["maxDisplayCodeChange", gates.maxDisplayCodeChange],
  ];
  for (const [name, value] of values) if (value !== undefined) nonNegative(value, name);
}

function validateCalibrationArray(value: ArrayLike<number> | number | undefined, expected: number, name: string, alternateExpected?: number): void {
  if (value === undefined || typeof value === "number") {
    if (typeof value === "number") assertFinite(value, name);
    return;
  }
  if (value.length !== expected && value.length !== alternateExpected) {
    throw new SlmError("CALIBRATION_MISMATCH", `${name} length does not match FFT grid`, {
      stage: "VALIDATING",
      details: { expected, alternateExpected, actual: value.length },
    });
  }
  for (let index = 0; index < value.length; index += 1) assertFinite(value[index]!, `${name}[${index}]`);
}

function validateFieldOfView(
  atoms: Required<InitialAtom>[],
  targets: Required<TargetSite>[],
  staticTraps: Required<StaticTrap>[],
  calibration: CalibrationPackage,
): void {
  const fov = calibration.manifest.fieldOfViewUm;
  if (!fov) return;
  for (const point of [...atoms, ...targets, ...staticTraps]) {
    if (point.xUm < fov.xMinUm || point.xUm > fov.xMaxUm || point.yUm < fov.yMinUm || point.yUm > fov.yMaxUm) {
      throw new SlmError("OUT_OF_BOUNDS", "Point lies outside the calibrated field of view", {
        stage: "VALIDATING",
        details: { point },
      });
    }
  }
}

function validateForbiddenTargets(targets: Required<TargetSite>[], regions: ForbiddenRegion[]): void {
  for (const target of targets) {
    if (regions.some((region) => pointInForbiddenRegion(target, region))) {
      throw new SlmError("INVALID_TARGET_GEOMETRY", `Target site ${target.siteId} lies in a forbidden region`, {
        stage: "VALIDATING",
        details: { siteId: target.siteId },
      });
    }
  }
}

function validateRequiredTargetSeparation(targets: Required<TargetSite>[], minimum: number): void {
  const required = targets.filter((target) => target.required);
  for (let first = 0; first < required.length; first += 1) {
    for (let second = first + 1; second < required.length; second += 1) {
      if (distance(required[first]!, required[second]!) < minimum - 1e-9) {
        throw new SlmError("INVALID_TARGET_GEOMETRY", "Required target sites violate minimum separation", {
          stage: "VALIDATING",
          details: { first: required[first]!.siteId, second: required[second]!.siteId, minimum },
        });
      }
    }
  }
}

function makeAutomaticParkingSites(
  atoms: Required<InitialAtom>[],
  targets: Required<TargetSite>[],
  assignment: Required<AssignmentConfig>,
  planner: Required<PlannerConfig>,
  calibration: CalibrationPackage,
): { xUm: number; yUm: number }[] {
  const points = [...atoms, ...targets];
  const xMin = Math.min(0, ...points.map((point) => point.xUm));
  const xMax = Math.max(0, ...points.map((point) => point.xUm));
  const yMin = Math.min(0, ...points.map((point) => point.yUm));
  const yMax = Math.max(0, ...points.map((point) => point.yUm));
  const gap = Math.max(assignment.stayToleranceUm * 4, planner.minimumSeparationUm * 2);
  const fov = calibration.manifest.fieldOfViewUm;
  const count = Math.max(1, atoms.filter((atom) => atom.movable).length - targets.filter((target) => target.required).length);
  const sites: { xUm: number; yUm: number }[] = [];
  for (let index = 0; index < count; index += 1) {
    let x = xMin + index * gap;
    let y = yMax + gap;
    if (fov) {
      x = Math.min(fov.xMaxUm, Math.max(fov.xMinUm, x));
      y = Math.min(fov.yMaxUm, Math.max(fov.yMinUm, y));
    }
    sites.push({ xUm: x, yUm: y });
  }
  return sites;
}

function validateInitialAtomSeparation(atoms: Required<InitialAtom>[], planner: Required<PlannerConfig>): void {
  for (let first = 0; first < atoms.length; first += 1) {
    for (let second = first + 1; second < atoms.length; second += 1) {
      const a = atoms[first]!;
      const b = atoms[second]!;
      const safe = planner.minimumSeparationUm + planner.kSigma * (a.localizationSigmaUm + b.localizationSigmaUm) + planner.geometricMarginUm;
      if (distance(a, b) < safe - 1e-9) {
        throw new SlmError("INVALID_TARGET_GEOMETRY", "Initial atoms violate the configured safe separation", {
          stage: "VALIDATING",
          details: { first: a.atomId, second: b.atomId, measured: distance(a, b), required: safe },
        });
      }
    }
  }
}

function validateTargetsAgainstStaticGeometry(
  targets: Required<TargetSite>[],
  atoms: Required<InitialAtom>[],
  traps: Required<StaticTrap>[],
  planner: Required<PlannerConfig>,
): void {
  const staticPoints = [
    ...atoms.filter((atom) => !atom.movable),
    ...traps.filter((trap) => trap.containsAtom),
  ];
  const clearance = planner.minimumSeparationUm + planner.geometricMarginUm;
  for (const target of targets.filter((candidate) => candidate.required)) {
    if (staticPoints.some((point) => distance(target, point) < clearance - 1e-9)) {
      throw new SlmError("INVALID_TARGET_GEOMETRY", `Target site ${target.siteId} is too close to a static occupied trap`, {
        stage: "VALIDATING",
        details: { siteId: target.siteId, clearance },
      });
    }
  }
}

function validateStaticGeometrySeparation(
  atoms: Required<InitialAtom>[],
  traps: Required<StaticTrap>[],
  planner: Required<PlannerConfig>,
): void {
  const staticPoints = [...atoms.filter((atom) => !atom.movable), ...traps.filter((trap) => trap.containsAtom)];
  const minimum = planner.minimumSeparationUm + planner.geometricMarginUm;
  for (let first = 0; first < staticPoints.length; first += 1) {
    for (let second = first + 1; second < staticPoints.length; second += 1) {
      if (distance(staticPoints[first]!, staticPoints[second]!) < minimum - 1e-9) {
        throw new SlmError("INVALID_TARGET_GEOMETRY", "Static occupied traps violate minimum separation", {
          stage: "VALIDATING",
          details: { minimum },
        });
      }
    }
  }
  for (const atom of atoms.filter((candidate) => candidate.movable)) {
    if (staticPoints.some((point) => distance(atom, point) < planner.minimumSeparationUm + planner.kSigma * atom.localizationSigmaUm + planner.geometricMarginUm - 1e-9)) {
      throw new SlmError("INVALID_TARGET_GEOMETRY", `Movable atom ${atom.atomId} overlaps a static occupied trap`, {
        stage: "VALIDATING",
        details: { atomId: atom.atomId },
      });
    }
  }
}

function validateParkingSites(
  sites: { xUm: number; yUm: number }[],
  calibration: CalibrationPackage,
): void {
  const fov = calibration.manifest.fieldOfViewUm;
  if (!fov) return;
  for (const site of sites) {
    if (site.xUm < fov.xMinUm || site.xUm > fov.xMaxUm || site.yUm < fov.yMinUm || site.yUm > fov.yMaxUm) {
      throw new SlmError("OUT_OF_BOUNDS", "Parking site lies outside the calibrated field of view", {
        stage: "VALIDATING",
        details: { site },
      });
    }
  }
}

function validateStaticTrapIds(traps: Required<StaticTrap>[]): void {
  const ids = new Set<number>();
  for (const trap of traps) {
    if (ids.has(trap.trapId)) duplicateId("static trap", trap.trapId);
    ids.add(trap.trapId);
  }
}

function validateStaticTrapAtomIds(atoms: Required<InitialAtom>[], traps: Required<StaticTrap>[]): void {
  const atomIds = new Set(atoms.map((atom) => atom.atomId));
  for (const trap of traps) {
    if (trap.atomId >= 0 && atomIds.has(trap.atomId)) {
      throw new SlmError("DUPLICATE_ID", `Atom ${trap.atomId} is claimed by both an initial atom and a static trap`, {
        stage: "VALIDATING",
        details: { atomId: trap.atomId, trapId: trap.trapId },
      });
    }
  }
}

function validatePoint(x: number, y: number, name: string): void {
  assertFinite(x, `${name}.xUm`);
  assertFinite(y, `${name}.yUm`);
}

function validateId(id: number, name: string): void {
  if (!Number.isInteger(id) || id < 0 || id > 0xffffffff || !Number.isSafeInteger(id)) {
    throw new SlmError("INVALID_ARGUMENT", `${name} must be a non-negative uint32`, { stage: "VALIDATING" });
  }
}

function duplicateId(kind: string, id: number): never {
  throw new SlmError("DUPLICATE_ID", `Duplicate ${kind} identifier ${id}`, {
    stage: "VALIDATING",
    details: { kind, id },
  });
}

function validateNonNegative(value: number, name: string): void {
  assertFinite(value, name);
  if (value < 0) throw new SlmError("INVALID_ARGUMENT", `${name} must be non-negative`, { stage: "VALIDATING" });
}

function positive(value: number, name: string): void {
  assertFinite(value, name);
  if (value <= 0) throw new SlmError("INVALID_ARGUMENT", `${name} must be positive`, { stage: "VALIDATING" });
}

function nonNegative(value: number, name: string): void {
  assertFinite(value, name);
  if (value < 0) throw new SlmError("INVALID_ARGUMENT", `${name} must be non-negative`, { stage: "VALIDATING" });
}

function positiveInteger(value: number, name: string): void {
  positive(value, name);
  if (!Number.isInteger(value)) throw new SlmError("INVALID_ARGUMENT", `${name} must be an integer`, { stage: "VALIDATING" });
}

function nonNegativeInteger(value: number, name: string): void {
  nonNegative(value, name);
  if (!Number.isInteger(value)) throw new SlmError("INVALID_ARGUMENT", `${name} must be an integer`, { stage: "VALIDATING" });
}

export function inputHash(request: RearrangementRequest, normalized: NormalizedInput): string {
  return hashValue({
    request,
    atoms: normalized.initialAtoms,
    targets: normalized.targetSites,
    staticTraps: normalized.staticTraps,
    forbiddenRegions: normalized.forbiddenRegions,
    planner: normalized.plannerConfig,
    assignment: normalized.assignmentConfig,
    motion: normalized.motionConfig,
    hologram: normalized.hologramConfig,
    calibrationId: normalized.calibration.manifest.calibrationId,
  });
}

export function calibrationHash(calibration: CalibrationPackage): string {
  return hashValue({
    manifest: calibration.manifest,
    incidentAmplitude: calibration.incidentAmplitude,
    apertureMask: calibration.apertureMask,
    coordinateTransform: calibration.coordinateTransform,
    aberrationPhase: calibration.aberrationPhase,
    phaseResponseLut: calibration.phaseResponseLut,
    inversePhaseLut: calibration.inversePhaseLut,
    carrierGrating: calibration.carrierGrating,
    digitalLens: calibration.digitalLens,
    phaseSigns: calibration.phaseSigns,
  });
}

export const normalizeInput = normalizeAndValidate;
