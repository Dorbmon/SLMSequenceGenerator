// src/errors.ts
var SlmError = class extends Error {
  code;
  stage;
  retryable;
  details;
  constructor(code, message, options = {}) {
    super(message, { cause: options.cause });
    this.name = "SlmError";
    this.code = code;
    this.stage = options.stage ?? "UNKNOWN";
    this.retryable = options.retryable ?? false;
    this.details = options.details ?? {};
  }
};
var SlmCompilationError = class extends SlmError {
  issues;
  constructor(code, message, issues = [], options = {}) {
    super(code, message, options);
    this.name = "SlmCompilationError";
    this.issues = issues;
  }
};
function assertFinite(value, name) {
  if (!Number.isFinite(value)) {
    throw new SlmError("INVALID_ARGUMENT", `${name} must be finite`, {
      stage: "VALIDATING",
      details: { name, value }
    });
  }
}
function checkAbort(signal) {
  if (signal?.aborted) {
    throw new SlmError("CANCELLED", "Compilation was cancelled", {
      stage: "CANCELLED",
      retryable: true
    });
  }
}

// src/util.ts
var TAU = Math.PI * 2;
function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}
function wrapPhase(value) {
  const wrapped = ((value + Math.PI) % TAU + TAU) % TAU - Math.PI;
  return wrapped === -Math.PI ? Math.PI : wrapped;
}
function angularDistance(a, b) {
  return Math.abs(wrapPhase(a - b));
}
function smoothstep5(s) {
  const t = clamp(s, 0, 1);
  return t * t * t * (10 + t * (-15 + 6 * t));
}
function stableStringify(value) {
  return stringifyValue(value);
}
function stringifyValue(value) {
  if (value === null) return "null";
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return JSON.stringify(String(value));
    return JSON.stringify(Object.is(value, -0) ? 0 : value);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "bigint") return JSON.stringify(value.toString());
  if (typeof value === "undefined") return "null";
  if (ArrayBuffer.isView(value)) {
    return `[${Array.from(value, (item) => stringifyValue(item)).join(",")}]`;
  }
  if (Array.isArray(value)) {
    return `[${value.map((item) => stringifyValue(item)).join(",")}]`;
  }
  if (typeof value === "object") {
    const record = value;
    const keys = Object.keys(record).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${stringifyValue(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(String(value));
}
function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
function hashValue(value) {
  return hashString(stableStringify(value));
}
function crc32(bytes) {
  let crc = 4294967295;
  for (let index = 0; index < bytes.length; index += 1) {
    crc ^= bytes[index] & 255;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = crc >>> 1 ^ (crc & 1 ? 3988292384 : 0);
    }
  }
  return (crc ^ 4294967295) >>> 0;
}
function asNumberArray(value, expected, name) {
  if (value.length !== expected) {
    throw new SlmError("CALIBRATION_MISMATCH", `${name} must contain ${expected} values`, {
      stage: "VALIDATING",
      details: { expected, actual: value.length, name }
    });
  }
  const result = Array.from(value, Number);
  if (result.some((item) => !Number.isFinite(item))) {
    throw new SlmError("INVALID_ARGUMENT", `${name} contains a non-finite value`, {
      stage: "VALIDATING"
    });
  }
  return result;
}
function numberOr(value, fallback) {
  return value === void 0 ? fallback : value;
}
function integerOr(value, fallback) {
  const result = numberOr(value, fallback);
  if (!Number.isInteger(result)) {
    throw new SlmError("INVALID_ARGUMENT", `Expected an integer, received ${result}`, {
      stage: "VALIDATING"
    });
  }
  return result;
}
function cloneTyped(value) {
  return new value.constructor(value);
}

// src/geometry.ts
function distanceSquared(a, b) {
  const dx = a.xUm - b.xUm;
  const dy = a.yUm - b.yUm;
  return dx * dx + dy * dy;
}
function distance(a, b) {
  return Math.sqrt(distanceSquared(a, b));
}
function lerpPoint(a, b, t) {
  return { xUm: a.xUm + (b.xUm - a.xUm) * t, yUm: a.yUm + (b.yUm - a.yUm) * t };
}
function distancePointToSegment(point, segment) {
  const dx = segment.end.xUm - segment.start.xUm;
  const dy = segment.end.yUm - segment.start.yUm;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return distance(point, segment.start);
  const t = clamp(
    ((point.xUm - segment.start.xUm) * dx + (point.yUm - segment.start.yUm) * dy) / lengthSquared,
    0,
    1
  );
  return distance(point, lerpPoint(segment.start, segment.end, t));
}
function orientation(a, b, c) {
  return (b.xUm - a.xUm) * (c.yUm - a.yUm) - (b.yUm - a.yUm) * (c.xUm - a.xUm);
}
function onSegment(a, b, point, epsilon = 1e-10) {
  return point.xUm >= Math.min(a.xUm, b.xUm) - epsilon && point.xUm <= Math.max(a.xUm, b.xUm) + epsilon && point.yUm >= Math.min(a.yUm, b.yUm) - epsilon && point.yUm <= Math.max(a.yUm, b.yUm) + epsilon;
}
function segmentsIntersect(a, b, epsilon = 1e-10) {
  const o1 = orientation(a.start, a.end, b.start);
  const o2 = orientation(a.start, a.end, b.end);
  const o3 = orientation(b.start, b.end, a.start);
  const o4 = orientation(b.start, b.end, a.end);
  if ((o1 > epsilon && o2 < -epsilon || o1 < -epsilon && o2 > epsilon) && (o3 > epsilon && o4 < -epsilon || o3 < -epsilon && o4 > epsilon)) {
    return true;
  }
  return Math.abs(o1) <= epsilon && onSegment(a.start, a.end, b.start, epsilon) || Math.abs(o2) <= epsilon && onSegment(a.start, a.end, b.end, epsilon) || Math.abs(o3) <= epsilon && onSegment(b.start, b.end, a.start, epsilon) || Math.abs(o4) <= epsilon && onSegment(b.start, b.end, a.end, epsilon);
}
function segmentDistance(a, b) {
  if (segmentsIntersect(a, b)) return 0;
  return Math.min(
    distancePointToSegment(a.start, b),
    distancePointToSegment(a.end, b),
    distancePointToSegment(b.start, a),
    distancePointToSegment(b.end, a)
  );
}
function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const current = polygon[i];
    const previous = polygon[j];
    if (onSegment(previous, current, point, 1e-10)) return true;
    const crosses = current.yUm > point.yUm !== previous.yUm > point.yUm && point.xUm < (previous.xUm - current.xUm) * (point.yUm - current.yUm) / (previous.yUm - current.yUm) + current.xUm;
    if (crosses) inside = !inside;
  }
  return inside;
}
function regionPolygon(region) {
  if (region.type === "polygon") {
    const points2 = [];
    for (let index = 0; index + 1 < region.coordinates.length; index += 2) {
      points2.push({ xUm: region.coordinates[index], yUm: region.coordinates[index + 1] });
    }
    return points2;
  }
  if (region.type === "axisAlignedBox") {
    const [xMin, yMin, xMax, yMax] = region.coordinates;
    return [
      { xUm: xMin, yUm: yMin },
      { xUm: xMax, yUm: yMin },
      { xUm: xMax, yUm: yMax },
      { xUm: xMin, yUm: yMax }
    ];
  }
  const [x, y, radius] = region.coordinates;
  const count = 32;
  const points = [];
  for (let index = 0; index < count; index += 1) {
    const angle = index / count * Math.PI * 2;
    points.push({ xUm: x + Math.cos(angle) * radius, yUm: y + Math.sin(angle) * radius });
  }
  return points;
}
function distancePointToPolygon(point, polygon) {
  if (polygon.length === 0) return Number.POSITIVE_INFINITY;
  if (pointInPolygon(point, polygon)) return 0;
  let result = Number.POSITIVE_INFINITY;
  for (let index = 0; index < polygon.length; index += 1) {
    const segment = {
      start: polygon[index],
      end: polygon[(index + 1) % polygon.length]
    };
    result = Math.min(result, distancePointToSegment(point, segment));
  }
  return result;
}
function pointInForbiddenRegion(point, region) {
  const clearance = Math.max(0, region.clearanceUm ?? 0);
  if (region.type === "circle") {
    const [x, y, radius] = region.coordinates;
    return Math.hypot(point.xUm - x, point.yUm - y) <= radius + clearance;
  }
  const polygon = regionPolygon(region);
  return distancePointToPolygon(point, polygon) <= clearance || pointInPolygon(point, polygon);
}
function segmentIntersectsForbiddenRegion(segment, region, samples = 32) {
  if (region.type === "circle") {
    const [x, y, radius] = region.coordinates;
    return distancePointToSegment(
      { xUm: x, yUm: y },
      segment
    ) <= radius + Math.max(0, region.clearanceUm ?? 0);
  }
  if (pointInForbiddenRegion(segment.start, region) || pointInForbiddenRegion(segment.end, region)) {
    return true;
  }
  const polygon = regionPolygon(region);
  for (let index = 0; index < polygon.length; index += 1) {
    const edge = { start: polygon[index], end: polygon[(index + 1) % polygon.length] };
    if (segmentsIntersect(segment, edge)) return true;
    const clearance = Math.max(0, region.clearanceUm ?? 0);
    if (clearance > 0 && segmentDistance(segment, edge) <= clearance) return true;
  }
  for (let index = 1; index < samples; index += 1) {
    if (pointInForbiddenRegion(segmentPoint(segment, index / samples), region)) return true;
  }
  return false;
}
function segmentPoint(segment, fraction) {
  return lerpPoint(segment.start, segment.end, fraction);
}
function pathIntersectsForbiddenRegion(path, regions) {
  for (const point of path) {
    if (regions.some((region) => pointInForbiddenRegion(point, region))) return true;
  }
  for (let index = 0; index + 1 < path.length; index += 1) {
    const segment = { start: path[index], end: path[index + 1] };
    if (regions.some((region) => segmentIntersectsForbiddenRegion(segment, region))) return true;
  }
  return false;
}
function pointClearOfStaticAtoms(point, staticPoints, clearance) {
  return staticPoints.every((staticPoint) => distance(point, staticPoint) >= clearance - 1e-9);
}
function pathClearOfStaticAtoms(path, staticPoints, clearance) {
  for (const point of staticPoints) {
    for (let index = 0; index + 1 < path.length; index += 1) {
      if (distancePointToSegment(point, { start: path[index], end: path[index + 1] }) < clearance - 1e-9) {
        return false;
      }
    }
    if (path.some((pathPoint) => distance(pathPoint, point) < clearance - 1e-9)) return false;
  }
  return true;
}
function boundingBox(points, padding = 0) {
  if (points.length === 0) return { xMin: -padding, xMax: padding, yMin: -padding, yMax: padding };
  return {
    xMin: Math.min(...points.map((point) => point.xUm)) - padding,
    xMax: Math.max(...points.map((point) => point.xUm)) + padding,
    yMin: Math.min(...points.map((point) => point.yUm)) - padding,
    yMax: Math.max(...points.map((point) => point.yUm)) + padding
  };
}

// src/normalization.ts
var DEFAULT_PLANNER_CONFIG = {
  minimumSeparationUm: 1,
  kSigma: 3,
  geometricMarginUm: 0.1,
  duplicatePointToleranceUm: 0,
  duplicatePointPolicy: "reject",
  gridResolutionUm: 0.5,
  planningTickUs: 1e3,
  maxSearchTicks: 512,
  maxAStarExpansions: 5e4,
  maxCbsNodes: 1e3,
  ecbsLimit: 12,
  maxPriorityRetries: 4
};
var DEFAULT_ASSIGNMENT_CONFIG = {
  distanceWeight: 1,
  obstacleWeight: 1e3,
  staticObstacleWeight: 100,
  groupMismatchPenalty: 1e3,
  groupMismatchPolicy: "forbid",
  extraAtomPolicy: "KEEP",
  parkingSites: [],
  stayToleranceUm: 0.25,
  maxAssignmentRetries: 2,
  conflictPenalty: 1e3
};
var DEFAULT_MOTION_CONFIG = {
  framePeriodUs: 1e3,
  preMoveDwellUs: 1e3,
  postMoveSettleUs: 1e3,
  minDwellBeforeMoveUs: 0,
  maxVelocityUmPerUs: 0.05,
  maxAccelerationUmPerUs2: 1e-3,
  maxJerkUmPerUs3: 1e-6,
  maxPositionChangePerFrameUm: Number.POSITIVE_INFINITY,
  maxIntensityChangePerFrame: Number.POSITIVE_INFINITY,
  movingTrapIntensity: 1,
  defaultTrapIntensity: 1,
  maxValidationDepth: 12
};
var DEFAULT_HOLOGRAM_CONFIG = {
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
  requireConvergence: false
};
function normalizeAndValidate(request, compilerConfig = {}) {
  if (!request || !Array.isArray(request.targetSites)) {
    throw new SlmError("INVALID_ARGUMENT", "targetSites must be an array", { stage: "VALIDATING" });
  }
  const plannerConfig = normalizePlannerConfig({
    ...DEFAULT_PLANNER_CONFIG,
    ...compilerConfig.planner,
    ...request.plannerConfig
  });
  const assignmentOverrides = { ...compilerConfig.assignment, ...request.assignmentConfig };
  const assignmentConfig = normalizeAssignmentConfig({
    ...DEFAULT_ASSIGNMENT_CONFIG,
    ...assignmentOverrides,
    ...assignmentOverrides.extraAtomPolicy === void 0 && assignmentOverrides.parkingSites && assignmentOverrides.parkingSites.length > 0 ? { extraAtomPolicy: "PARK_AND_RELEASE" } : {}
  });
  const motionConfig = normalizeMotionConfig({
    ...DEFAULT_MOTION_CONFIG,
    ...compilerConfig.motion,
    ...request.motionConfig
  });
  const calibration = resolveCalibration(request, compilerConfig, motionConfig, plannerConfig);
  const hologramConfig = normalizeHologramConfig({
    ...DEFAULT_HOLOGRAM_CONFIG,
    width: calibration.manifest.fftWidth ?? calibration.manifest.activeWidth,
    height: calibration.manifest.fftHeight ?? calibration.manifest.activeHeight,
    ...compilerConfig.hologram,
    ...request.hologramConfig
  });
  validateCalibration(calibration, hologramConfig, compilerConfig.simulationMode === true, compilerConfig);
  const requestedWavelength = request.wavelengthNm ?? compilerConfig.wavelengthNm;
  if (requestedWavelength !== void 0 && requestedWavelength !== calibration.manifest.wavelengthNm) {
    throw new SlmError("CALIBRATION_MISMATCH", "Calibration wavelength does not match the requested wavelength", {
      stage: "VALIDATING",
      details: { requestedWavelength, calibrationWavelength: calibration.manifest.wavelengthNm }
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
          measured: movableCount
        }
      ],
      { stage: "VALIDATING" }
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
    hologramConfig
  };
}
function normalizeLatticeOrAtoms(input) {
  if (Array.isArray(input)) return input.map((atom) => ({ ...atom }));
  const lattice = input;
  if (!lattice || !Array.isArray(lattice.sites) || !(lattice.occupied instanceof Uint8Array)) {
    throw new SlmError("INVALID_ARGUMENT", "initialAtoms must be atoms or an occupied lattice", {
      stage: "VALIDATING"
    });
  }
  if (lattice.sites.length !== lattice.occupied.length) {
    throw new SlmError("INVALID_ARGUMENT", "lattice sites and occupancy lengths differ", {
      stage: "VALIDATING"
    });
  }
  return lattice.sites.flatMap(
    (point, index) => lattice.occupied[index] ? [{ xUm: point.xUm, yUm: point.yUm }] : []
  );
}
function normalizeAtoms(atoms, planner) {
  const usedIds = /* @__PURE__ */ new Set();
  const result = [];
  for (let index = 0; index < atoms.length; index += 1) {
    const atom = atoms[index];
    validatePoint(atom.xUm, atom.yUm, `initialAtoms[${index}]`);
    const atomId = atom.atomId ?? index;
    validateId(atomId, `initialAtoms[${index}].atomId`);
    if (usedIds.has(atomId)) duplicateId("atom", atomId);
    usedIds.add(atomId);
    const normalized = {
      atomId,
      xUm: atom.xUm,
      yUm: atom.yUm,
      group: atom.group ?? 0,
      movable: atom.movable ?? true,
      initialTrapIntensity: atom.initialTrapIntensity ?? 1,
      localizationSigmaUm: atom.localizationSigmaUm ?? 0
    };
    validateId(normalized.group, `initialAtoms[${index}].group`);
    if (typeof normalized.movable !== "boolean") throw new SlmError("INVALID_ARGUMENT", "movable must be boolean", { stage: "VALIDATING" });
    validateNonNegative(normalized.initialTrapIntensity, "initialTrapIntensity");
    validateNonNegative(normalized.localizationSigmaUm, "localizationSigmaUm");
    result.push(normalized);
  }
  return mergeDuplicatePoints(result, planner.duplicatePointToleranceUm, planner.duplicatePointPolicy);
}
function normalizeTargets(targets, planner) {
  const usedIds = /* @__PURE__ */ new Set();
  const result = [];
  for (let index = 0; index < targets.length; index += 1) {
    const target = targets[index];
    validatePoint(target.xUm, target.yUm, `targetSites[${index}]`);
    const siteId = target.siteId ?? index;
    validateId(siteId, `targetSites[${index}].siteId`);
    if (usedIds.has(siteId)) duplicateId("target site", siteId);
    usedIds.add(siteId);
    const normalized = {
      siteId,
      xUm: target.xUm,
      yUm: target.yUm,
      required: target.required ?? true,
      requiredAtomId: target.requiredAtomId ?? -1,
      requiredGroup: target.requiredGroup ?? -1,
      finalTrapIntensity: target.finalTrapIntensity ?? 1
    };
    if (normalized.requiredAtomId !== -1) validateId(normalized.requiredAtomId, "requiredAtomId");
    if (normalized.requiredGroup !== -1) validateId(normalized.requiredGroup, "requiredGroup");
    if (typeof normalized.required !== "boolean") throw new SlmError("INVALID_ARGUMENT", "required must be boolean", { stage: "VALIDATING" });
    validateNonNegative(normalized.finalTrapIntensity, "finalTrapIntensity");
    result.push(normalized);
  }
  return mergeDuplicatePoints(result, planner.duplicatePointToleranceUm, planner.duplicatePointPolicy);
}
function normalizeStaticTraps(traps, planner) {
  const usedIds = /* @__PURE__ */ new Set();
  const result = [];
  for (let index = 0; index < traps.length; index += 1) {
    const trap = traps[index];
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
      atomId: trap.atomId ?? -1
    });
    if (trap.atomId !== void 0) validateId(trap.atomId, `staticTraps[${index}].atomId`);
  }
  return mergeDuplicatePoints(result, planner.duplicatePointToleranceUm, planner.duplicatePointPolicy);
}
function normalizeForbiddenRegions(regions) {
  return regions.map((region, index) => {
    if (!region || !["circle", "axisAlignedBox", "polygon"].includes(region.type) || !Array.isArray(region.coordinates)) {
      throw new SlmError("INVALID_ARGUMENT", `Invalid forbidden region at index ${index}`, {
        stage: "VALIDATING"
      });
    }
    const expected = region.type === "circle" ? 3 : region.type === "axisAlignedBox" ? 4 : 6;
    if (region.type === "polygon" && (region.coordinates.length < 6 || region.coordinates.length % 2 !== 0)) {
      throw new SlmError("INVALID_ARGUMENT", `Polygon ${index} needs at least three points`, {
        stage: "VALIDATING"
      });
    }
    if (region.type !== "polygon" && region.coordinates.length !== expected) {
      throw new SlmError("INVALID_ARGUMENT", `Forbidden region ${index} has invalid coordinates`, {
        stage: "VALIDATING"
      });
    }
    region.coordinates.forEach(
      (value, coordinateIndex) => assertFinite(value, `forbiddenRegions[${index}].coordinates[${coordinateIndex}]`)
    );
    if (region.type === "circle" && region.coordinates[2] < 0) {
      throw new SlmError("INVALID_ARGUMENT", "Circle radius cannot be negative", { stage: "VALIDATING" });
    }
    if (region.type === "axisAlignedBox" && (region.coordinates[0] > region.coordinates[2] || region.coordinates[1] > region.coordinates[3])) {
      throw new SlmError("INVALID_ARGUMENT", "Axis-aligned box bounds are inverted", { stage: "VALIDATING" });
    }
    if (region.clearanceUm !== void 0) validateNonNegative(region.clearanceUm, "region clearanceUm");
    return { type: region.type, coordinates: [...region.coordinates], ...region.clearanceUm === void 0 ? {} : { clearanceUm: region.clearanceUm } };
  });
}
function mergeDuplicatePoints(points, tolerance, policy) {
  if (tolerance <= 0 || points.length < 2) return points;
  const result = [];
  for (const point of points) {
    const duplicateIndex = result.findIndex((other) => distance(point, other) <= tolerance);
    if (duplicateIndex < 0) {
      result.push(point);
    } else if (policy === "reject") {
      throw new SlmError("INVALID_TARGET_GEOMETRY", "Duplicate points violate duplicatePointToleranceUm", {
        stage: "VALIDATING",
        details: { tolerance }
      });
    }
  }
  return result;
}
function normalizePlannerConfig(config) {
  const result = config;
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
function normalizeAssignmentConfig(config) {
  const result = config;
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
    "RELEASE_IN_PLACE"
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
function normalizeMotionConfig(config) {
  const result = config;
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
function normalizeHologramConfig(config) {
  const result = config;
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
function resolveCalibration(request, config, motion, planner) {
  const calibration = request.calibration ?? (request.calibrationId && config.calibrations?.[request.calibrationId]) ?? config.calibration;
  if (calibration) {
    if (!calibration.manifest || request.calibrationId && calibration.manifest.calibrationId !== request.calibrationId) {
      throw new SlmError("CALIBRATION_MISMATCH", "Requested calibrationId does not match calibration package", {
        stage: "VALIDATING"
      });
    }
    validateCalibrationManifest(calibration);
    return calibration;
  }
  if (config.simulationMode !== true) {
    throw new SlmError("CALIBRATION_MISMATCH", "A measured calibration package is required", { stage: "VALIDATING" });
  }
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
      coordinateConvention: "+x right, +y up"
    },
    coordinateTransform: { originXUm: width / 2, originYUm: height / 2, scaleX: 1, scaleY: 1 }
  };
}
function validateCalibrationManifest(calibration) {
  const manifest = calibration?.manifest;
  if (!manifest || typeof manifest.calibrationId !== "string" || manifest.calibrationId.length === 0) {
    throw new SlmError("CALIBRATION_MISMATCH", "Calibration manifest needs a calibrationId", { stage: "VALIDATING" });
  }
  positiveInteger(manifest.activeWidth, "calibration activeWidth");
  positiveInteger(manifest.activeHeight, "calibration activeHeight");
  if (manifest.fftWidth !== void 0) positiveInteger(manifest.fftWidth, "calibration fftWidth");
  if (manifest.fftHeight !== void 0) positiveInteger(manifest.fftHeight, "calibration fftHeight");
  positive(manifest.wavelengthNm, "calibration wavelengthNm");
  if (manifest.pixelPitchUm !== void 0) positive(manifest.pixelPitchUm, "calibration pixelPitchUm");
  if (manifest.fftWidth !== void 0 && manifest.fftHeight === void 0 || manifest.fftHeight !== void 0 && manifest.fftWidth === void 0) {
    throw new SlmError("CALIBRATION_MISMATCH", "Both FFT dimensions must be specified", { stage: "VALIDATING" });
  }
  if ((manifest.fftWidth ?? manifest.activeWidth) < manifest.activeWidth || (manifest.fftHeight ?? manifest.activeHeight) < manifest.activeHeight) {
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
function validateCalibration(calibration, hologram, simulationMode, compilerConfig = {}) {
  validateCalibrationManifest(calibration);
  const manifest = calibration.manifest;
  const expectedWidth = manifest.fftWidth ?? manifest.activeWidth;
  const expectedHeight = manifest.fftHeight ?? manifest.activeHeight;
  if (hologram.width !== expectedWidth || hologram.height !== expectedHeight) {
    throw new SlmError("CALIBRATION_MISMATCH", "Hologram dimensions do not match the calibration FFT grid", {
      stage: "VALIDATING",
      details: { expectedWidth, expectedHeight, width: hologram.width, height: hologram.height }
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
    if (requirements.slmModel !== void 0 && requirements.slmModel !== manifest.slmModel) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration SLM model does not match the requested device", { stage: "VALIDATING" });
    }
    if (requirements.serialNumber !== void 0 && requirements.serialNumber !== manifest.serialNumber) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration serial number does not match the requested device", { stage: "VALIDATING" });
    }
    if (requirements.wavelengthNm !== void 0 && requirements.wavelengthNm !== manifest.wavelengthNm) {
      throw new SlmError("CALIBRATION_MISMATCH", "Calibration wavelength does not match the requested device", { stage: "VALIDATING" });
    }
  }
  validateCalibrationChecksums(calibration, compilerConfig.strictCalibration === true);
  const transform = calibration.coordinateTransform;
  if (transform) {
    for (const value of [transform.originXUm, transform.originYUm, transform.scaleX, transform.scaleY, transform.rotationRad, transform.offsetX, transform.offsetY, transform.a, transform.b, transform.c, transform.d]) {
      if (value !== void 0) assertFinite(value, "coordinate transform");
    }
  }
  if (calibration.phaseSigns) {
    for (const sign of [calibration.phaseSigns.aberration, calibration.phaseSigns.grating, calibration.phaseSigns.lens]) {
      if (sign !== void 0 && sign !== 1 && sign !== -1) {
        throw new SlmError("CALIBRATION_MISMATCH", "Phase correction signs must be +1 or -1", { stage: "VALIDATING" });
      }
    }
  }
}
function validateCalibrationPackage(calibration, options = {}) {
  validateCalibrationManifest(calibration);
  const hologram = normalizeHologramConfig({
    ...DEFAULT_HOLOGRAM_CONFIG,
    width: options.width ?? calibration.manifest.fftWidth ?? calibration.manifest.activeWidth,
    height: options.height ?? calibration.manifest.fftHeight ?? calibration.manifest.activeHeight
  });
  validateCalibration(calibration, hologram, options.simulationMode === true);
}
function validateCalibrationChecksums(calibration, strict) {
  const checksums = calibration.manifest.checksums;
  const arrays = [
    ["incidentAmplitude", calibration.incidentAmplitude],
    ["apertureMask", calibration.apertureMask],
    ["aberrationPhase", calibration.aberrationPhase],
    ["phaseResponseLut", calibration.phaseResponseLut],
    ["inversePhaseLut", calibration.inversePhaseLut],
    ["carrierGrating", calibration.carrierGrating],
    ["digitalLens", calibration.digitalLens]
  ];
  for (const [name, value] of arrays) {
    if (value === void 0) continue;
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
function validateCalibrationLut(value, name) {
  if (!value) return;
  for (let index = 0; index < value.length; index += 1) assertFinite(value[index], `${name}[${index}]`);
}
function validateCalibrationNonNegative(value, name) {
  if (value === void 0) return;
  if (typeof value === "number") {
    if (value < 0) throw new SlmError("CALIBRATION_MISMATCH", `${name} cannot be negative`, { stage: "VALIDATING" });
    return;
  }
  for (let index = 0; index < value.length; index += 1) {
    if (value[index] < 0) throw new SlmError("CALIBRATION_MISMATCH", `${name} cannot be negative`, { stage: "VALIDATING" });
  }
}
function validateQualityGates(gates) {
  if (!gates) throw new SlmError("INVALID_ARGUMENT", "qualityGates must be an object", { stage: "VALIDATING" });
  const values = [
    ["maxIntensityCoefficientOfVariation", gates.maxIntensityCoefficientOfVariation],
    ["minIntensityToMeanRatio", gates.minIntensityToMeanRatio],
    ["minDiffractionEfficiency", gates.minDiffractionEfficiency],
    ["maxGhostIntensity", gates.maxGhostIntensity],
    ["maxTargetPhaseErrorRad", gates.maxTargetPhaseErrorRad],
    ["maxPhaseChangeRad", gates.maxPhaseChangeRad],
    ["maxDisplayCodeChange", gates.maxDisplayCodeChange]
  ];
  for (const [name, value] of values) if (value !== void 0) nonNegative(value, name);
}
function validateCalibrationArray(value, expected, name, alternateExpected) {
  if (value === void 0 || typeof value === "number") {
    if (typeof value === "number") assertFinite(value, name);
    return;
  }
  if (value.length !== expected && value.length !== alternateExpected) {
    throw new SlmError("CALIBRATION_MISMATCH", `${name} length does not match FFT grid`, {
      stage: "VALIDATING",
      details: { expected, alternateExpected, actual: value.length }
    });
  }
  for (let index = 0; index < value.length; index += 1) assertFinite(value[index], `${name}[${index}]`);
}
function validateFieldOfView(atoms, targets, staticTraps, calibration) {
  const fov = calibration.manifest.fieldOfViewUm;
  if (!fov) return;
  for (const point of [...atoms, ...targets, ...staticTraps]) {
    if (point.xUm < fov.xMinUm || point.xUm > fov.xMaxUm || point.yUm < fov.yMinUm || point.yUm > fov.yMaxUm) {
      throw new SlmError("OUT_OF_BOUNDS", "Point lies outside the calibrated field of view", {
        stage: "VALIDATING",
        details: { point }
      });
    }
  }
}
function validateForbiddenTargets(targets, regions) {
  for (const target of targets) {
    if (regions.some((region) => pointInForbiddenRegion(target, region))) {
      throw new SlmError("INVALID_TARGET_GEOMETRY", `Target site ${target.siteId} lies in a forbidden region`, {
        stage: "VALIDATING",
        details: { siteId: target.siteId }
      });
    }
  }
}
function validateRequiredTargetSeparation(targets, minimum) {
  const required = targets.filter((target) => target.required);
  for (let first = 0; first < required.length; first += 1) {
    for (let second = first + 1; second < required.length; second += 1) {
      if (distance(required[first], required[second]) < minimum - 1e-9) {
        throw new SlmError("INVALID_TARGET_GEOMETRY", "Required target sites violate minimum separation", {
          stage: "VALIDATING",
          details: { first: required[first].siteId, second: required[second].siteId, minimum }
        });
      }
    }
  }
}
function makeAutomaticParkingSites(atoms, targets, assignment, planner, calibration) {
  const points = [...atoms, ...targets];
  const xMin = Math.min(0, ...points.map((point) => point.xUm));
  const xMax = Math.max(0, ...points.map((point) => point.xUm));
  const yMin = Math.min(0, ...points.map((point) => point.yUm));
  const yMax = Math.max(0, ...points.map((point) => point.yUm));
  const gap = Math.max(assignment.stayToleranceUm * 4, planner.minimumSeparationUm * 2);
  const fov = calibration.manifest.fieldOfViewUm;
  const count = Math.max(1, atoms.filter((atom) => atom.movable).length - targets.filter((target) => target.required).length);
  const sites = [];
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
function validateInitialAtomSeparation(atoms, planner) {
  for (let first = 0; first < atoms.length; first += 1) {
    for (let second = first + 1; second < atoms.length; second += 1) {
      const a = atoms[first];
      const b = atoms[second];
      const safe = planner.minimumSeparationUm + planner.kSigma * (a.localizationSigmaUm + b.localizationSigmaUm) + planner.geometricMarginUm;
      if (distance(a, b) < safe - 1e-9) {
        throw new SlmError("INVALID_TARGET_GEOMETRY", "Initial atoms violate the configured safe separation", {
          stage: "VALIDATING",
          details: { first: a.atomId, second: b.atomId, measured: distance(a, b), required: safe }
        });
      }
    }
  }
}
function validateTargetsAgainstStaticGeometry(targets, atoms, traps, planner) {
  const staticPoints = [
    ...atoms.filter((atom) => !atom.movable),
    ...traps.filter((trap) => trap.containsAtom)
  ];
  const clearance = planner.minimumSeparationUm + planner.geometricMarginUm;
  for (const target of targets.filter((candidate) => candidate.required)) {
    if (staticPoints.some((point) => distance(target, point) < clearance - 1e-9)) {
      throw new SlmError("INVALID_TARGET_GEOMETRY", `Target site ${target.siteId} is too close to a static occupied trap`, {
        stage: "VALIDATING",
        details: { siteId: target.siteId, clearance }
      });
    }
  }
}
function validateStaticGeometrySeparation(atoms, traps, planner) {
  const staticPoints = [...atoms.filter((atom) => !atom.movable), ...traps.filter((trap) => trap.containsAtom)];
  const minimum = planner.minimumSeparationUm + planner.geometricMarginUm;
  for (let first = 0; first < staticPoints.length; first += 1) {
    for (let second = first + 1; second < staticPoints.length; second += 1) {
      if (distance(staticPoints[first], staticPoints[second]) < minimum - 1e-9) {
        throw new SlmError("INVALID_TARGET_GEOMETRY", "Static occupied traps violate minimum separation", {
          stage: "VALIDATING",
          details: { minimum }
        });
      }
    }
  }
  for (const atom of atoms.filter((candidate) => candidate.movable)) {
    if (staticPoints.some((point) => distance(atom, point) < planner.minimumSeparationUm + planner.kSigma * atom.localizationSigmaUm + planner.geometricMarginUm - 1e-9)) {
      throw new SlmError("INVALID_TARGET_GEOMETRY", `Movable atom ${atom.atomId} overlaps a static occupied trap`, {
        stage: "VALIDATING",
        details: { atomId: atom.atomId }
      });
    }
  }
}
function validateParkingSites(sites, calibration) {
  const fov = calibration.manifest.fieldOfViewUm;
  if (!fov) return;
  for (const site of sites) {
    if (site.xUm < fov.xMinUm || site.xUm > fov.xMaxUm || site.yUm < fov.yMinUm || site.yUm > fov.yMaxUm) {
      throw new SlmError("OUT_OF_BOUNDS", "Parking site lies outside the calibrated field of view", {
        stage: "VALIDATING",
        details: { site }
      });
    }
  }
}
function validateStaticTrapIds(traps) {
  const ids = /* @__PURE__ */ new Set();
  for (const trap of traps) {
    if (ids.has(trap.trapId)) duplicateId("static trap", trap.trapId);
    ids.add(trap.trapId);
  }
}
function validateStaticTrapAtomIds(atoms, traps) {
  const atomIds = new Set(atoms.map((atom) => atom.atomId));
  for (const trap of traps) {
    if (trap.atomId >= 0 && atomIds.has(trap.atomId)) {
      throw new SlmError("DUPLICATE_ID", `Atom ${trap.atomId} is claimed by both an initial atom and a static trap`, {
        stage: "VALIDATING",
        details: { atomId: trap.atomId, trapId: trap.trapId }
      });
    }
  }
}
function validatePoint(x, y, name) {
  assertFinite(x, `${name}.xUm`);
  assertFinite(y, `${name}.yUm`);
}
function validateId(id, name) {
  if (!Number.isInteger(id) || id < 0 || id > 4294967295 || !Number.isSafeInteger(id)) {
    throw new SlmError("INVALID_ARGUMENT", `${name} must be a non-negative uint32`, { stage: "VALIDATING" });
  }
}
function duplicateId(kind, id) {
  throw new SlmError("DUPLICATE_ID", `Duplicate ${kind} identifier ${id}`, {
    stage: "VALIDATING",
    details: { kind, id }
  });
}
function validateNonNegative(value, name) {
  assertFinite(value, name);
  if (value < 0) throw new SlmError("INVALID_ARGUMENT", `${name} must be non-negative`, { stage: "VALIDATING" });
}
function positive(value, name) {
  assertFinite(value, name);
  if (value <= 0) throw new SlmError("INVALID_ARGUMENT", `${name} must be positive`, { stage: "VALIDATING" });
}
function nonNegative(value, name) {
  assertFinite(value, name);
  if (value < 0) throw new SlmError("INVALID_ARGUMENT", `${name} must be non-negative`, { stage: "VALIDATING" });
}
function positiveInteger(value, name) {
  positive(value, name);
  if (!Number.isInteger(value)) throw new SlmError("INVALID_ARGUMENT", `${name} must be an integer`, { stage: "VALIDATING" });
}
function nonNegativeInteger(value, name) {
  nonNegative(value, name);
  if (!Number.isInteger(value)) throw new SlmError("INVALID_ARGUMENT", `${name} must be an integer`, { stage: "VALIDATING" });
}
function inputHash(request, normalized) {
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
    calibrationId: normalized.calibration.manifest.calibrationId
  });
}
function calibrationHash(calibration) {
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
    phaseSigns: calibration.phaseSigns
  });
}
var normalizeInput = normalizeAndValidate;

// src/assignment.ts
var ASSIGNMENT_INFINITY = 1e30;
function hungarianSolve(costMatrix) {
  const rows = costMatrix.length;
  if (rows === 0) return { assignment: [], cost: 0, feasible: true };
  const columns = costMatrix[0]?.length ?? 0;
  if (columns === 0 || costMatrix.some((row) => row.length !== columns)) {
    throw new SlmError("INVALID_ARGUMENT", "Hungarian cost matrix must be rectangular and non-empty", {
      stage: "ASSIGNING"
    });
  }
  if (rows > columns) {
    return { assignment: Array(rows).fill(-1), cost: Number.POSITIVE_INFINITY, feasible: false };
  }
  const u = Array(rows + 1).fill(0);
  const v = Array(columns + 1).fill(0);
  const p = Array(columns + 1).fill(0);
  const way = Array(columns + 1).fill(0);
  for (let row = 1; row <= rows; row += 1) {
    p[0] = row;
    let columnZero = 0;
    const minValue = Array(columns + 1).fill(ASSIGNMENT_INFINITY);
    const used = Array(columns + 1).fill(false);
    do {
      used[columnZero] = true;
      const rowZero = p[columnZero];
      let delta = ASSIGNMENT_INFINITY;
      let nextColumn = 0;
      for (let column = 1; column <= columns; column += 1) {
        if (used[column]) continue;
        const matrixValue = costMatrix[rowZero - 1][column - 1];
        const reduced = matrixValue - u[rowZero] - v[column];
        if (reduced < minValue[column]) {
          minValue[column] = reduced;
          way[column] = columnZero;
        }
        if (minValue[column] < delta || minValue[column] === delta && column < nextColumn) {
          delta = minValue[column];
          nextColumn = column;
        }
      }
      if (!Number.isFinite(delta) || delta >= ASSIGNMENT_INFINITY / 2) {
        return { assignment: Array(rows).fill(-1), cost: Number.POSITIVE_INFINITY, feasible: false };
      }
      for (let column = 0; column <= columns; column += 1) {
        if (used[column]) {
          const assignedRow = p[column] ?? 0;
          u[assignedRow] = (u[assignedRow] ?? 0) + delta;
          v[column] = (v[column] ?? 0) - delta;
        } else {
          minValue[column] = minValue[column] - delta;
        }
      }
      columnZero = nextColumn;
    } while (p[columnZero] !== 0);
    do {
      const previousColumn = way[columnZero];
      p[columnZero] = p[previousColumn];
      columnZero = previousColumn;
    } while (columnZero !== 0);
  }
  const assignment = Array(rows).fill(-1);
  for (let column = 1; column <= columns; column += 1) {
    const assignedRow = p[column] ?? 0;
    if (assignedRow > 0) assignment[assignedRow - 1] = column - 1;
  }
  let cost = 0;
  for (let row = 0; row < rows; row += 1) {
    const column = assignment[row];
    if (column < 0 || costMatrix[row][column] >= ASSIGNMENT_INFINITY / 2) {
      return { assignment, cost: Number.POSITIVE_INFINITY, feasible: false };
    }
    cost += costMatrix[row][column];
  }
  return { assignment, cost, feasible: Number.isFinite(cost) };
}
function buildAssignmentCostMatrix(problem) {
  const config = {
    distanceWeight: 1,
    obstacleWeight: 1e3,
    staticObstacleWeight: 100,
    groupMismatchPenalty: 1e3,
    groupMismatchPolicy: "forbid",
    extraAtomPolicy: "KEEP",
    parkingSites: [],
    stayToleranceUm: 0.25,
    maxAssignmentRetries: 2,
    conflictPenalty: 1e3,
    ...problem.config
  };
  const targets = problem.targets.filter((target) => target.required !== false);
  const dummyCount = Math.max(0, problem.atoms.length - targets.length);
  const columns = targets.length + dummyCount;
  if (problem.atoms.length > 0 && columns === 0) {
    throw new SlmError("ASSIGNMENT_INFEASIBLE", "No destination columns are available", { stage: "ASSIGNING" });
  }
  const matrix = problem.atoms.map((atom, atomIndex) => {
    const targetCosts = targets.map(
      (target, targetIndex) => assignmentEdgeCost(atom, target, atomIndex, targetIndex, problem, config)
    );
    const extraCosts = Array.from(
      { length: dummyCount },
      (_, dummyIndex) => dummyDestinationCost(atom, dummyIndex, problem, config)
    );
    return [...targetCosts, ...extraCosts];
  });
  return { matrix, targets, dummyCount };
}
function assignAtoms(input, conflictPenalties = /* @__PURE__ */ new Map()) {
  const movable = input.initialAtoms.filter((atom) => atom.movable);
  const staticAtoms = input.initialAtoms.filter((atom) => !atom.movable);
  const staticPoints = [
    ...staticAtoms,
    ...input.staticTraps.filter((trap) => trap.containsAtom)
  ];
  const { matrix, targets, dummyCount } = buildAssignmentCostMatrix({
    atoms: movable,
    targets: input.targetSites,
    forbiddenRegions: input.forbiddenRegions,
    staticPoints,
    config: input.assignmentConfig,
    conflictPenalties
  });
  const solved = hungarianSolve(matrix);
  if (!solved.feasible) {
    throw new SlmError("ASSIGNMENT_INFEASIBLE", "No assignment satisfies the identity and group constraints", {
      stage: "ASSIGNING",
      details: { matrix }
    });
  }
  const result = [];
  for (const atom of staticAtoms) {
    const sourceIndex = input.initialAtoms.indexOf(atom);
    result.push({
      atomId: atom.atomId,
      sourceIndex,
      targetSiteId: null,
      targetIndex: null,
      disposition: "STAY",
      assignmentCost: 0
    });
  }
  for (let movableIndex = 0; movableIndex < movable.length; movableIndex += 1) {
    const atom = movable[movableIndex];
    const column = solved.assignment[movableIndex];
    const sourceIndex = input.initialAtoms.indexOf(atom);
    if (column >= 0 && column < targets.length) {
      const target = targets[column];
      const targetIndex = input.targetSites.findIndex((candidate) => candidate.siteId === target.siteId);
      const alreadyThere = distance(atom, target) <= input.assignmentConfig.stayToleranceUm;
      result.push({
        atomId: atom.atomId,
        sourceIndex,
        targetSiteId: target.siteId,
        targetIndex,
        disposition: alreadyThere ? "STAY" : "MOVE_TO_TARGET",
        assignmentCost: matrix[movableIndex][column]
      });
    } else if (column >= targets.length && column < targets.length + dummyCount) {
      const policy = input.assignmentConfig.extraAtomPolicy;
      result.push({
        atomId: atom.atomId,
        sourceIndex,
        targetSiteId: null,
        targetIndex: null,
        disposition: policy === "KEEP" ? "KEEP" : policy === "RELEASE_IN_PLACE" ? "RELEASE" : "PARK",
        assignmentCost: matrix[movableIndex][column],
        ...policy === "PARK" || policy === "PARK_AND_RELEASE" ? { parkingSiteIndex: (column - targets.length) % Math.max(1, input.assignmentConfig.parkingSites.length) } : {}
      });
    } else {
      throw new SlmError("ASSIGNMENT_INFEASIBLE", "Hungarian solver returned an incomplete assignment", {
        stage: "ASSIGNING"
      });
    }
  }
  return result.sort((a, b) => a.sourceIndex - b.sourceIndex);
}
function assignmentEdgeCost(atom, target, atomIndex, targetIndex, problem, config) {
  if (target.requiredAtomId !== void 0 && target.requiredAtomId !== -1 && target.requiredAtomId !== atom.atomId) {
    return ASSIGNMENT_INFINITY;
  }
  if (target.requiredGroup !== void 0 && target.requiredGroup !== -1 && target.requiredGroup !== atom.group) {
    if (config.groupMismatchPolicy === "forbid") return ASSIGNMENT_INFINITY;
  }
  const path = [
    { xUm: atom.xUm, yUm: atom.yUm },
    { xUm: target.xUm, yUm: target.yUm }
  ];
  const squaredDistance = (atom.xUm - target.xUm) ** 2 + (atom.yUm - target.yUm) ** 2;
  let cost = config.distanceWeight * squaredDistance;
  if (problem.forbiddenRegions && pathIntersectsForbiddenRegion(path, problem.forbiddenRegions)) {
    cost += config.obstacleWeight;
  }
  if (problem.staticPoints && !pathClearOfStaticAtoms(path, problem.staticPoints, config.stayToleranceUm)) {
    cost += config.staticObstacleWeight;
  }
  if (target.requiredGroup !== void 0 && target.requiredGroup !== -1 && target.requiredGroup !== atom.group) {
    cost += config.groupMismatchPenalty;
  }
  if (distance(atom, target) <= config.stayToleranceUm) cost *= 1e-3;
  cost += problem.conflictPenalties?.get(`${atom.atomId}:${target.siteId}`) ?? 0;
  return cost + atomIndex * 1e-12 + targetIndex * 1e-15;
}
function dummyDestinationCost(atom, dummyIndex, problem, config) {
  if (config.extraAtomPolicy === "KEEP" || config.extraAtomPolicy === "RELEASE_IN_PLACE") return 0 + dummyIndex * 1e-15;
  const parking = config.parkingSites[dummyIndex % Math.max(1, config.parkingSites.length)];
  if (!parking) return 0.5 + dummyIndex * 1e-15;
  const path = [{ xUm: atom.xUm, yUm: atom.yUm }, parking];
  let cost = config.distanceWeight * distance(atom, parking) ** 2;
  if (problem.forbiddenRegions && pathIntersectsForbiddenRegion(path, problem.forbiddenRegions)) {
    cost += config.obstacleWeight;
  }
  return cost + dummyIndex * 1e-15;
}
function assignmentCost(assignments) {
  return assignments.reduce((sum, assignment) => sum + assignment.assignmentCost, 0);
}
var solveAssignment = assignAtoms;

// src/trajectory.ts
var EPSILON = 1e-9;
function minimumJerk(s) {
  return smoothstep5(s);
}
function minimumJerkDerivative(s) {
  const t = clamp(s, 0, 1);
  return 30 * t * t * (t - 1) * (t - 1);
}
function minimumJerkSecondDerivative(s) {
  const t = clamp(s, 0, 1);
  return 60 * t - 180 * t * t + 120 * t * t * t;
}
function minimumJerkThirdDerivative(s) {
  const t = clamp(s, 0, 1);
  return 60 - 360 * t + 360 * t * t;
}
function parameterizeTrajectories(input, paths) {
  const commonMoveDuration = Math.max(
    input.motionConfig.framePeriodUs,
    ...paths.flatMap((path) => path.waypointsUm.slice(1).map((point, index) => {
      const previous = path.waypointsUm[index];
      const length = distance(previous, point);
      return length <= EPSILON ? input.motionConfig.framePeriodUs : minimumSegmentDuration(length, input, input.plannerConfig.planningTickUs);
    }))
  );
  const trajectories = paths.map((path) => {
    const atom = input.initialAtoms.find((candidate) => candidate.atomId === path.atomId);
    if (!atom) throw new SlmError("INTERNAL_ERROR", `Unknown atom ${path.atomId} in planned path`, { stage: "PARAMETERIZING" });
    const assignment = findAssignment(input, path);
    const initialIntensity = atom.initialTrapIntensity;
    const finalIntensity = targetIntensity(input, assignment, initialIntensity);
    const moving = path.waypointsUm.some((point, index) => index > 0 && distance(point, path.waypointsUm[index - 1]) > EPSILON);
    const waypoints = [];
    const basePreDwell = Math.max(input.motionConfig.preMoveDwellUs, input.motionConfig.minDwellBeforeMoveUs);
    const boostRamp = moving ? intensityRampDuration(Math.abs(input.motionConfig.movingTrapIntensity - initialIntensity), input, input.motionConfig.minDwellBeforeMoveUs) : 0;
    const preDwell = roundUp(basePreDwell + boostRamp, input.motionConfig.framePeriodUs);
    const scheduledStart = roundUp(Math.max(0, path.startTimeUs ?? 0), input.motionConfig.framePeriodUs);
    const ticks = path.discreteTicks.length === path.waypointsUm.length ? path.discreteTicks : path.waypointsUm.map((_, index) => index);
    let cursor = 0;
    const firstPoint = path.waypointsUm[0];
    waypoints.push({ ...firstPoint, arrivalTimeUs: 0 });
    if (scheduledStart > 0) {
      waypoints.push({ ...firstPoint, arrivalTimeUs: scheduledStart });
      cursor = scheduledStart;
    }
    if (preDwell > scheduledStart) {
      waypoints.push({ ...firstPoint, arrivalTimeUs: preDwell });
      cursor = preDwell;
    }
    for (let index = 1; index < path.waypointsUm.length; index += 1) {
      const previous = path.waypointsUm[index - 1];
      const current = path.waypointsUm[index];
      const tickDelta = Math.max(1, (ticks[index] ?? index) - (ticks[index - 1] ?? index - 1));
      const requestedDuration = tickDelta * input.plannerConfig.planningTickUs;
      const length = distance(previous, current);
      const duration = roundUp(
        Math.max(requestedDuration, commonMoveDuration * tickDelta),
        input.motionConfig.framePeriodUs
      );
      cursor += duration;
      waypoints.push({ ...current, arrivalTimeUs: cursor });
    }
    if (waypoints.length === 1) {
      cursor = Math.max(cursor, preDwell);
    }
    return {
      atomId: path.atomId,
      trapId: path.trapId,
      targetSiteId: path.goalSiteId,
      waypoints,
      startTimeUs: 0,
      endTimeUs: cursor,
      moving,
      disposition: path.disposition,
      initialIntensity,
      finalIntensity
    };
  });
  const finalTime = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  return trajectories.map((trajectory) => {
    if (trajectory.endTimeUs < finalTime) {
      const last = trajectory.waypoints.at(-1);
      trajectory.waypoints.push({ xUm: last.xUm, yUm: last.yUm, arrivalTimeUs: finalTime });
      trajectory.endTimeUs = finalTime;
    }
    return trajectory;
  });
}
function minimumSegmentDuration(length, input, requestedDuration = 0) {
  const motion = input.motionConfig;
  const speedBound = 1.875 * length / motion.maxVelocityUmPerUs;
  const accelerationBound = Math.sqrt(5.774 * length / motion.maxAccelerationUmPerUs2);
  const jerkBound = Math.cbrt(60 * length / motion.maxJerkUmPerUs3);
  let duration = Math.max(requestedDuration, speedBound, accelerationBound, jerkBound, motion.framePeriodUs);
  duration = roundUp(duration, motion.framePeriodUs);
  if (Number.isFinite(motion.maxPositionChangePerFrameUm)) {
    while (duration < 1e15 && maximumPerFrameDisplacement(length, duration, motion.framePeriodUs) > motion.maxPositionChangePerFrameUm + EPSILON) {
      duration += motion.framePeriodUs;
    }
  }
  return duration;
}
function sampleTrajectory(trajectory, timeUs) {
  const waypoints = trajectory.waypoints;
  if (waypoints.length === 0) return { xUm: 0, yUm: 0 };
  if (timeUs <= waypoints[0].arrivalTimeUs) return pointOf(waypoints[0]);
  const last = waypoints.length - 1;
  if (timeUs >= waypoints[last].arrivalTimeUs) return pointOf(waypoints[last]);
  for (let index = 0; index < last; index += 1) {
    const start = waypoints[index];
    const end = waypoints[index + 1];
    if (timeUs <= end.arrivalTimeUs) {
      const duration = end.arrivalTimeUs - start.arrivalTimeUs;
      const fraction = duration <= 0 ? 1 : (timeUs - start.arrivalTimeUs) / duration;
      const q = minimumJerk(fraction);
      return {
        xUm: start.xUm + (end.xUm - start.xUm) * q,
        yUm: start.yUm + (end.yUm - start.yUm) * q
      };
    }
  }
  return pointOf(waypoints[last]);
}
function sampleTrajectoryIntensity(trajectory, timeUs, input) {
  const initial = trajectory.initialIntensity ?? input.motionConfig.defaultTrapIntensity;
  const final = trajectory.finalIntensity ?? initial;
  if (!trajectory.moving) {
    const settleDuration2 = intensityRampDuration(Math.abs(final - initial), input, input.motionConfig.postMoveSettleUs);
    const settled2 = smoothstep5((timeUs - trajectory.endTimeUs) / settleDuration2);
    const settledIntensity = initial + (final - initial) * settled2;
    if (trajectory.disposition === "RELEASE" || trajectory.disposition === "PARK" && input.assignmentConfig.extraAtomPolicy === "PARK_AND_RELEASE") {
      const releaseStart = trajectory.endTimeUs + settleDuration2;
      const releaseDuration = intensityRampDuration(final, input);
      return settledIntensity * (1 - smoothstep5((timeUs - releaseStart) / releaseDuration));
    }
    return settledIntensity;
  }
  const firstMovingIndex = findFirstMovingSegment(trajectory);
  const firstMovingStart = firstMovingIndex < 0 ? 0 : trajectory.waypoints[firstMovingIndex].arrivalTimeUs;
  const lastMovingIndex = findLastMovingSegment(trajectory);
  const lastMovingEnd = lastMovingIndex < 0 ? trajectory.endTimeUs : trajectory.waypoints[lastMovingIndex + 1].arrivalTimeUs;
  const boost = input.motionConfig.movingTrapIntensity;
  const rampUpDuration = intensityRampDuration(Math.abs(boost - initial), input, input.motionConfig.minDwellBeforeMoveUs);
  const rampStart = firstMovingStart - rampUpDuration;
  if (timeUs < rampStart) return initial;
  const boostStart = firstMovingStart;
  const settleStart = Math.max(lastMovingEnd, boostStart);
  if (timeUs < boostStart) {
    return initial + (boost - initial) * smoothstep5((timeUs - rampStart) / rampUpDuration);
  }
  if (timeUs <= settleStart) {
    return boost;
  }
  const settleDuration = intensityRampDuration(Math.abs(final - boost), input, input.motionConfig.postMoveSettleUs);
  const settled = smoothstep5((timeUs - settleStart) / settleDuration);
  const postValue = boost + (final - boost) * settled;
  if (trajectory.disposition === "PARK" && input.assignmentConfig.extraAtomPolicy === "PARK_AND_RELEASE") {
    const releaseStart = settleStart + settleDuration;
    const releaseDuration = intensityRampDuration(final, input, input.motionConfig.postMoveSettleUs);
    return postValue * (1 - smoothstep5((timeUs - releaseStart) / releaseDuration));
  }
  return postValue;
}
function sampleTrapFrames(input, trajectories, seed = input.hologramConfig.deterministicSeed) {
  const period = input.motionConfig.framePeriodUs;
  const totalTrajectoryTime = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  const settleEnds = trajectories.map((trajectory) => {
    const initial = trajectory.initialIntensity ?? input.motionConfig.defaultTrapIntensity;
    const final = trajectory.finalIntensity ?? initial;
    const boost = input.motionConfig.movingTrapIntensity;
    const movementEnd = trajectory.moving ? lastMovingEndTime(trajectory) : trajectory.endTimeUs;
    const settleDuration = intensityRampDuration(
      Math.abs(final - (trajectory.moving ? boost : initial)),
      input,
      input.motionConfig.postMoveSettleUs
    );
    const boostStart = trajectory.moving ? firstMovingStartTime(trajectory) : movementEnd;
    const settleStart = trajectory.moving ? Math.max(movementEnd, boostStart) : movementEnd;
    const release = trajectory.disposition === "RELEASE" || trajectory.disposition === "PARK" && input.assignmentConfig.extraAtomPolicy === "PARK_AND_RELEASE" ? intensityRampDuration(final, input) : 0;
    return settleStart + settleDuration + release;
  });
  const totalTime = roundUp(Math.max(totalTrajectoryTime, ...settleEnds), period);
  const phases = trapPhaseMap(input, trajectories, seed);
  const staticStates = input.staticTraps.map((trap) => ({
    trapId: trap.trapId,
    atomId: trap.atomId < 0 ? null : trap.atomId,
    xUm: trap.xUm,
    yUm: trap.yUm,
    intensity: trap.intensity,
    targetPhaseRad: phases.get(trap.trapId),
    flags: trap.containsAtom ? 4 : 0
  }));
  const frames = [];
  for (let time = 0, frameIndex = 0; time <= totalTime; time += period, frameIndex += 1) {
    const dynamicStates = trajectories.map((trajectory) => {
      const point = sampleTrajectory(trajectory, time);
      const released = sampleTrajectoryIntensity(trajectory, time, input) <= EPSILON;
      return {
        trapId: trajectory.trapId,
        atomId: released ? null : trajectory.atomId,
        xUm: point.xUm,
        yUm: point.yUm,
        intensity: Math.max(0, sampleTrajectoryIntensity(trajectory, time, input)),
        targetPhaseRad: phases.get(trajectory.trapId),
        flags: (trajectory.moving && !released ? 1 : 0) | (released ? 2 : 0)
      };
    });
    const traps = [...staticStates, ...dynamicStates].sort((a, b) => a.trapId - b.trapId);
    frames.push({ frameIndex, timeUs: time, traps });
  }
  return frames;
}
function validateTrapFrames(frames, input, checkMotionLimits = true) {
  const errors = [];
  const warnings = [];
  let minimumSeparation = Number.POSITIVE_INFINITY;
  let maximumSpeed = 0;
  let maximumAcceleration = 0;
  let maximumJerk = 0;
  const previousByTrap = /* @__PURE__ */ new Map();
  let previousTime = -Infinity;
  for (const [frameIndex, frame] of frames.entries()) {
    if (frame.frameIndex !== frameIndex || !Number.isInteger(frame.timeUs) || frame.timeUs < 0 || frame.timeUs <= previousTime) {
      errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "Frame indices or times are not strictly increasing", frameIndex });
    }
    if (frameIndex === 0 && frame.timeUs !== 0) {
      errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "The initial trap frame must be at time zero", frameIndex });
    }
    if (frameIndex > 0 && frame.timeUs - previousTime !== input.motionConfig.framePeriodUs) {
      errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAP_FRAMES", message: "Trap frames are not sampled at the configured SLM period", frameIndex, measured: frame.timeUs - previousTime, configured: input.motionConfig.framePeriodUs });
    }
    previousTime = frame.timeUs;
    const currentIds = /* @__PURE__ */ new Set();
    const currentAtomIds = /* @__PURE__ */ new Set();
    for (const [trapId, previous] of previousByTrap) {
      if (!frame.traps.some((trap) => trap.trapId === trapId)) {
        if (previous.intensity > EPSILON) {
          errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A live trap disappeared without a zero-intensity ramp", frameIndex });
        }
        previousByTrap.delete(trapId);
      }
    }
    for (let trapIndex = 0; trapIndex < frame.traps.length; trapIndex += 1) {
      const trap = frame.traps[trapIndex];
      if (trapIndex > 0 && frame.traps[trapIndex - 1].trapId >= trap.trapId) {
        errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "Trap states are not sorted by trapId", frameIndex });
      }
      if (currentIds.has(trap.trapId)) {
        errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A trap identifier appears more than once in a frame", frameIndex });
      }
      currentIds.add(trap.trapId);
      if (trap.atomId !== null) {
        if (currentAtomIds.has(trap.atomId)) {
          errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "An atom is owned by more than one trap", frameIndex, atomIds: [trap.atomId] });
        }
        currentAtomIds.add(trap.atomId);
      }
      const staticTrap = input.staticTraps.find((candidate) => candidate.trapId === trap.trapId);
      if (staticTrap && (Math.abs(staticTrap.xUm - trap.xUm) > EPSILON || Math.abs(staticTrap.yUm - trap.yUm) > EPSILON)) {
        errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A static trap changed position", frameIndex });
      }
      if (staticTrap && staticTrap.atomId >= 0 && trap.atomId !== staticTrap.atomId) {
        errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A static trap changed atom identity", frameIndex });
      }
      const previous = previousByTrap.get(trap.trapId);
      if (!previous && frameIndex > 0 && trap.intensity > EPSILON) {
        errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A new trap must be introduced at zero intensity", frameIndex });
      }
      if (![trap.xUm, trap.yUm, trap.intensity, trap.targetPhaseRad].every(Number.isFinite)) {
        errors.push({ code: "NUMERIC_ERROR", stage: "TRAP_FRAMES", message: `Non-finite trap state at frame ${frameIndex}`, frameIndex });
      }
      if (input.calibration.manifest.fieldOfViewUm) {
        const fov = input.calibration.manifest.fieldOfViewUm;
        if (trap.xUm < fov.xMinUm || trap.xUm > fov.xMaxUm || trap.yUm < fov.yMinUm || trap.yUm > fov.yMaxUm) {
          errors.push({ code: "OUT_OF_BOUNDS", stage: "TRAP_FRAMES", message: "Trap left calibrated field of view", frameIndex });
        }
      }
      if (previous) {
        if (previous.atomId !== null && trap.atomId !== null && previous.atomId !== trap.atomId) {
          errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "An atom changed trap identity", frameIndex });
        }
        if (previous.atomId !== null && trap.atomId === null && trap.intensity > EPSILON) {
          errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A live atom was removed before its trap reached zero intensity", frameIndex, atomIds: [previous.atomId] });
        }
        if (previous.atomId === null && trap.atomId !== null && previous.intensity > EPSILON) {
          errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A trap was born without a zero-intensity ramp", frameIndex });
        }
        const delta = distance(previous, trap);
        if (Number.isFinite(input.motionConfig.maxPositionChangePerFrameUm) && delta > input.motionConfig.maxPositionChangePerFrameUm + EPSILON) {
          errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAP_FRAMES", message: "Trap moved too far in one SLM frame", frameIndex, measured: delta, configured: input.motionConfig.maxPositionChangePerFrameUm });
        }
        const intensityDelta = Math.abs(previous.intensity - trap.intensity);
        if (Number.isFinite(input.motionConfig.maxIntensityChangePerFrame) && intensityDelta > input.motionConfig.maxIntensityChangePerFrame + EPSILON) {
          errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAP_FRAMES", message: "Trap intensity changed too quickly", frameIndex, measured: intensityDelta, configured: input.motionConfig.maxIntensityChangePerFrame });
        }
      }
      previousByTrap.set(trap.trapId, trap);
    }
    for (let first = 0; first < frame.traps.length; first += 1) {
      const a = frame.traps[first];
      if (a.intensity <= EPSILON) continue;
      for (let second = first + 1; second < frame.traps.length; second += 1) {
        const b = frame.traps[second];
        if (b.intensity <= EPSILON) continue;
        const separation = distance(a, b);
        minimumSeparation = Math.min(minimumSeparation, separation);
        const atomA = input.initialAtoms.find((atom) => atom.atomId === a.atomId);
        const atomB = input.initialAtoms.find((atom) => atom.atomId === b.atomId);
        const safe = input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * ((atomA?.localizationSigmaUm ?? 0) + (atomB?.localizationSigmaUm ?? 0)) + input.plannerConfig.geometricMarginUm;
        if (separation < safe - EPSILON) {
          errors.push({ code: "COLLISION_VALIDATION_FAILED", stage: "TRAP_FRAMES", message: "Trap frame violates minimum separation", frameIndex, measured: separation, configured: safe });
        }
      }
      if (input.forbiddenRegions.some((region) => pointInForbiddenRegion(a, region))) {
        errors.push({ code: "FORBIDDEN_REGION", stage: "TRAP_FRAMES", message: "Trap frame intersects a forbidden region", frameIndex });
      }
    }
  }
  if (frames.length > 1) {
    const trajectories = trajectoriesFromFrames(frames, input);
    const interpolated = validateContinuousTrajectories(trajectories, input, checkMotionLimits);
    errors.push(...interpolated.errors);
    warnings.push(...interpolated.warnings);
    minimumSeparation = Math.min(minimumSeparation, interpolated.minimumAtomSeparationUm);
    maximumSpeed = Math.max(maximumSpeed, interpolated.maximumSpeedUmPerUs);
    maximumAcceleration = Math.max(maximumAcceleration, interpolated.maximumAccelerationUmPerUs2);
    maximumJerk = Math.max(maximumJerk, interpolated.maximumJerkUmPerUs3);
  }
  return {
    accepted: errors.length === 0,
    errors,
    warnings,
    minimumAtomSeparationUm: Number.isFinite(minimumSeparation) ? minimumSeparation : Number.POSITIVE_INFINITY,
    maximumSpeedUmPerUs: maximumSpeed,
    maximumAccelerationUmPerUs2: maximumAcceleration,
    maximumJerkUmPerUs3: maximumJerk,
    frameCount: frames.length
  };
}
function validateContinuousTrajectories(trajectories, input, checkMotionLimits = true) {
  const errors = [];
  const warnings = [];
  let minimumSeparation = Number.POSITIVE_INFINITY;
  let maximumSpeed = 0;
  let maximumAcceleration = 0;
  let maximumJerk = 0;
  const maxTime = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  const staticPoints = [
    ...input.initialAtoms.filter((atom) => !atom.movable),
    ...input.staticTraps.filter((trap) => trap.containsAtom)
  ];
  for (const trajectory of trajectories) {
    for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
      const start = trajectory.waypoints[index];
      const end = trajectory.waypoints[index + 1];
      const length = distance(start, end);
      const duration = end.arrivalTimeUs - start.arrivalTimeUs;
      if (duration <= 0) {
        errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAJECTORIES", message: "Trajectory times are not increasing", atomIds: [trajectory.atomId] });
        continue;
      }
      maximumSpeed = Math.max(maximumSpeed, length * 1.875 / duration);
      maximumAcceleration = Math.max(maximumAcceleration, length * 5.774 / (duration * duration));
      maximumJerk = Math.max(maximumJerk, length * 60 / (duration * duration * duration));
      if (checkMotionLimits && (maximumSpeed > input.motionConfig.maxVelocityUmPerUs + EPSILON || maximumAcceleration > input.motionConfig.maxAccelerationUmPerUs2 + EPSILON || maximumJerk > input.motionConfig.maxJerkUmPerUs3 + EPSILON)) {
        errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAJECTORIES", message: "Trajectory exceeds a configured motion limit", atomIds: [trajectory.atomId] });
      }
      if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([pointOf(start), pointOf(end)], [region]))) {
        errors.push({ code: "FORBIDDEN_REGION", stage: "TRAJECTORIES", message: "Trajectory segment intersects a forbidden region", atomIds: [trajectory.atomId] });
      }
    }
    const ownAtom = input.initialAtoms.find((atom) => atom.atomId === trajectory.atomId);
    const ownStaticTrap = trajectory.staticTrap ? input.staticTraps.find((trap) => trap.trapId === trajectory.trapId) : void 0;
    if (ownStaticTrap && trajectory.waypoints.some((waypoint) => Math.abs(waypoint.xUm - ownStaticTrap.xUm) > EPSILON || Math.abs(waypoint.yUm - ownStaticTrap.yUm) > EPSILON)) {
      errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAJECTORIES", message: "A static trap trajectory moved", atomIds: [trajectory.atomId] });
    }
    const ownStaticPoints = staticPoints.filter((point) => {
      if (ownAtom && point.xUm === ownAtom.xUm && point.yUm === ownAtom.yUm && !ownAtom.movable) return false;
      if (ownStaticTrap && point.xUm === ownStaticTrap.xUm && point.yUm === ownStaticTrap.yUm) return false;
      return true;
    });
    const staticClearance = input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * (ownAtom?.localizationSigmaUm ?? 0) + input.plannerConfig.geometricMarginUm;
    for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
      const start = trajectory.waypoints[index];
      const end = trajectory.waypoints[index + 1];
      const segment = { start: pointOf(start), end: pointOf(end) };
      if (ownStaticPoints.some((point) => distancePointToSegment(point, segment) < staticClearance - EPSILON)) {
        errors.push({ code: "COLLISION_VALIDATION_FAILED", stage: "TRAJECTORIES", message: "Trajectory intersects a static occupied trap", atomIds: [trajectory.atomId], configured: staticClearance });
      }
    }
  }
  const atomsById = new Map(input.initialAtoms.map((atom) => [atom.atomId, atom]));
  for (let first = 0; first < trajectories.length; first += 1) {
    for (let second = first + 1; second < trajectories.length; second += 1) {
      const a = trajectories[first];
      const b = trajectories[second];
      const atomA = atomsById.get(a.atomId);
      const atomB = atomsById.get(b.atomId);
      const safe = atomA && atomB ? input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * (atomA.localizationSigmaUm + atomB.localizationSigmaUm) + input.plannerConfig.geometricMarginUm : input.plannerConfig.minimumSeparationUm + input.plannerConfig.geometricMarginUm;
      const breakpoints = /* @__PURE__ */ new Set([0, maxTime]);
      a.waypoints.forEach((waypoint) => breakpoints.add(waypoint.arrivalTimeUs));
      b.waypoints.forEach((waypoint) => breakpoints.add(waypoint.arrivalTimeUs));
      const sorted = [...breakpoints].sort((x, y) => x - y);
      for (let interval = 0; interval + 1 < sorted.length; interval += 1) {
        const startTime = sorted[interval];
        const endTime = sorted[interval + 1];
        const pairResult = validatePairInterval(a, b, startTime, endTime, safe, input.motionConfig.maxValidationDepth);
        minimumSeparation = Math.min(minimumSeparation, pairResult.minimum);
        if (!pairResult.accepted) {
          errors.push({ code: "COLLISION_VALIDATION_FAILED", stage: "TRAJECTORIES", message: "Continuous trajectories violate safe separation or could not be proven safe", atomIds: [a.atomId, b.atomId], measured: pairResult.minimum, configured: safe });
          break;
        }
      }
    }
  }
  for (const trajectory of trajectories) {
    const points = trajectory.waypoints.map(pointOf);
    if (pathIntersectsForbiddenRegion(points, input.forbiddenRegions)) {
      errors.push({ code: "FORBIDDEN_REGION", stage: "TRAJECTORIES", message: "Trajectory enters a forbidden region", atomIds: [trajectory.atomId] });
    }
  }
  return {
    accepted: errors.length === 0,
    errors,
    warnings,
    minimumAtomSeparationUm: Number.isFinite(minimumSeparation) ? minimumSeparation : Number.POSITIVE_INFINITY,
    maximumSpeedUmPerUs: maximumSpeed,
    maximumAccelerationUmPerUs2: maximumAcceleration,
    maximumJerkUmPerUs3: maximumJerk,
    frameCount: 0
  };
}
function validatePairInterval(first, second, startTime, endTime, safe, maxDepth) {
  let minimum = Number.POSITIVE_INFINITY;
  const firstSpeed = maximumTrajectorySpeed(first);
  const secondSpeed = maximumTrajectorySpeed(second);
  const relativeSpeed = firstSpeed + secondSpeed;
  const visit = (left, right, depth) => {
    const middle = left + (right - left) / 2;
    const leftDistance = distance(sampleTrajectory(first, left), sampleTrajectory(second, left));
    const middleDistance = distance(sampleTrajectory(first, middle), sampleTrajectory(second, middle));
    const rightDistance = distance(sampleTrajectory(first, right), sampleTrajectory(second, right));
    minimum = Math.min(minimum, leftDistance, middleDistance, rightDistance);
    if (minimum < safe - EPSILON) return false;
    const duration = right - left;
    if (duration <= EPSILON || middleDistance > safe + relativeSpeed * duration / 2 + EPSILON) return true;
    if (depth >= maxDepth) return false;
    return visit(left, middle, depth + 1) && visit(middle, right, depth + 1);
  };
  return { accepted: visit(startTime, endTime, 0), minimum };
}
function maximumTrajectorySpeed(trajectory) {
  let maximum = 0;
  for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
    const start = trajectory.waypoints[index];
    const end = trajectory.waypoints[index + 1];
    const duration = end.arrivalTimeUs - start.arrivalTimeUs;
    if (duration > 0) maximum = Math.max(maximum, distance(start, end) * 1.875 / duration);
  }
  return maximum;
}
function targetPhaseForTrap(trapId, seed) {
  const hash = Number.parseInt(hashString(`${seed}:${trapId}`), 16) >>> 0;
  return hash / 4294967295 * Math.PI * 2 - Math.PI;
}
function trapPhaseMap(input, trajectories, seed) {
  const phases = /* @__PURE__ */ new Map();
  for (const trap of input.staticTraps) phases.set(trap.trapId, targetPhaseForTrap(trap.trapId, seed));
  for (const trajectory of trajectories) phases.set(trajectory.trapId, targetPhaseForTrap(trajectory.trapId, seed));
  return phases;
}
var generateTrapFrames = sampleTrapFrames;
function findAssignment(input, path) {
  if (path.goalSiteId === null) return void 0;
  const targetIndex = input.targetSites.findIndex((target) => target.siteId === path.goalSiteId);
  if (targetIndex < 0) return void 0;
  return {
    atomId: path.atomId,
    sourceIndex: input.initialAtoms.findIndex((atom) => atom.atomId === path.atomId),
    targetSiteId: path.goalSiteId,
    targetIndex,
    disposition: path.disposition,
    assignmentCost: 0
  };
}
function targetIntensity(input, assignment, initial) {
  if (!assignment || assignment.targetIndex === null) return initial;
  return input.targetSites[assignment.targetIndex]?.finalTrapIntensity ?? initial;
}
function pointOf(waypoint) {
  return { xUm: waypoint.xUm, yUm: waypoint.yUm };
}
function roundUp(value, period) {
  if (value <= 0) return 0;
  return Math.ceil(value / period - 1e-12) * period;
}
function intensityRampDuration(delta, input, minimum = 0) {
  const period = input.motionConfig.framePeriodUs;
  const maximumDelta = input.motionConfig.maxIntensityChangePerFrame;
  const requiredFrames = Number.isFinite(maximumDelta) && maximumDelta > 0 ? Math.ceil(1.875 * Math.abs(delta) / maximumDelta) : 1;
  return Math.max(period, roundUp(minimum, period), requiredFrames * period);
}
function maximumPerFrameDisplacement(length, duration, period) {
  if (length <= EPSILON) return 0;
  let maximum = 0;
  const samples = Math.max(1, Math.ceil(duration / period));
  for (let index = 0; index < samples; index += 1) {
    const a = minimumJerk(index / samples);
    const b = minimumJerk((index + 1) / samples);
    maximum = Math.max(maximum, length * (b - a));
  }
  return maximum;
}
function findFirstMovingSegment(trajectory) {
  for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
    if (distance(trajectory.waypoints[index], trajectory.waypoints[index + 1]) > EPSILON) return index;
  }
  return -1;
}
function firstMovingStartTime(trajectory) {
  const index = findFirstMovingSegment(trajectory);
  return index < 0 ? trajectory.startTimeUs : trajectory.waypoints[index].arrivalTimeUs;
}
function findLastMovingSegment(trajectory) {
  for (let index = trajectory.waypoints.length - 2; index >= 0; index -= 1) {
    if (distance(trajectory.waypoints[index], trajectory.waypoints[index + 1]) > EPSILON) return index;
  }
  return -1;
}
function lastMovingEndTime(trajectory) {
  const index = findLastMovingSegment(trajectory);
  return index < 0 ? trajectory.endTimeUs : trajectory.waypoints[index + 1].arrivalTimeUs;
}
function trajectoriesFromFrames(frames, input) {
  const first = frames[0];
  if (!first) return [];
  return first.traps.map((trap) => ({
    atomId: trap.atomId ?? -1,
    trapId: trap.trapId,
    targetSiteId: null,
    waypoints: frames.map((frame) => {
      const current = frame.traps.find((candidate) => candidate.trapId === trap.trapId) ?? trap;
      return { xUm: current.xUm, yUm: current.yUm, arrivalTimeUs: frame.timeUs };
    }),
    startTimeUs: 0,
    endTimeUs: frames.at(-1).timeUs,
    moving: true,
    staticTrap: input?.staticTraps.some((candidate) => candidate.trapId === trap.trapId) ?? false
  }));
}

// src/planning.ts
var EPSILON2 = 1e-9;
function atomTrapIds(input) {
  const used = new Set(input.staticTraps.map((trap) => trap.trapId));
  const result = /* @__PURE__ */ new Map();
  let next = Math.max(-1, ...used, ...input.initialAtoms.map((atom) => atom.atomId)) + 1;
  for (const atom of input.initialAtoms) {
    let trapId = atom.atomId;
    if (used.has(trapId)) {
      while (used.has(next)) next += 1;
      if (next > 4294967295) throw new SlmError("INVALID_ARGUMENT", "No uint32 trap identifier is available", { stage: "PLANNING" });
      trapId = next;
      next += 1;
    }
    used.add(trapId);
    result.set(atom.atomId, trapId);
  }
  return result;
}
function buildDirectPaths(input, assignments) {
  const trapIds = atomTrapIds(input);
  return assignments.map((assignment) => {
    const atom = input.initialAtoms[assignment.sourceIndex];
    if (!atom) throw new SlmError("INTERNAL_ERROR", `Missing atom at source index ${assignment.sourceIndex}`, { stage: "PLANNING" });
    const destination = destinationForAssignment(input, assignment, atom);
    const moving = distance(atom, destination) > EPSILON2;
    return {
      atomId: atom.atomId,
      trapId: trapIds.get(atom.atomId),
      goalSiteId: assignment.targetSiteId,
      disposition: assignment.disposition,
      waypointsUm: moving ? [{ xUm: atom.xUm, yUm: atom.yUm }, destination] : [{ xUm: atom.xUm, yUm: atom.yUm }],
      discreteTicks: moving ? [0, 1] : [0]
    };
  });
}
function planMotion(input, assignments) {
  const directPaths = buildDirectPaths(input, assignments);
  const movingPathIndices = directPaths.map((path, index) => ({ path, index })).filter(({ index }) => {
    const atom = input.initialAtoms[assignments[index]?.sourceIndex ?? -1];
    return atom?.movable && (assignments[index]?.disposition === "MOVE_TO_TARGET" || assignments[index]?.disposition === "PARK");
  }).map(({ index }) => index);
  const conflicts = findPathConflicts(directPaths, assignments, input);
  const staticConflicts = directPaths.map((path, index) => ({ path, index })).filter(({ path }) => pathTouchesStaticGeometry(path, input));
  for (const conflict of staticConflicts) {
    const assignment = assignments[conflict.index];
    const atom = assignment ? input.initialAtoms[assignment.sourceIndex] : void 0;
    if (atom?.movable && (assignment?.disposition === "MOVE_TO_TARGET" || assignment?.disposition === "PARK")) {
      if (!movingPathIndices.includes(conflict.index)) movingPathIndices.push(conflict.index);
    } else {
      throw new SlmError("PATH_NOT_FOUND", "A fixed trap path intersects static geometry", {
        stage: "PLANNING",
        details: { atomId: conflict.path.atomId }
      });
    }
  }
  movingPathIndices.sort((a, b) => a - b);
  let paths = directPaths;
  let conflictComponentCount = connectedConflictComponents(conflicts).length + (staticConflicts.length > 0 ? 1 : 0);
  let directPathCount = directPaths.filter((path) => path.waypointsUm.length <= 2).length;
  if (conflicts.length > 0 || staticConflicts.length > 0) {
    if (movingPathIndices.length === 0) {
      throw new SlmError("PATH_NOT_FOUND", "A fixed trap path conflicts with static geometry", { stage: "PLANNING" });
    }
    const planned = planWithCbs(input, directPaths, movingPathIndices);
    if (!planned) {
      const fallback = serializedDetourPlan(input, directPaths, movingPathIndices);
      if (!fallback) {
        throw new SlmError("PATH_NOT_FOUND", "No collision-free path plan was found", {
          stage: "PLANNING",
          retryable: true,
          details: { conflicts }
        });
      }
      paths = fallback;
    } else {
      paths = planned;
    }
    directPathCount = paths.filter((path) => path.waypointsUm.length <= 2 && path.discreteTicks.length <= 2).length;
  } else {
    directPathCount = directPaths.filter((path) => path.waypointsUm.length <= 2).length;
  }
  const trajectories = parameterizeTrajectories(input, paths);
  const validation = validateContinuousTrajectories(trajectories, input);
  if (!validation.accepted) {
    throw new SlmError("COLLISION_VALIDATION_FAILED", "Continuous trajectory validation failed", {
      stage: "PLANNING",
      retryable: true,
      details: { validation }
    });
  }
  const waitCount = paths.reduce((count, path) => count + countRepeatedWaypoints(path.waypointsUm), 0);
  const detourCount = paths.filter((path) => path.waypointsUm.length > 2).length;
  const makespanUs = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  const minimumValidatedSeparationUm = validation.minimumAtomSeparationUm;
  return {
    assignment: assignments,
    plannedPaths: paths,
    trajectories,
    directPathCount,
    conflictComponentCount,
    waitCount,
    detourCount,
    makespanUs,
    minimumValidatedSeparationUm
  };
}
function findPathConflicts(paths, assignments, input) {
  const conflicts = [];
  const maxTicks = Math.max(1, ...paths.map((path) => path.discreteTicks.at(-1) ?? 0));
  for (let first = 0; first < paths.length; first += 1) {
    for (let second = first + 1; second < paths.length; second += 1) {
      const atomA = input.initialAtoms[assignments[first]?.sourceIndex ?? -1];
      const atomB = input.initialAtoms[assignments[second]?.sourceIndex ?? -1];
      if (!atomA || !atomB) continue;
      const safe = pairSafeSeparation(atomA, atomB, input);
      for (let tick = 0; tick < maxTicks; tick += 1) {
        const a0 = plannedPointAt(paths[first], tick);
        const a1 = plannedPointAt(paths[first], tick + 1);
        const b0 = plannedPointAt(paths[second], tick);
        const b1 = plannedPointAt(paths[second], tick + 1);
        if (distance(a0, b0) < safe - EPSILON2) {
          conflicts.push({ first, second, time: tick, kind: "vertex" });
          break;
        }
        if (distance(a0, b1) < safe - EPSILON2 && distance(a1, b0) < safe - EPSILON2) {
          conflicts.push({ first, second, time: tick, kind: "edge" });
          break;
        }
        if (segmentDistance({ start: a0, end: a1 }, { start: b0, end: b1 }) < safe - EPSILON2) {
          conflicts.push({ first, second, time: tick, kind: "near-edge" });
          break;
        }
      }
    }
  }
  return conflicts;
}
function connectedConflictComponents(conflicts) {
  const vertices = /* @__PURE__ */ new Set();
  conflicts.forEach((conflict) => {
    vertices.add(conflict.first);
    vertices.add(conflict.second);
  });
  const components = [];
  const remaining = new Set(vertices);
  while (remaining.size > 0) {
    const start = [...remaining].sort((a, b) => a - b)[0];
    const component = [start];
    remaining.delete(start);
    for (let index = 0; index < component.length; index += 1) {
      const current = component[index];
      for (const conflict of conflicts) {
        const neighbor = conflict.first === current ? conflict.second : conflict.second === current ? conflict.first : -1;
        if (neighbor >= 0 && remaining.has(neighbor)) {
          remaining.delete(neighbor);
          component.push(neighbor);
        }
      }
    }
    components.push(component.sort((a, b) => a - b));
  }
  return components;
}
function planWithCbs(input, directPaths, activeIndices) {
  if (activeIndices.length === 0) return directPaths;
  const graph = buildPlanningGraph(input, directPaths, activeIndices);
  const starts = [];
  const goals = [];
  for (const pathIndex of activeIndices) {
    const path = directPaths[pathIndex];
    const start = exactGraphNode(graph.nodes, path.waypointsUm[0]);
    const goal = exactGraphNode(graph.nodes, path.waypointsUm.at(-1));
    if (start < 0 || goal < 0) return void 0;
    starts.push(start);
    goals.push(goal);
  }
  const initialPaths = [];
  for (let agent = 0; agent < activeIndices.length; agent += 1) {
    const path = lowLevelAStar(graph, starts[agent], goals[agent], [], input.plannerConfig.maxSearchTicks, input.plannerConfig.maxAStarExpansions);
    if (!path) return void 0;
    initialPaths.push(path);
  }
  const priorityPlan = prioritizedSpaceTimePlan(
    graph,
    starts,
    goals,
    activeIndices,
    directPaths,
    input
  );
  if (priorityPlan && !findGridConflict(priorityPlan, activeIndices, directPaths, input, graph)) {
    return replaceGridPaths(directPaths, activeIndices, priorityPlan, graph);
  }
  const open = [{ constraints: [], paths: initialPaths, cost: cbsCost(initialPaths) }];
  let expanded = 0;
  while (open.length > 0 && expanded < input.plannerConfig.maxCbsNodes) {
    open.sort((a, b) => a.cost - b.cost || a.constraints.length - b.constraints.length);
    const current = open.shift();
    expanded += 1;
    const conflict = findGridConflict(current.paths, activeIndices, directPaths, input, graph);
    if (!conflict) {
      const result = [...directPaths];
      return replaceGridPaths(result, activeIndices, current.paths, graph);
    }
    const branches = conflict.kind === "vertex" ? [
      { agent: conflict.first, constraint: { agent: conflict.first, time: conflict.time, node: conflict.nodeA } },
      { agent: conflict.second, constraint: { agent: conflict.second, time: conflict.time, node: conflict.nodeB } }
    ] : [
      { agent: conflict.first, constraint: { agent: conflict.first, time: conflict.time, from: conflict.fromA, to: conflict.toA } },
      { agent: conflict.second, constraint: { agent: conflict.second, time: conflict.time, from: conflict.fromB, to: conflict.toB } }
    ];
    for (const branch of branches) {
      const constraints = [...current.constraints, branch.constraint];
      const paths = current.paths.slice();
      const replanned = lowLevelAStar(
        graph,
        starts[branch.agent],
        goals[branch.agent],
        constraints.filter((constraint) => constraint.agent === branch.agent),
        input.plannerConfig.maxSearchTicks,
        input.plannerConfig.maxAStarExpansions
      );
      if (replanned) {
        paths[branch.agent] = replanned;
        open.push({ constraints, paths, cost: cbsCost(paths) });
      }
    }
  }
  return void 0;
}
function replaceGridPaths(directPaths, activeIndices, gridPaths, graph) {
  const result = [...directPaths];
  for (let agent = 0; agent < activeIndices.length; agent += 1) {
    const pathIndex = activeIndices[agent];
    const gridPath = gridPaths[agent];
    result[pathIndex] = {
      ...directPaths[pathIndex],
      waypointsUm: gridPath.nodes.map((node) => ({ ...graph.nodes[node].point })),
      discreteTicks: [...gridPath.ticks]
    };
  }
  return result;
}
function prioritizedSpaceTimePlan(graph, starts, goals, activeIndices, directPaths, input) {
  const orders = [];
  const baseline = Array.from({ length: starts.length }, (_, index) => index);
  orders.push(baseline);
  orders.push([...baseline].sort((a, b) => {
    const lengthA = directPaths[activeIndices[a]].waypointsUm;
    const lengthB = directPaths[activeIndices[b]].waypointsUm;
    return pathLength(lengthB) - pathLength(lengthA) || a - b;
  }));
  orders.push([...baseline].reverse());
  const uniqueOrders = orders.filter((order, index) => orders.findIndex((candidate) => candidate.every((value, position) => value === order[position])) === index);
  for (const order of uniqueOrders.slice(0, input.plannerConfig.maxPriorityRetries)) {
    const paths = Array(starts.length);
    let success = true;
    for (const agent of order) {
      const constraints = [];
      for (let other = 0; other < starts.length; other += 1) {
        const reserved = paths[other];
        if (!reserved) continue;
        for (let time = 0; time <= input.plannerConfig.maxSearchTicks; time += 1) {
          const node = gridNodeAt(reserved, time);
          constraints.push({ agent, time, node });
          const next = gridNodeAt(reserved, time + 1);
          if (next !== node) constraints.push({ agent, time, from: next, to: node });
        }
      }
      const path = lowLevelAStar(graph, starts[agent], goals[agent], constraints, input.plannerConfig.maxSearchTicks, input.plannerConfig.maxAStarExpansions);
      if (!path) {
        success = false;
        break;
      }
      paths[agent] = path;
    }
    if (success && paths.every((path) => path !== void 0)) return paths;
  }
  return void 0;
}
function findGridConflict(paths, activeIndices, directPaths, input, graph) {
  const maxTicks = Math.max(1, ...paths.map((path) => path.ticks.at(-1) ?? 0));
  for (let first = 0; first < paths.length; first += 1) {
    for (let second = first + 1; second < paths.length; second += 1) {
      const pathAtomA = directPaths[activeIndices[first]]?.atomId;
      const pathAtomB = directPaths[activeIndices[second]]?.atomId;
      const atomA = input.initialAtoms.find((atom) => atom.atomId === pathAtomA);
      const atomB = input.initialAtoms.find((atom) => atom.atomId === pathAtomB);
      const safe = atomA && atomB ? pairSafeSeparation(atomA, atomB, input) : input.plannerConfig.minimumSeparationUm;
      for (let time = 0; time < maxTicks; time += 1) {
        const nodeA = gridNodeAt(paths[first], time);
        const nodeB = gridNodeAt(paths[second], time);
        if (nodeA === nodeB) {
          return { first, second, time, kind: "vertex", nodeA, nodeB };
        }
        const fromA = nodeA;
        const toA = gridNodeAt(paths[first], time + 1);
        const fromB = nodeB;
        const toB = gridNodeAt(paths[second], time + 1);
        if (toA === fromB && toB === fromA) {
          return { first, second, time, kind: "edge", fromA, toA, fromB, toB };
        }
        if (segmentDistance(
          { start: graph.nodes[fromA].point, end: graph.nodes[toA].point },
          { start: graph.nodes[fromB].point, end: graph.nodes[toB].point }
        ) < safe - EPSILON2) {
          return { first, second, time, kind: "edge", fromA, toA, fromB, toB };
        }
      }
    }
  }
  return void 0;
}
function lowLevelAStar(graph, start, goal, constraints, maxTicks, maxExpansions) {
  const open = [{ node: start, time: 0, g: 0, f: 0, key: stateKey(start, 0) }];
  const best = /* @__PURE__ */ new Map([[stateKey(start, 0), 0]]);
  const previous = /* @__PURE__ */ new Map();
  let expansions = 0;
  while (open.length > 0 && expansions < maxExpansions) {
    open.sort((a, b) => a.f - b.f || a.time - b.time || a.node - b.node);
    const current = open.shift();
    if (current.node === goal) return reconstructGridPath(current.key, current.node, current.time, previous);
    if (current.time >= maxTicks) continue;
    expansions += 1;
    const neighbors = [...graph.nodes[current.node].neighbors, current.node].sort((a, b) => a - b);
    for (const next of neighbors) {
      const nextTime = current.time + 1;
      if (hasVertexConstraint(constraints, next, nextTime) || hasEdgeConstraint(constraints, current.node, next, current.time)) continue;
      const key = stateKey(next, nextTime);
      const g = current.g + 1;
      if (g >= (best.get(key) ?? Number.POSITIVE_INFINITY)) continue;
      best.set(key, g);
      previous.set(key, { key: current.key, node: current.node, time: current.time });
      const goalPoint = graph.nodes[goal].point;
      const nextPoint = graph.nodes[next].point;
      const heuristic = distance(nextPoint, goalPoint);
      open.push({ node: next, time: nextTime, g, f: g + heuristic, key });
    }
  }
  return void 0;
}
function stateKey(node, time) {
  return `${node}@${time}`;
}
function reconstructGridPath(key, node, time, previous) {
  const nodes = [node];
  const ticks = [time];
  let current = key;
  while (previous.has(current)) {
    const prior = previous.get(current);
    nodes.push(prior.node);
    ticks.push(prior.time);
    current = prior.key;
  }
  nodes.reverse();
  ticks.reverse();
  return { nodes, ticks };
}
function buildPlanningGraph(input, paths, activeIndices) {
  const anchors = activeIndices.flatMap((index) => {
    const path = paths[index];
    return [path.waypointsUm[0], path.waypointsUm.at(-1)];
  });
  const staticPoints = [
    ...input.initialAtoms.filter((atom) => !atom.movable),
    ...input.staticTraps.filter((trap) => trap.containsAtom)
  ];
  const activeSet = new Set(activeIndices);
  for (let index = 0; index < paths.length; index += 1) {
    if (!activeSet.has(index)) staticPoints.push(paths[index].waypointsUm.at(-1));
  }
  const allPoints = [...anchors, ...staticPoints, ...input.assignmentConfig.parkingSites];
  const maximumSigma = Math.max(0, ...input.initialAtoms.map((atom) => atom.localizationSigmaUm));
  const safeResolution = input.plannerConfig.minimumSeparationUm + 2 * input.plannerConfig.kSigma * maximumSigma + input.plannerConfig.geometricMarginUm;
  let resolution = Math.max(input.plannerConfig.gridResolutionUm, safeResolution * 1.25);
  const bounds = boundingBox(allPoints, Math.max(input.plannerConfig.minimumSeparationUm * 3, resolution * 2));
  const width = Math.max(1, Math.ceil((bounds.xMax - bounds.xMin) / resolution));
  const height = Math.max(1, Math.ceil((bounds.yMax - bounds.yMin) / resolution));
  const maxAxis = 128;
  if (width > maxAxis) resolution = (bounds.xMax - bounds.xMin) / maxAxis;
  if (height > maxAxis) resolution = Math.max(resolution, (bounds.yMax - bounds.yMin) / maxAxis);
  const xValues = [];
  const yValues = [];
  const xCount = Math.ceil((bounds.xMax - bounds.xMin) / resolution);
  const yCount = Math.ceil((bounds.yMax - bounds.yMin) / resolution);
  for (let index = 0; index <= xCount; index += 1) xValues.push(Math.min(bounds.xMax, bounds.xMin + index * resolution));
  for (let index = 0; index <= yCount; index += 1) yValues.push(Math.min(bounds.yMax, bounds.yMin + index * resolution));
  for (const anchor of anchors) {
    xValues.push(anchor.xUm);
    yValues.push(anchor.yUm);
  }
  const unique = (values) => [...new Set(values.map((value) => Number(value.toFixed(9))))].sort((a, b) => a - b);
  const xs = unique(xValues);
  const ys = unique(yValues);
  const nodes = [];
  const nodeKey = /* @__PURE__ */ new Map();
  const safeStatic = input.plannerConfig.minimumSeparationUm + 2 * input.plannerConfig.kSigma * maximumSigma + input.plannerConfig.geometricMarginUm;
  for (const x of xs) {
    for (const y of ys) {
      const point = { xUm: x, yUm: y };
      if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([point], [region]))) continue;
      if (!pointClearOfStaticAtoms(point, staticPoints, safeStatic)) continue;
      const key = coordinateKey(point);
      nodeKey.set(key, nodes.length);
      nodes.push({ point, neighbors: [] });
    }
  }
  for (const anchor of anchors) {
    const key = coordinateKey(anchor);
    if (nodeKey.has(key)) continue;
    if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([anchor], [region]))) continue;
    if (!pointClearOfStaticAtoms(anchor, staticPoints, safeStatic)) continue;
    nodeKey.set(key, nodes.length);
    nodes.push({ point: { ...anchor }, neighbors: [] });
  }
  const bucketSize = Math.max(resolution * 1.6, 1e-6);
  const buckets = /* @__PURE__ */ new Map();
  nodes.forEach((node, index) => {
    const key = bucketKey(node.point, bucketSize);
    const bucket = buckets.get(key) ?? [];
    bucket.push(index);
    buckets.set(key, bucket);
  });
  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes[index];
    const bx = Math.floor(node.point.xUm / bucketSize);
    const by = Math.floor(node.point.yUm / bucketSize);
    const candidates = /* @__PURE__ */ new Set();
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (const candidate of buckets.get(`${bx + dx},${by + dy}`) ?? []) candidates.add(candidate);
      }
    }
    for (const candidate of candidates) {
      if (candidate <= index) continue;
      const other = nodes[candidate];
      const edgeLength = distance(node.point, other.point);
      if (edgeLength > resolution * Math.SQRT2 * 1.25 + 1e-7) continue;
      const segment = { start: node.point, end: other.point };
      if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([segment.start, segment.end], [region]))) continue;
      if (!pointClearOfStaticAtoms(segment.start, staticPoints, safeStatic) || !pointClearOfStaticAtoms(segment.end, staticPoints, safeStatic)) continue;
      if (staticPoints.some((point) => distancePointToSegment(point, segment) < safeStatic - EPSILON2)) continue;
      node.neighbors.push(candidate);
      other.neighbors.push(index);
    }
  }
  nodes.forEach((node) => node.neighbors.sort((a, b) => a - b));
  return { nodes };
}
function serializedDetourPlan(input, directPaths, activeIndices) {
  const result = directPaths.map((path) => ({ ...path, waypointsUm: path.waypointsUm.map((point) => ({ ...point })), discreteTicks: [...path.discreteTicks] }));
  const staticPoints = [
    ...input.initialAtoms.filter((atom) => !atom.movable),
    ...input.staticTraps.filter((trap) => trap.containsAtom)
  ];
  const activeSet = new Set(activeIndices);
  for (let index = 0; index < result.length; index += 1) {
    if (!activeSet.has(index)) staticPoints.push(result[index].waypointsUm.at(-1));
  }
  const active = activeIndices.slice().sort((a, b) => a - b);
  let tick = 0;
  const occupied = [...staticPoints];
  for (const index of active) {
    const path = result[index];
    const start = path.waypointsUm[0];
    const end = path.waypointsUm.at(-1);
    if (path.waypointsUm.length === 1) {
      tick += 1;
      occupied.push(end);
      continue;
    }
    const candidates = detourCandidates(start, end, input, occupied);
    let selected;
    for (const candidate of candidates) {
      const candidatePath = [start, candidate, end];
      if (!pathIntersectsForbiddenRegion(candidatePath, input.forbiddenRegions) && pathClearOfPoints(candidatePath, occupied, input.plannerConfig.minimumSeparationUm + input.plannerConfig.geometricMarginUm)) {
        selected = candidatePath;
        break;
      }
    }
    if (!selected) {
      if (!pathIntersectsForbiddenRegion([start, end], input.forbiddenRegions) && pathClearOfPoints([start, end], occupied, input.plannerConfig.minimumSeparationUm + input.plannerConfig.geometricMarginUm)) {
        selected = [start, end];
      } else {
        return void 0;
      }
    }
    const waitTicks = tick;
    const movementTicks = Math.max(1, selected.length - 1);
    path.waypointsUm = selected;
    path.discreteTicks = selected.map((_, pointIndex) => waitTicks + pointIndex);
    if (waitTicks > 0) {
      path.waypointsUm = [...Array.from({ length: waitTicks + 1 }, () => ({ ...start })), ...selected.slice(1)];
      path.discreteTicks = path.waypointsUm.map((_, pointIndex) => pointIndex);
    }
    tick += movementTicks + 1;
    occupied.push(end);
  }
  return result;
}
function detourCandidates(start, end, input, occupied) {
  const points = [...occupied, start, end, ...input.assignmentConfig.parkingSites];
  const bounds = boundingBox(points, input.plannerConfig.minimumSeparationUm * 4 + input.plannerConfig.geometricMarginUm);
  const offsets = [
    { xUm: bounds.xMin, yUm: bounds.yMin },
    { xUm: bounds.xMin, yUm: bounds.yMax },
    { xUm: bounds.xMax, yUm: bounds.yMin },
    { xUm: bounds.xMax, yUm: bounds.yMax },
    { xUm: (bounds.xMin + bounds.xMax) / 2, yUm: bounds.yMin },
    { xUm: (bounds.xMin + bounds.xMax) / 2, yUm: bounds.yMax },
    { xUm: bounds.xMin, yUm: (bounds.yMin + bounds.yMax) / 2 },
    { xUm: bounds.xMax, yUm: (bounds.yMin + bounds.yMax) / 2 },
    { xUm: (start.xUm + end.xUm) / 2, yUm: (start.yUm + end.yUm) / 2 + input.plannerConfig.minimumSeparationUm * 2 },
    { xUm: (start.xUm + end.xUm) / 2, yUm: (start.yUm + end.yUm) / 2 - input.plannerConfig.minimumSeparationUm * 2 }
  ];
  return offsets.sort((a, b) => distance(start, a) + distance(a, end) - distance(start, b) - distance(b, end));
}
function destinationForAssignment(input, assignment, atom) {
  if (assignment.targetIndex !== null && input.targetSites[assignment.targetIndex]) {
    const target = input.targetSites[assignment.targetIndex];
    return { xUm: target.xUm, yUm: target.yUm };
  }
  if (assignment.disposition === "PARK") {
    const parking = input.assignmentConfig.parkingSites[assignment.parkingSiteIndex ?? assignment.sourceIndex % Math.max(1, input.assignmentConfig.parkingSites.length)];
    if (parking) return { ...parking };
  }
  return { xUm: atom.xUm, yUm: atom.yUm };
}
function pairSafeSeparation(a, b, input) {
  return input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * ((a.localizationSigmaUm ?? 0) + (b.localizationSigmaUm ?? 0)) + input.plannerConfig.geometricMarginUm;
}
function pathTouchesStaticGeometry(path, input) {
  const atom = input.initialAtoms.find((candidate) => candidate.atomId === path.atomId);
  const sigma = atom?.localizationSigmaUm ?? 0;
  const clearance = input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * sigma + input.plannerConfig.geometricMarginUm;
  const staticPoints = [
    ...input.initialAtoms.filter((candidate) => !candidate.movable && candidate.atomId !== path.atomId),
    ...input.staticTraps.filter((trap) => trap.containsAtom && trap.atomId !== path.atomId)
  ];
  if (pathClearOfPoints(path.waypointsUm, staticPoints, clearance) === false) return true;
  return pathIntersectsForbiddenRegion(path.waypointsUm, input.forbiddenRegions);
}
function plannedPointAt(path, tick) {
  if (path.waypointsUm.length === 1) return path.waypointsUm[0];
  const ticks = path.discreteTicks;
  if (tick <= ticks[0]) return path.waypointsUm[0];
  const last = ticks.length - 1;
  if (tick >= ticks[last]) return path.waypointsUm[last];
  for (let index = 0; index < last; index += 1) {
    const startTick = ticks[index];
    const endTick = ticks[index + 1];
    if (tick >= startTick && tick <= endTick) {
      const fraction = endTick === startTick ? 1 : (tick - startTick) / (endTick - startTick);
      return {
        xUm: path.waypointsUm[index].xUm + (path.waypointsUm[index + 1].xUm - path.waypointsUm[index].xUm) * fraction,
        yUm: path.waypointsUm[index].yUm + (path.waypointsUm[index + 1].yUm - path.waypointsUm[index].yUm) * fraction
      };
    }
  }
  return path.waypointsUm[last];
}
function gridNodeAt(path, time) {
  if (time <= path.ticks[0]) return path.nodes[0];
  const last = path.ticks.length - 1;
  if (time >= path.ticks[last]) return path.nodes[last];
  for (let index = 0; index < last; index += 1) {
    if (time < path.ticks[index + 1]) return path.nodes[index];
  }
  return path.nodes[last];
}
function hasVertexConstraint(constraints, node, time) {
  return constraints.some((constraint) => constraint.time === time && constraint.node === node);
}
function hasEdgeConstraint(constraints, from, to, time) {
  return constraints.some(
    (constraint) => constraint.time === time && constraint.from === from && constraint.to === to
  );
}
function cbsCost(paths) {
  return paths.reduce((sum, path) => sum + (path.ticks.at(-1) ?? 0), 0);
}
function exactGraphNode(nodes, point) {
  return nodes.findIndex((node) => distance(node.point, point) <= 1e-8);
}
function coordinateKey(point) {
  return `${point.xUm.toFixed(9)},${point.yUm.toFixed(9)}`;
}
function bucketKey(point, size) {
  return `${Math.floor(point.xUm / size)},${Math.floor(point.yUm / size)}`;
}
function pathClearOfPoints(path, points, clearance) {
  return points.every((point) => {
    if (path.some((pathPoint) => distance(pathPoint, point) < clearance - EPSILON2)) return false;
    for (let index = 0; index + 1 < path.length; index += 1) {
      if (distancePointToSegment(point, { start: path[index], end: path[index + 1] }) < clearance - EPSILON2) return false;
    }
    return true;
  });
}
function countRepeatedWaypoints(points) {
  let count = 0;
  for (let index = 1; index < points.length; index += 1) {
    if (distance(points[index - 1], points[index]) <= EPSILON2) count += 1;
  }
  return count;
}
function pathLength(points) {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) length += distance(points[index - 1], points[index]);
  return length;
}
var planPaths = planMotion;

// src/fft.ts
function createComplexField(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new SlmError("INVALID_ARGUMENT", "Complex field dimensions must be positive integers", { stage: "SOLVING_SLM_FRAMES" });
  }
  return { real: new Float64Array(width * height), imag: new Float64Array(width * height), width, height };
}
function fft1d(real, imag, inverse = false) {
  if (real.length !== imag.length) throw new SlmError("INVALID_ARGUMENT", "FFT real and imaginary lengths differ", { stage: "SOLVING_SLM_FRAMES" });
  if (real.length === 0) return;
  if ((real.length & real.length - 1) !== 0) {
    dft1d(real, imag, inverse);
    return;
  }
  bitReversePermutation(real, imag);
  for (let size = 2; size <= real.length; size <<= 1) {
    const half = size >>> 1;
    const sign = inverse ? 1 : -1;
    const angle = sign * (2 * Math.PI / size);
    const rootReal = Math.cos(angle);
    const rootImag = Math.sin(angle);
    for (let offset = 0; offset < real.length; offset += size) {
      let twiddleReal = 1;
      let twiddleImag = 0;
      for (let index = 0; index < half; index += 1) {
        const even = offset + index;
        const odd = even + half;
        const productReal = real[odd] * twiddleReal - imag[odd] * twiddleImag;
        const productImag = real[odd] * twiddleImag + imag[odd] * twiddleReal;
        const evenReal = real[even];
        const evenImag = imag[even];
        real[even] = evenReal + productReal;
        imag[even] = evenImag + productImag;
        real[odd] = evenReal - productReal;
        imag[odd] = evenImag - productImag;
        const nextTwiddleReal = twiddleReal * rootReal - twiddleImag * rootImag;
        twiddleImag = twiddleReal * rootImag + twiddleImag * rootReal;
        twiddleReal = nextTwiddleReal;
      }
    }
  }
  if (inverse) {
    const scale = 1 / real.length;
    for (let index = 0; index < real.length; index += 1) {
      real[index] = real[index] * scale;
      imag[index] = imag[index] * scale;
    }
  }
}
function fft2d(field, inverse = false) {
  const { width, height, real, imag } = field;
  const rowReal = new Float64Array(width);
  const rowImag = new Float64Array(width);
  for (let y = 0; y < height; y += 1) {
    const rowOffset = y * width;
    for (let x = 0; x < width; x += 1) {
      rowReal[x] = real[rowOffset + x];
      rowImag[x] = imag[rowOffset + x];
    }
    fft1d(rowReal, rowImag, inverse);
    for (let x = 0; x < width; x += 1) {
      real[rowOffset + x] = rowReal[x];
      imag[rowOffset + x] = rowImag[x];
    }
  }
  const columnReal = new Float64Array(height);
  const columnImag = new Float64Array(height);
  for (let x = 0; x < width; x += 1) {
    for (let y = 0; y < height; y += 1) {
      columnReal[y] = real[y * width + x];
      columnImag[y] = imag[y * width + x];
    }
    fft1d(columnReal, columnImag, inverse);
    for (let y = 0; y < height; y += 1) {
      real[y * width + x] = columnReal[y];
      imag[y * width + x] = columnImag[y];
    }
  }
}
function cloneComplexField(field) {
  return { real: new Float64Array(field.real), imag: new Float64Array(field.imag), width: field.width, height: field.height };
}
function sampleComplex(field, x, y, bilinear = true) {
  if (!bilinear) {
    const ix = clampIndex(Math.round(x), field.width);
    const iy = clampIndex(Math.round(y), field.height);
    const index = iy * field.width + ix;
    return { real: field.real[index], imag: field.imag[index] };
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const tx = x - x0;
  const ty = y - y0;
  const p00 = fieldValue(field, x0, y0);
  const p10 = fieldValue(field, x1, y0);
  const p01 = fieldValue(field, x0, y1);
  const p11 = fieldValue(field, x1, y1);
  return {
    real: lerp(lerp(p00.real, p10.real, tx), lerp(p01.real, p11.real, tx), ty),
    imag: lerp(lerp(p00.imag, p10.imag, tx), lerp(p01.imag, p11.imag, tx), ty)
  };
}
function scatterComplex(field, x, y, real, imag, bilinear = false) {
  if (!bilinear) {
    const ix = clampIndex(Math.round(x), field.width);
    const iy = clampIndex(Math.round(y), field.height);
    const index = iy * field.width + ix;
    field.real[index] = real;
    field.imag[index] = imag;
    return;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  for (const [xOffset, xWeight] of [[0, 1 - tx], [1, tx]]) {
    for (const [yOffset, yWeight] of [[0, 1 - ty], [1, ty]]) {
      const ix = clampIndex(x0 + xOffset, field.width);
      const iy = clampIndex(y0 + yOffset, field.height);
      const index = iy * field.width + ix;
      const weight = xWeight * yWeight;
      field.real[index] = field.real[index] + real * weight;
      field.imag[index] = field.imag[index] + imag * weight;
    }
  }
}
function scatterComplexAdjoint(field, x, y, real, imag, bilinear = true) {
  if (!bilinear) {
    scatterComplex(field, x, y, real, imag, false);
    return;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = x - x0;
  const ty = y - y0;
  const entries = /* @__PURE__ */ new Map();
  for (const [xOffset, xWeight] of [[0, 1 - tx], [1, tx]]) {
    for (const [yOffset, yWeight] of [[0, 1 - ty], [1, ty]]) {
      const ix = clampIndex(x0 + xOffset, field.width);
      const iy = clampIndex(y0 + yOffset, field.height);
      const index = iy * field.width + ix;
      entries.set(index, (entries.get(index) ?? 0) + xWeight * yWeight);
    }
  }
  const norm = [...entries.values()].reduce((sum, weight) => sum + weight * weight, 0);
  if (norm <= 0) return;
  for (const [index, weight] of entries) {
    field.real[index] = field.real[index] + real * weight / norm;
    field.imag[index] = field.imag[index] + imag * weight / norm;
  }
}
function fieldPower(field) {
  const power = new Float64Array(field.real.length);
  for (let index = 0; index < power.length; index += 1) power[index] = field.real[index] ** 2 + field.imag[index] ** 2;
  return power;
}
function bitReversePermutation(real, imag) {
  let reversed = 0;
  for (let index = 1; index < real.length; index += 1) {
    let bit = real.length >>> 1;
    while (reversed & bit) {
      reversed ^= bit;
      bit >>>= 1;
    }
    reversed ^= bit;
    if (index < reversed) {
      [real[index], real[reversed]] = [real[reversed], real[index]];
      [imag[index], imag[reversed]] = [imag[reversed], imag[index]];
    }
  }
}
function dft1d(real, imag, inverse) {
  const inputReal = new Float64Array(real);
  const inputImag = new Float64Array(imag);
  const sign = inverse ? 1 : -1;
  const size = real.length;
  for (let output = 0; output < size; output += 1) {
    let sumReal = 0;
    let sumImag = 0;
    for (let input = 0; input < size; input += 1) {
      const angle = sign * 2 * Math.PI * output * input / size;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      sumReal += inputReal[input] * cosine - inputImag[input] * sine;
      sumImag += inputReal[input] * sine + inputImag[input] * cosine;
    }
    const scale = inverse ? 1 / size : 1;
    real[output] = sumReal * scale;
    imag[output] = sumImag * scale;
  }
}
function fieldValue(field, x, y) {
  const ix = clampIndex(x, field.width);
  const iy = clampIndex(y, field.height);
  const index = iy * field.width + ix;
  return { real: field.real[index], imag: field.imag[index] };
}
function clampIndex(value, size) {
  return Math.max(0, Math.min(size - 1, value));
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}

// src/hologram.ts
var SequentialWgsSolver = class {
  width;
  height;
  calibration;
  config;
  accepted;
  candidate;
  constructor(calibration, config = {}) {
    this.calibration = calibration;
    this.config = {
      width: config.width ?? calibration.manifest.fftWidth ?? calibration.manifest.activeWidth,
      height: config.height ?? calibration.manifest.fftHeight ?? calibration.manifest.activeHeight,
      format: config.format ?? "UINT8",
      targetPhaseMode: config.targetPhaseMode ?? "PHASE_LOCKED_WGS",
      firstFrameIterations: config.firstFrameIterations ?? 12,
      subsequentFrameIterations: config.subsequentFrameIterations ?? 4,
      maxIterations: config.maxIterations ?? 64,
      gamma: config.gamma ?? 0.7,
      epsilon: config.epsilon ?? 1e-8,
      minWeight: config.minWeight ?? 0.1,
      maxWeight: config.maxWeight ?? 10,
      convergenceTolerance: config.convergenceTolerance ?? 1e-4,
      backgroundPolicy: config.backgroundPolicy ?? "PRESERVE",
      oversampling: config.oversampling ?? 1,
      qualityGates: config.qualityGates ?? {},
      maxInsertedFrames: config.maxInsertedFrames ?? 32,
      deterministicSeed: config.deterministicSeed ?? 1,
      measureSolveTime: config.measureSolveTime ?? false,
      requireConvergence: config.requireConvergence ?? false
    };
    this.width = this.config.width;
    this.height = this.config.height;
    this.beginSequence();
  }
  beginSequence() {
    this.accepted = void 0;
    this.candidate = void 0;
  }
  solveSequentialFrame(frame, iterationBudget) {
    if (frame.traps.some((trap) => !Number.isFinite(trap.xUm) || !Number.isFinite(trap.yUm) || !Number.isFinite(trap.intensity) || !Number.isFinite(trap.targetPhaseRad))) {
      throw new SlmError("NUMERIC_ERROR", "Trap frame contains a non-finite value", { stage: "SOLVING_SLM_FRAMES" });
    }
    const started = this.config.measureSolveTime ? nowMs() : 0;
    const pixels = this.width * this.height;
    const mappedTargets = frame.traps.map((trap) => this.mapCoordinate(trap));
    for (const target of mappedTargets) {
      if (!Number.isFinite(target.x) || !Number.isFinite(target.y) || target.x < -0.5 || target.x > this.width - 0.5 || target.y < -0.5 || target.y > this.height - 0.5) {
        throw new SlmError("OUT_OF_BOUNDS", "A calibrated trap coordinate lies outside the FFT grid", {
          stage: "SOLVING_SLM_FRAMES",
          details: { target, width: this.width, height: this.height }
        });
      }
    }
    const previous = this.accepted;
    const targetPhases = /* @__PURE__ */ new Map();
    const weights = /* @__PURE__ */ new Map();
    for (const trap of frame.traps) {
      targetPhases.set(
        trap.trapId,
        this.config.targetPhaseMode === "PHASE_INTERPOLATED_WGS" ? trap.targetPhaseRad : previous?.targetPhases.get(trap.trapId) ?? trap.targetPhaseRad
      );
      weights.set(trap.trapId, previous?.weights.get(trap.trapId) ?? 1);
    }
    let phase = previous ? new Float64Array(previous.phase) : this.initializeSuperposition(frame);
    const iterations = Math.min(
      Math.max(1, iterationBudget ?? (previous ? this.config.subsequentFrameIterations : this.config.firstFrameIterations)),
      this.config.maxIterations
    );
    let measured = frame.traps.map(() => ({ real: 0, imag: 0 }));
    let amplitudes = frame.traps.map(() => 0);
    let converged = frame.traps.length === 0;
    let maxRelativeError = Number.POSITIVE_INFINITY;
    let performedIterations = 0;
    for (let iteration = 0; iteration < iterations; iteration += 1) {
      performedIterations += 1;
      const forward = this.forwardField(phase);
      measured = mappedTargets.map((target) => sampleComplex(forward, target.x, target.y, true));
      amplitudes = measured.map((value) => Math.hypot(value.real, value.imag));
      const desired = frame.traps.map((trap) => Math.sqrt(Math.max(0, trap.intensity)));
      let numerator = 0;
      let denominator = 0;
      for (let index = 0; index < desired.length; index += 1) {
        numerator += desired[index] * amplitudes[index];
        denominator += desired[index] * desired[index];
      }
      const scale = denominator > this.config.epsilon ? numerator / denominator : 1;
      for (let index = 0; index < frame.traps.length; index += 1) {
        const trap = frame.traps[index];
        if (desired[index] <= this.config.epsilon) continue;
        const oldWeight = weights.get(trap.trapId) ?? 1;
        const ratio = scale * desired[index] / (amplitudes[index] + this.config.epsilon);
        weights.set(trap.trapId, clamp(oldWeight * ratio ** this.config.gamma, this.config.minWeight, this.config.maxWeight));
      }
      normalizeWeights(weights);
      const constrained = this.config.backgroundPolicy === "PRESERVE" ? cloneField(forward) : createComplexField(this.width, this.height);
      for (const target of mappedTargets) clearComplexSupport(constrained, target.x, target.y);
      for (let index = 0; index < frame.traps.length; index += 1) {
        const trap = frame.traps[index];
        const target = mappedTargets[index];
        const targetPhase = chooseTargetPhase(this.config.targetPhaseMode, targetPhases.get(trap.trapId) ?? trap.targetPhaseRad, measured[index]);
        const targetAmplitude = (weights.get(trap.trapId) ?? 1) * desired[index];
        scatterComplexAdjoint(constrained, target.x, target.y, targetAmplitude * Math.cos(targetPhase), targetAmplitude * Math.sin(targetPhase), true);
      }
      fft2d(constrained, true);
      for (let index = 0; index < phase.length; index += 1) {
        const magnitude = Math.hypot(constrained.real[index], constrained.imag[index]);
        if (magnitude > this.config.epsilon) phase[index] = Math.atan2(constrained.imag[index], constrained.real[index]);
      }
      maxRelativeError = maximumRelativeAmplitudeError(amplitudes, desired, scale, this.config.epsilon);
      const phaseError = maximumTargetPhaseError(frame.traps, measured, targetPhases, this.config.targetPhaseMode);
      converged = maxRelativeError <= this.config.convergenceTolerance && phaseError <= phaseConvergenceTolerance(this.config);
      if (converged) break;
    }
    const displayPhase = this.composeDisplayPhase(phase);
    const codes = this.quantize(displayPhase);
    const optimizedForward = this.forwardField(phase);
    const optimizedMeasured = frame.traps.map((trap) => {
      const mapped = this.mapCoordinate(trap);
      return sampleComplex(optimizedForward, mapped.x, mapped.y, true);
    });
    const optimizedAmplitudes = optimizedMeasured.map((value) => Math.hypot(value.real, value.imag));
    const optimizedDesired = frame.traps.map((trap) => Math.sqrt(Math.max(0, trap.intensity)));
    maxRelativeError = maximumRelativeAmplitudeError(
      optimizedAmplitudes,
      optimizedDesired,
      fitAmplitudeScale(optimizedAmplitudes, optimizedDesired, this.config.epsilon),
      this.config.epsilon
    );
    converged = maxRelativeError <= this.config.convergenceTolerance && maximumTargetPhaseError(frame.traps, optimizedMeasured, targetPhases, this.config.targetPhaseMode) <= phaseConvergenceTolerance(this.config);
    const finalForward = this.forwardField(this.decodeCodes(codes, displayPhase));
    const finalMeasured = frame.traps.map((trap) => {
      const mapped = this.mapCoordinate(trap);
      return sampleComplex(finalForward, mapped.x, mapped.y, true);
    });
    const finalAmplitudes = finalMeasured.map((value) => Math.hypot(value.real, value.imag));
    const measuredPhases = /* @__PURE__ */ new Map();
    const measuredIntensities = /* @__PURE__ */ new Map();
    frame.traps.forEach((trap, index) => {
      measuredPhases.set(trap.trapId, Math.atan2(finalMeasured[index].imag, finalMeasured[index].real));
      measuredIntensities.set(trap.trapId, finalAmplitudes[index] ** 2);
    });
    const finalDesired = frame.traps.map((trap) => Math.sqrt(Math.max(0, trap.intensity)));
    const finalScale = fitAmplitudeScale(finalAmplitudes, finalDesired, this.config.epsilon);
    maxRelativeError = maximumRelativeAmplitudeError(finalAmplitudes, finalDesired, finalScale, this.config.epsilon);
    converged = maxRelativeError <= this.config.convergenceTolerance && maximumTargetPhaseError(frame.traps, finalMeasured, targetPhases, this.config.targetPhaseMode) <= phaseConvergenceTolerance(this.config);
    const metrics = this.evaluateMetrics(
      frame,
      finalForward,
      finalMeasured,
      finalAmplitudes,
      codes,
      previous,
      performedIterations,
      converged,
      maxRelativeError,
      started,
      targetPhases,
      measuredPhases,
      measuredIntensities,
      weights
    );
    this.candidate = {
      phase,
      weights: new Map(weights),
      targetPhases: new Map(targetPhases),
      measuredPhases,
      measuredIntensities,
      codes,
      frameIndex: frame.frameIndex
    };
    metrics.accepted = passesQualityGates(metrics, this.config.qualityGates) && (!this.config.requireConvergence || metrics.converged);
    return { pixels: new codes.constructor(codes), metrics };
  }
  commitFrameState() {
    if (!this.candidate) throw new SlmError("INVALID_ARGUMENT", "No hologram candidate is available to commit", { stage: "SOLVING_SLM_FRAMES" });
    this.accepted = {
      phase: new Float64Array(this.candidate.phase),
      weights: new Map(this.candidate.weights),
      targetPhases: new Map(this.candidate.targetPhases),
      measuredPhases: new Map(this.candidate.measuredPhases),
      measuredIntensities: new Map(this.candidate.measuredIntensities),
      codes: new this.candidate.codes.constructor(this.candidate.codes),
      frameIndex: this.candidate.frameIndex
    };
    this.candidate = void 0;
  }
  rollbackToPreviousAcceptedFrame() {
    this.candidate = void 0;
  }
  get acceptedFrameIndex() {
    return this.accepted?.frameIndex;
  }
  initializeSuperposition(frame) {
    const phase = new Float64Array(this.width * this.height);
    if (frame.traps.length === 0) return phase;
    for (let y = 0; y < this.height; y += 1) {
      for (let x = 0; x < this.width; x += 1) {
        let real = 0;
        let imag = 0;
        for (const trap of frame.traps) {
          const mapped = this.mapCoordinate(trap);
          const amplitude = Math.sqrt(Math.max(0, trap.intensity));
          const angle = TAU * (mapped.x * x / this.width + mapped.y * y / this.height) + trap.targetPhaseRad;
          real += amplitude * Math.cos(angle);
          imag += amplitude * Math.sin(angle);
        }
        phase[y * this.width + x] = Math.atan2(imag, real);
      }
    }
    return phase;
  }
  forwardField(phase) {
    const field = createComplexField(this.width, this.height);
    const activePixels = this.calibration.manifest.activeWidth * this.calibration.manifest.activeHeight;
    for (let index = 0; index < phase.length; index += 1) {
      const x = index % this.width;
      const y = Math.floor(index / this.width);
      const amplitude = this.calibrationValueAt(this.calibration.incidentAmplitude, index, x, y, 1, activePixels) * this.calibrationValueAt(this.calibration.apertureMask, index, x, y, 1, activePixels);
      field.real[index] = amplitude * Math.cos(phase[index]);
      field.imag[index] = amplitude * Math.sin(phase[index]);
    }
    fft2d(field, false);
    return field;
  }
  calibrationValueAt(value, index, x, y, fallback, activePixels) {
    if (value === void 0 || typeof value === "number") return typeof value === "number" ? value : fallback;
    if (value.length === this.width * this.height) return value[index] ?? fallback;
    if (value.length === activePixels) {
      const xStart = Math.floor((this.width - this.calibration.manifest.activeWidth) / 2);
      const yStart = Math.floor((this.height - this.calibration.manifest.activeHeight) / 2);
      if (x < xStart || y < yStart || x >= xStart + this.calibration.manifest.activeWidth || y >= yStart + this.calibration.manifest.activeHeight) {
        return 0;
      }
      const activeIndex = (y - yStart) * this.calibration.manifest.activeWidth + (x - xStart);
      return value[activeIndex] ?? fallback;
    }
    return fallback;
  }
  mapCoordinate(trap) {
    const transform = this.calibration.coordinateTransform;
    if (transform?.physicalToFft) {
      const mapped = transform.physicalToFft({ xUm: trap.xUm, yUm: trap.yUm });
      return "x" in mapped ? mapped : { x: mapped.xUm, y: mapped.yUm };
    }
    if (transform && [transform.a, transform.b, transform.c, transform.d].every((value) => value !== void 0)) {
      return {
        x: transform.a * trap.xUm + transform.b * trap.yUm + (transform.offsetX ?? 0),
        y: transform.c * trap.xUm + transform.d * trap.yUm + (transform.offsetY ?? 0)
      };
    }
    const originX = transform?.originXUm ?? this.width / 2;
    const originY = transform?.originYUm ?? this.height / 2;
    const scaleX = transform?.scaleX ?? 1;
    const scaleY = transform?.scaleY ?? 1;
    const rotation = transform?.rotationRad ?? 0;
    const x = trap.xUm * scaleX;
    const y = trap.yUm * scaleY;
    return {
      x: originX + Math.cos(rotation) * x - Math.sin(rotation) * y,
      y: originY - Math.sin(rotation) * x - Math.cos(rotation) * y
    };
  }
  composeDisplayPhase(phase) {
    const display = new Float64Array(phase.length);
    const signs = this.calibration.phaseSigns;
    for (let index = 0; index < phase.length; index += 1) {
      display[index] = wrapPhase(
        phase[index] + (signs?.aberration ?? 1) * calibrationValue(this.calibration.aberrationPhase, index, 0) + (signs?.grating ?? 1) * calibrationValue(this.calibration.carrierGrating, index, 0) + (signs?.lens ?? 1) * calibrationValue(this.calibration.digitalLens, index, 0)
      );
    }
    return display;
  }
  quantize(phase) {
    const maxCode = this.config.format === "UINT8" ? 255 : 65535;
    const result = this.config.format === "UINT8" ? new Uint8Array(phase.length) : new Uint16Array(phase.length);
    for (let index = 0; index < phase.length; index += 1) {
      const target = wrapPhase(phase[index]);
      let code = inverseLut(target, this.calibration, maxCode);
      code = Math.round(clamp(code, 0, maxCode));
      result[index] = code;
    }
    return result;
  }
  decodeCodes(codes, fallback) {
    const decoded = new Float64Array(codes.length);
    const maxCode = this.config.format === "UINT8" ? 255 : 65535;
    const lut = this.calibration.phaseResponseLut;
    for (let index = 0; index < codes.length; index += 1) {
      if (!lut || lut.length < 2) {
        const inverse = this.calibration.inversePhaseLut;
        if (inverse && inverse.length > 1) {
          let nearest = 0;
          let nearestError = Number.POSITIVE_INFINITY;
          for (let lutIndex = 0; lutIndex < inverse.length; lutIndex += 1) {
            const error = Math.abs(inverse[lutIndex] - codes[index]);
            if (error < nearestError) {
              nearestError = error;
              nearest = lutIndex;
            }
          }
          decoded[index] = wrapPhase(nearest / (inverse.length - 1) * TAU - Math.PI);
        } else {
          decoded[index] = wrapPhase(codes[index] / maxCode * TAU - Math.PI);
        }
        continue;
      }
      const position = codes[index] / maxCode * (lut.length - 1);
      const low = Math.floor(position);
      const high = Math.min(lut.length - 1, low + 1);
      const fraction = position - low;
      decoded[index] = wrapPhase((lut[low] ?? fallback[index]) * (1 - fraction) + (lut[high] ?? fallback[index]) * fraction);
    }
    return decoded;
  }
  evaluateMetrics(frame, field, measured, amplitudes, codes, previous, iterations, converged, relativeError, started, targetPhases, measuredPhases, measuredIntensities, weights) {
    const intensities = amplitudes.map((amplitude) => amplitude * amplitude);
    const mean = meanOf(intensities);
    const std = standardDeviation(intensities, mean);
    const totalPower = field.real.reduce((sum, value, index) => sum + value * value + field.imag[index] * field.imag[index], 0);
    const targetPower = intensities.reduce((sum, value) => sum + value, 0);
    const ghost = maximumGhostIntensity(field, frame.traps.map((trap) => this.mapCoordinate(trap)));
    const phaseError = maximumTargetPhaseError(frame.traps, measured, targetPhases, this.config.targetPhaseMode);
    const phaseChange = previous ? maximumMapPhaseChange(previous.measuredPhases, measuredPhases) : 0;
    const codeChange = previous ? maximumCodeChange(previous.codes, codes) : 0;
    const flags = [];
    if (!converged) flags.push("NOT_CONVERGED");
    if (frame.traps.some((trap) => trap.intensity > this.config.epsilon) && (mean <= this.config.epsilon || targetPower <= this.config.epsilon)) flags.push("ZERO_TARGET_OUTPUT");
    if (!Number.isFinite(relativeError) || !Number.isFinite(totalPower) || ![mean, std, phaseError, phaseChange, codeChange, ...weights.values()].every(Number.isFinite)) flags.push("NUMERIC_ERROR");
    const transitionMinimum = estimateTransitionMinimumIntensity(measuredIntensities, frame, previous, codes, this.config.format);
    return {
      frameIndex: frame.frameIndex,
      timeUs: frame.timeUs,
      iterations,
      converged,
      targetIntensityMean: mean,
      targetIntensityStd: std,
      targetIntensityCoefficientOfVariation: mean > 0 ? std / mean : 0,
      minimumToMeanIntensityRatio: mean > 0 && intensities.length > 0 ? Math.min(...intensities) / mean : 1,
      diffractionEfficiency: totalPower > 0 ? targetPower / totalPower : 0,
      maximumGhostIntensity: ghost,
      maximumWgsWeight: Math.max(0, ...weights.values()),
      maximumTargetPhaseErrorRad: phaseError,
      targetPhaseChangeRad: phaseChange,
      displayCodeChange: codeChange,
      estimatedTransitionMinimumIntensity: transitionMinimum,
      solveTimeMs: this.config.measureSolveTime ? nowMs() - started : 0,
      refinementCount: 0,
      numericalValid: !flags.includes("NUMERIC_ERROR"),
      accepted: false,
      flags
    };
  }
};
var WgsSolver = SequentialWgsSolver;
function solveHologramFrame(frame, calibration, config = {}) {
  const solver = new SequentialWgsSolver(calibration, config);
  const result = solver.solveSequentialFrame(frame);
  if (result.metrics.accepted) solver.commitFrameState();
  return result;
}
function chooseTargetPhase(mode, persistent, measured) {
  const measuredPhase = Math.atan2(measured.imag, measured.real);
  if (mode === "REFERENCE_WGS") return measuredPhase;
  if (mode === "SOFT_PHASE_LOCKED_WGS") return wrapPhase(persistent + 0.2 * wrapPhase(measuredPhase - persistent));
  return persistent;
}
function maximumTargetPhaseError(traps, measured, targetPhases, mode) {
  if (traps.length === 0 || mode === "REFERENCE_WGS") return 0;
  return Math.max(...traps.map((trap, index) => angularDistance(Math.atan2(measured[index].imag, measured[index].real), targetPhases.get(trap.trapId) ?? trap.targetPhaseRad)));
}
function maximumRelativeAmplitudeError(amplitudes, desired, scale, epsilon) {
  if (desired.length === 0) return 0;
  return Math.max(...desired.map((value, index) => Math.abs(amplitudes[index] - scale * value) / (Math.abs(scale * value) + epsilon)));
}
function phaseConvergenceTolerance(config) {
  const codeCount = config.format === "UINT8" ? 256 : 65536;
  return Math.max(config.convergenceTolerance, 1e-3, TAU / codeCount * 1.5);
}
function fitAmplitudeScale(amplitudes, desired, epsilon) {
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < desired.length; index += 1) {
    numerator += desired[index] * amplitudes[index];
    denominator += desired[index] * desired[index];
  }
  return denominator > epsilon ? numerator / denominator : 1;
}
function normalizeWeights(weights) {
  if (weights.size === 0) return;
  const mean = [...weights.values()].reduce((sum, value) => sum + value, 0) / weights.size;
  if (mean <= 0 || !Number.isFinite(mean)) return;
  for (const [id, value] of weights) weights.set(id, value / mean);
}
function passesQualityGates(metrics, gates) {
  if (!metrics.numericalValid) return false;
  if (metrics.flags.includes("ZERO_TARGET_OUTPUT")) return false;
  if (gates?.maxIntensityCoefficientOfVariation !== void 0 && metrics.targetIntensityCoefficientOfVariation > gates.maxIntensityCoefficientOfVariation) return false;
  if (gates?.minIntensityToMeanRatio !== void 0 && metrics.minimumToMeanIntensityRatio < gates.minIntensityToMeanRatio) return false;
  if (gates?.minDiffractionEfficiency !== void 0 && metrics.diffractionEfficiency < gates.minDiffractionEfficiency) return false;
  if (gates?.maxGhostIntensity !== void 0 && metrics.maximumGhostIntensity > gates.maxGhostIntensity) return false;
  if (gates?.maxTargetPhaseErrorRad !== void 0 && metrics.maximumTargetPhaseErrorRad > gates.maxTargetPhaseErrorRad) return false;
  if (gates?.maxPhaseChangeRad !== void 0 && metrics.targetPhaseChangeRad > gates.maxPhaseChangeRad) return false;
  if (gates?.maxDisplayCodeChange !== void 0 && metrics.displayCodeChange > gates.maxDisplayCodeChange) return false;
  return true;
}
function calibrationValue(value, index, fallback) {
  if (value === void 0) return fallback;
  if (typeof value === "number") return value;
  return value[index] ?? fallback;
}
function clearComplexSupport(field, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  for (const ix of [x0, x0 + 1]) {
    for (const iy of [y0, y0 + 1]) {
      const clampedX = Math.max(0, Math.min(field.width - 1, ix));
      const clampedY = Math.max(0, Math.min(field.height - 1, iy));
      const index = clampedY * field.width + clampedX;
      field.real[index] = 0;
      field.imag[index] = 0;
    }
  }
}
function inverseLut(phase, calibration, maxCode) {
  if (calibration.inversePhaseLut && calibration.inversePhaseLut.length > 1) {
    const position = (phase + Math.PI) / TAU * (calibration.inversePhaseLut.length - 1);
    const low = Math.floor(position);
    const high = Math.min(calibration.inversePhaseLut.length - 1, low + 1);
    const fraction = position - low;
    return (calibration.inversePhaseLut[low] ?? 0) * (1 - fraction) + (calibration.inversePhaseLut[high] ?? 0) * fraction;
  }
  if (calibration.phaseResponseLut && calibration.phaseResponseLut.length > 1) {
    const lut = calibration.phaseResponseLut;
    const phaseRange = calibration.manifest.phaseConvention === "ZERO_TO_TWO_PI" || calibration.manifest.phaseConvention === void 0 && lut[0] >= -1e-9 && lut[lut.length - 1] > Math.PI ? phase < 0 ? phase + TAU : phase : phase;
    const increasing = lut[0] <= lut[lut.length - 1];
    let low = 0;
    let high = lut.length - 1;
    while (high - low > 1) {
      const middle = low + high >>> 1;
      const before = lut[middle];
      if (increasing ? before < phaseRange : before > phaseRange) low = middle;
      else high = middle;
    }
    const bestCode = Math.abs(lut[low] - phaseRange) <= Math.abs(lut[high] - phaseRange) ? low : high;
    return bestCode / Math.max(1, lut.length - 1) * maxCode;
  }
  return (phase + Math.PI) / TAU * maxCode;
}
function cloneField(field) {
  return { real: new Float64Array(field.real), imag: new Float64Array(field.imag), width: field.width, height: field.height };
}
function maximumGhostIntensity(field, targets) {
  if (targets.length === 0) return 0;
  let maximum = 0;
  for (let y = 0; y < field.height; y += 1) {
    for (let x = 0; x < field.width; x += 1) {
      if (targets.some((target) => Math.hypot(target.x - x, target.y - y) <= 1.5)) continue;
      const index = y * field.width + x;
      maximum = Math.max(maximum, field.real[index] ** 2 + field.imag[index] ** 2);
    }
  }
  return maximum;
}
function maximumMapPhaseChange(previous, current) {
  let maximum = 0;
  for (const [id, phase] of current) {
    const old = previous.get(id);
    if (old !== void 0) maximum = Math.max(maximum, angularDistance(old, phase));
  }
  return maximum;
}
function estimateTransitionMinimumIntensity(current, frame, previous, codes, format) {
  const values = frame.traps.map((trap) => current.get(trap.trapId) ?? 0);
  const currentMean = meanOf(values);
  const currentMinimum = currentMean > 0 && values.length > 0 ? Math.min(...values) / currentMean : values.length === 0 ? 1 : 0;
  if (!previous) return clamp(currentMinimum, 0, 1);
  const previousValues = [...previous.measuredIntensities.values()];
  const previousMean = meanOf(previousValues);
  const previousMinimum = previousMean > 0 && previousValues.length > 0 ? Math.min(...previousValues) / previousMean : 1;
  const maxCode = format === "UINT8" ? 255 : 65535;
  const codeFactor = 1 - clamp(maximumCodeChange(previous.codes, codes) / Math.max(1, maxCode), 0, 1);
  return clamp(Math.min(currentMinimum, previousMinimum, currentMinimum * codeFactor), 0, 1);
}
function maximumCodeChange(previous, current) {
  let maximum = 0;
  const length = Math.min(previous.length, current.length);
  for (let index = 0; index < length; index += 1) maximum = Math.max(maximum, Math.abs(previous[index] - current[index]));
  return maximum;
}
function meanOf(values) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}
function standardDeviation(values, mean) {
  return values.length === 0 ? 0 : Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}
function nowMs() {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}
function frameDescriptor(frame, pixels, width, height, format, byteOffset) {
  const bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  return {
    frameIndex: frame.frameIndex,
    timeUs: frame.timeUs,
    width,
    height,
    format,
    byteOffset,
    byteLength: pixels.byteLength,
    crc32: crc32(bytes)
  };
}

// src/frame-store.ts
var MemoryFrameStore = class {
  frames = [];
  get length() {
    return this.frames.length;
  }
  get count() {
    return this.frames.length;
  }
  append(frame) {
    this.frames.push(frame);
  }
  get(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.frames.length) {
      throw new RangeError(`Frame index ${index} is out of range`);
    }
    return this.frames[index];
  }
  toArray() {
    return [...this.frames];
  }
  clear() {
    this.frames.length = 0;
  }
  [Symbol.iterator]() {
    return this.frames[Symbol.iterator]();
  }
  async *[Symbol.asyncIterator]() {
    for (const frame of this.frames) yield frame;
  }
};
var MappedFrameStore = class {
  values;
  writer;
  constructor(writer, initial = []) {
    this.values = [...initial];
    this.writer = writer;
  }
  get length() {
    return this.values.length;
  }
  get count() {
    return this.values.length;
  }
  async append(frame) {
    await this.writer(frame, this.values.length);
    this.values.push(frame);
  }
  get(index) {
    if (index < 0 || index >= this.values.length) throw new RangeError(`Frame index ${index} is out of range`);
    return this.values[index];
  }
  toArray() {
    return [...this.values];
  }
  clear() {
    this.values.length = 0;
  }
  [Symbol.iterator]() {
    return this.values[Symbol.iterator]();
  }
  async *[Symbol.asyncIterator]() {
    for (const frame of this.values) yield frame;
  }
};

// src/storage.ts
function createSequencePackage(sequence) {
  const slmFrames = syncValue(sequence.slmFrameStore.toArray());
  const slmBytes = concatenateSlmFrames(slmFrames);
  const trapFrames = syncValue(sequence.trapFrameStore.toArray());
  const trapBytes = encodeTrapFrames(trapFrames);
  const descriptors = sequence.slmFrameDescriptors;
  const trapIndex = trapFrames.map((frame, index) => ({
    frameIndex: frame.frameIndex,
    timeUs: frame.timeUs,
    byteOffset: trapFrameOffset(trapFrames, index),
    byteLength: trapFrameByteLength(frame)
  }));
  return {
    "manifest.json": stableStringify(sequence.manifest),
    "assignment.json": stableStringify(sequence.assignment),
    "trajectories.json": stableStringify(sequence.trajectories),
    "trap-frames.bin": trapBytes,
    "trap-frames.index.json": stableStringify(trapIndex),
    "slm-frames.bin": slmBytes,
    "slm-frames.index.json": stableStringify(descriptors.map(descriptorForJson)),
    "frame-metrics.jsonl": sequence.frameMetrics.map((metric) => stableStringify(metric)).join("\n") + (sequence.frameMetrics.length ? "\n" : ""),
    "validation.json": stableStringify(sequence.validation),
    "calibration-manifest.json": stableStringify({ calibrationId: sequence.manifest.calibrationId, calibrationHash: sequence.manifest.calibrationHash })
  };
}
async function createSequencePackageAsync(sequence) {
  const [slmFrames, trapFrames] = await Promise.all([
    sequence.slmFrameStore.toArray(),
    sequence.trapFrameStore.toArray()
  ]);
  const slmBytes = concatenateSlmFrames(slmFrames);
  const trapBytes = encodeTrapFrames(trapFrames);
  const trapIndex = trapFrames.map((frame, index) => ({
    frameIndex: frame.frameIndex,
    timeUs: frame.timeUs,
    byteOffset: trapFrameOffset(trapFrames, index),
    byteLength: trapFrameByteLength(frame)
  }));
  return {
    "manifest.json": stableStringify(sequence.manifest),
    "assignment.json": stableStringify(sequence.assignment),
    "trajectories.json": stableStringify(sequence.trajectories),
    "trap-frames.bin": trapBytes,
    "trap-frames.index.json": stableStringify(trapIndex),
    "slm-frames.bin": slmBytes,
    "slm-frames.index.json": stableStringify(sequence.slmFrameDescriptors.map(descriptorForJson)),
    "frame-metrics.jsonl": sequence.frameMetrics.map((metric) => stableStringify(metric)).join("\n") + (sequence.frameMetrics.length ? "\n" : ""),
    "validation.json": stableStringify(sequence.validation),
    "calibration-manifest.json": stableStringify({ calibrationId: sequence.manifest.calibrationId, calibrationHash: sequence.manifest.calibrationHash })
  };
}
function encodeTrapFrames(frames) {
  const bytesPerState = 4 + 4 + 1 + 8 + 8 + 8 + 8 + 4;
  const bytesPerFrame = 4 + 8 + 4;
  const total = frames.reduce((sum, frame) => sum + bytesPerFrame + frame.traps.length * bytesPerState, 0);
  const output = new Uint8Array(total);
  const view = new DataView(output.buffer);
  let offset = 0;
  for (const frame of frames) {
    if (!Number.isInteger(frame.frameIndex) || frame.frameIndex < 0 || frame.frameIndex > 4294967295 || !Number.isInteger(frame.timeUs) || frame.timeUs < 0) {
      throw new SlmError("STORAGE_ERROR", "Trap frame index and time must fit the binary format", { stage: "WRITING_OUTPUT" });
    }
    view.setUint32(offset, frame.frameIndex, true);
    offset += 4;
    view.setBigInt64(offset, BigInt(Math.trunc(frame.timeUs)), true);
    offset += 8;
    view.setUint32(offset, frame.traps.length, true);
    offset += 4;
    for (const trap of frame.traps) {
      if (!Number.isInteger(trap.trapId) || trap.trapId < 0 || trap.trapId > 4294967295 || trap.atomId !== null && (!Number.isInteger(trap.atomId) || trap.atomId < 0 || trap.atomId > 4294967295)) {
        throw new SlmError("STORAGE_ERROR", "Trap identifiers must fit uint32", { stage: "WRITING_OUTPUT" });
      }
      view.setUint32(offset, trap.trapId, true);
      offset += 4;
      view.setUint32(offset, trap.atomId === null || trap.atomId === void 0 ? 0 : trap.atomId >>> 0, true);
      offset += 4;
      view.setUint8(offset, trap.atomId === null || trap.atomId === void 0 ? 0 : 1);
      offset += 1;
      view.setFloat64(offset, trap.xUm, true);
      offset += 8;
      view.setFloat64(offset, trap.yUm, true);
      offset += 8;
      view.setFloat64(offset, trap.intensity, true);
      offset += 8;
      view.setFloat64(offset, trap.targetPhaseRad, true);
      offset += 8;
      view.setUint32(offset, trap.flags, true);
      offset += 4;
    }
  }
  return output;
}
function decodeTrapFrames(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frames = [];
  let offset = 0;
  const stateBytes = 4 + 4 + 1 + 8 + 8 + 8 + 8 + 4;
  while (offset < bytes.byteLength) {
    if (offset + 16 > bytes.byteLength) throw new SlmError("STORAGE_ERROR", "Truncated trap-frame header", { stage: "VALIDATING" });
    const frameIndex = view.getUint32(offset, true);
    offset += 4;
    const timeUs = Number(view.getBigInt64(offset, true));
    offset += 8;
    const count = view.getUint32(offset, true);
    offset += 4;
    if (count > 16777216 || offset + count * stateBytes > bytes.byteLength) {
      throw new SlmError("STORAGE_ERROR", "Truncated trap-frame state data", { stage: "VALIDATING" });
    }
    const traps = [];
    for (let index = 0; index < count; index += 1) {
      const trapId = view.getUint32(offset, true);
      offset += 4;
      const atomValue = view.getUint32(offset, true);
      offset += 4;
      const atomPresent = view.getUint8(offset) !== 0;
      offset += 1;
      const xUm = view.getFloat64(offset, true);
      offset += 8;
      const yUm = view.getFloat64(offset, true);
      offset += 8;
      const intensity = view.getFloat64(offset, true);
      offset += 8;
      const targetPhaseRad = view.getFloat64(offset, true);
      offset += 8;
      const flags = view.getUint32(offset, true);
      offset += 4;
      traps.push({ trapId, atomId: atomPresent ? atomValue : null, xUm, yUm, intensity, targetPhaseRad, flags });
    }
    frames.push({ frameIndex, timeUs, traps });
  }
  return frames;
}
function concatenateSlmFrames(frames) {
  const byteLength = frames.reduce((sum, frame) => sum + frame.byteLength, 0);
  const output = new Uint8Array(byteLength);
  let offset = 0;
  for (const frame of frames) {
    output.set(new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength), offset);
    offset += frame.byteLength;
  }
  return output;
}
function verifySequencePackage(sequence, descriptors = sequence.slmFrameDescriptors) {
  const errors = [];
  const normalizedDescriptors = descriptors.map(normalizeDescriptor);
  const stored = sequence.slmFrameStore.toArray();
  if (stored instanceof Promise) throw new TypeError("This frame store is asynchronous; use verifySequencePackageAsync");
  const frames = stored;
  if (frames.length !== normalizedDescriptors.length) errors.push("SLM frame count does not match descriptor count");
  for (let index = 0; index < Math.min(frames.length, normalizedDescriptors.length); index += 1) {
    const descriptor = normalizedDescriptors[index];
    const bytes = new Uint8Array(frames[index].buffer, frames[index].byteOffset, frames[index].byteLength);
    const expectedOffset = normalizedDescriptors.slice(0, index).reduce((sum, item) => sum + item.byteLength, 0);
    if (descriptor.byteOffset !== BigInt(expectedOffset)) errors.push(`Frame ${index} byte offset mismatch`);
    if (descriptor.byteLength !== bytes.byteLength) errors.push(`Frame ${index} byte length mismatch`);
    if (descriptor.crc32 !== crc32(bytes)) errors.push(`Frame ${index} CRC32 mismatch`);
  }
  const manifestChecksum = sequence.manifest.checksums.frameDescriptors;
  if (manifestChecksum && manifestChecksum !== hashString(stableStringify(normalizedDescriptors))) errors.push("Manifest descriptor checksum mismatch");
  const slmChecksum = sequence.manifest.checksums.slmFrames;
  if (slmChecksum && slmChecksum !== hashString(stableStringify(normalizedDescriptors.map((descriptor) => descriptor.crc32)))) errors.push("Manifest SLM checksum mismatch");
  const trapStored = sequence.trapFrameStore.toArray();
  if (trapStored instanceof Promise) throw new TypeError("This frame store is asynchronous; use createSequencePackageAsync");
  const trapChecksum = sequence.manifest.checksums.trapFrames;
  if (trapChecksum && trapChecksum !== hashValue(trapStored)) errors.push("Manifest trap-frame checksum mismatch");
  return { valid: errors.length === 0, errors };
}
async function verifySequencePackageAsync(sequence, descriptors = sequence.slmFrameDescriptors) {
  const [stored, expectedDescriptors] = await Promise.all([
    sequence.slmFrameStore.toArray(),
    Promise.resolve(descriptors.map(normalizeDescriptor))
  ]);
  const errors = [];
  if (stored.length !== expectedDescriptors.length) errors.push("SLM frame count does not match descriptor count");
  for (let index = 0; index < Math.min(stored.length, expectedDescriptors.length); index += 1) {
    const descriptor = expectedDescriptors[index];
    const bytes = new Uint8Array(stored[index].buffer, stored[index].byteOffset, stored[index].byteLength);
    const expectedOffset = expectedDescriptors.slice(0, index).reduce((sum, item) => sum + item.byteLength, 0);
    if (descriptor.byteOffset !== BigInt(expectedOffset)) errors.push(`Frame ${index} byte offset mismatch`);
    if (descriptor.byteLength !== bytes.byteLength) errors.push(`Frame ${index} byte length mismatch`);
    if (descriptor.crc32 !== crc32(bytes)) errors.push(`Frame ${index} CRC32 mismatch`);
  }
  const manifestChecksum = sequence.manifest.checksums.frameDescriptors;
  if (manifestChecksum && manifestChecksum !== hashString(stableStringify(expectedDescriptors))) errors.push("Manifest descriptor checksum mismatch");
  const slmChecksum = sequence.manifest.checksums.slmFrames;
  if (slmChecksum && slmChecksum !== hashString(stableStringify(expectedDescriptors.map((descriptor) => descriptor.crc32)))) errors.push("Manifest SLM checksum mismatch");
  const trapStored = await sequence.trapFrameStore.toArray();
  const trapChecksum = sequence.manifest.checksums.trapFrames;
  if (trapChecksum && trapChecksum !== hashValue(trapStored)) errors.push("Manifest trap-frame checksum mismatch");
  return { valid: errors.length === 0, errors };
}
function verifySequencePackageFiles(sequence, files) {
  const result = verifySequencePackage(sequence);
  const errors = [...result.errors];
  let decoded;
  try {
    decoded = decodeTrapFrames(files["trap-frames.bin"]);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : "Unable to decode trap-frame binary");
    return { valid: false, errors };
  }
  const trapChecksum = sequence.manifest.checksums.trapFrames;
  if (trapChecksum && trapChecksum !== hashValue(decoded)) errors.push("Trap-frame binary checksum mismatch");
  const trapIndex = parseJsonArray(files["trap-frames.index.json"]);
  if (trapIndex.length !== decoded.length) errors.push("Trap-frame index count mismatch");
  for (let index = 0; index < Math.min(trapIndex.length, decoded.length); index += 1) {
    const entry = trapIndex[index];
    if (entry.frameIndex !== decoded[index].frameIndex || entry.timeUs !== decoded[index].timeUs || entry.byteOffset !== trapFrameOffset(decoded, index) || entry.byteLength !== trapFrameByteLength(decoded[index])) {
      errors.push(`Trap-frame index mismatch at frame ${index}`);
    }
  }
  const descriptors = sequence.slmFrameDescriptors;
  const packageDescriptors = parseJsonArray(files["slm-frames.index.json"]);
  if (packageDescriptors.length !== descriptors.length) errors.push("SLM frame index count mismatch");
  for (let index = 0; index < Math.min(packageDescriptors.length, descriptors.length); index += 1) {
    try {
      const packageDescriptor = normalizeDescriptor(packageDescriptors[index]);
      const expected = descriptors[index];
      if (packageDescriptor.frameIndex !== expected.frameIndex || packageDescriptor.timeUs !== expected.timeUs || packageDescriptor.byteOffset !== expected.byteOffset || packageDescriptor.byteLength !== expected.byteLength || packageDescriptor.crc32 !== expected.crc32) {
        errors.push(`SLM frame index mismatch at frame ${index}`);
      }
    } catch {
      errors.push(`Invalid SLM frame index at frame ${index}`);
    }
  }
  for (let index = 0; index < descriptors.length; index += 1) {
    const descriptor = descriptors[index];
    const start = Number(descriptor.byteOffset);
    const end = start + descriptor.byteLength;
    if (!Number.isSafeInteger(start) || start < 0 || end > files["slm-frames.bin"].byteLength) {
      errors.push(`SLM frame ${index} lies outside the binary`);
      continue;
    }
    const bytes = files["slm-frames.bin"].subarray(start, end);
    if (crc32(bytes) !== descriptor.crc32) errors.push(`SLM frame ${index} CRC32 mismatch in package`);
  }
  return { valid: errors.length === 0, errors };
}
var exportSequence = createSequencePackage;
function syncValue(value) {
  if (value instanceof Promise) {
    throw new TypeError("This frame store is asynchronous; use createSequencePackageAsync");
  }
  return value;
}
function descriptorForJson(descriptor) {
  return { ...descriptor, byteOffset: descriptor.byteOffset.toString() };
}
function normalizeDescriptor(descriptor) {
  return { ...descriptor, byteOffset: parseByteOffset(descriptor.byteOffset) };
}
function parseByteOffset(value) {
  if (typeof value === "bigint") return value;
  if (!/^\d+$/.test(value)) throw new TypeError(`Invalid decimal frame offset: ${value}`);
  return BigInt(value);
}
function parseJsonArray(value) {
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}
function trapFrameByteLength(frame) {
  return 16 + frame.traps.length * (4 + 4 + 1 + 8 + 8 + 8 + 8 + 4);
}
function trapFrameOffset(frames, frameIndex) {
  let offset = 0;
  for (let index = 0; index < frameIndex; index += 1) offset += trapFrameByteLength(frames[index]);
  return offset;
}

// src/compiler.ts
var SlmSequenceCompiler = class _SlmSequenceCompiler {
  disposed = false;
  cancelled = false;
  config;
  constructor(config = {}) {
    this.config = { ...config };
  }
  static async create(config = {}) {
    return new _SlmSequenceCompiler(config);
  }
  async compileRearrangement(request, options = {}) {
    this.assertUsable();
    const progress = (value) => options.onProgress?.(value);
    this.checkAbort(options.signal);
    progress({ stage: "VALIDATING", completed: 0, total: 1, message: "Normalizing inputs and calibration" });
    const normalized = normalizeAndValidate(request, this.config);
    progress({ stage: "VALIDATING", completed: 1, total: 1 });
    this.checkAbort(options.signal);
    progress({ stage: "ASSIGNING", completed: 0, total: 1 });
    const conflictPenalties = /* @__PURE__ */ new Map();
    let assignments = [];
    let motionPlan;
    let assignmentAttempts = 0;
    let lastPlanningError;
    const maximumAttempts = normalized.assignmentConfig.maxAssignmentRetries + 1;
    while (!motionPlan && assignmentAttempts < maximumAttempts) {
      this.checkAbort(options.signal);
      assignmentAttempts += 1;
      assignments = assignAtoms(normalized, conflictPenalties);
      progress({ stage: "ASSIGNING", completed: assignmentAttempts, total: maximumAttempts, message: `Assignment attempt ${assignmentAttempts}` });
      try {
        progress({ stage: "PLANNING", completed: 0, total: 1 });
        motionPlan = planMotion(normalized, assignments);
      } catch (error) {
        lastPlanningError = error;
        if (!(error instanceof SlmError) || error.code !== "PATH_NOT_FOUND" && error.code !== "COLLISION_VALIDATION_FAILED" || assignmentAttempts >= maximumAttempts) {
          throw error;
        }
        for (const assignment of assignments) {
          if (assignment.targetSiteId !== null) {
            const key = `${assignment.atomId}:${assignment.targetSiteId}`;
            conflictPenalties.set(key, (conflictPenalties.get(key) ?? 0) + normalized.assignmentConfig.conflictPenalty);
          }
        }
      }
    }
    if (!motionPlan) throw lastPlanningError instanceof Error ? lastPlanningError : new SlmError("PATH_NOT_FOUND", "No feasible motion plan was found", { stage: "PLANNING" });
    progress({ stage: "ASSIGNING", completed: assignmentAttempts, total: maximumAttempts, message: "Hungarian assignment complete" });
    progress({ stage: "PLANNING", completed: 1, total: 1, message: "Collision-free motion plan complete" });
    this.checkAbort(options.signal);
    const finalMotionPlan = motionPlan;
    progress({ stage: "PARAMETERIZING", completed: 0, total: finalMotionPlan.trajectories.length });
    progress({ stage: "PARAMETERIZING", completed: finalMotionPlan.trajectories.length, total: finalMotionPlan.trajectories.length });
    this.checkAbort(options.signal);
    progress({ stage: "GENERATING_TRAP_FRAMES", completed: 0, total: 1 });
    const trapFrames = sampleTrapFrames(normalized, finalMotionPlan.trajectories);
    let trapValidation = validateTrapFrames(trapFrames, normalized, false);
    if (!trapValidation.accepted) {
      throw compilationFailure("COLLISION_VALIDATION_FAILED", "Generated trap frames failed validation", trapValidation.errors);
    }
    progress({ stage: "GENERATING_TRAP_FRAMES", completed: 1, total: 1, message: `${trapFrames.length} trap frames generated` });
    this.checkAbort(options.signal);
    const trapFrameStore = new MemoryFrameStore();
    const slmFrameStore = options.outputStore ?? new MemoryFrameStore();
    const frameMetrics = [];
    const slmFrameDescriptors = [];
    const solver = new SequentialWgsSolver(normalized.calibration, normalized.hologramConfig);
    let byteOffset = 0n;
    let insertedFrames = 0;
    progress({ stage: "SOLVING_SLM_FRAMES", completed: 0, total: trapFrames.length });
    for (let index = 0; index < trapFrames.length; index += 1) {
      this.checkAbort(options.signal);
      const frame = trapFrames[index];
      let solve = solver.solveSequentialFrame(frame);
      let retry = 0;
      while (!solve.metrics.accepted && retry < 2) {
        retry += 1;
        solver.rollbackToPreviousAcceptedFrame();
        const budget = Math.min(
          normalized.hologramConfig.maxIterations,
          Math.max(solve.metrics.iterations + 1, solve.metrics.iterations * 2)
        );
        solve = solver.solveSequentialFrame(frame, budget);
      }
      if (!solve.metrics.accepted) {
        if (index > 0 && insertedFrames < normalized.hologramConfig.maxInsertedFrames && trapFrames[index].timeUs - trapFrames[index - 1].timeUs > 1) {
          const currentTime = trapFrames[index].timeUs;
          const originalMidpointTime = (trapFrames[index - 1].timeUs + trapFrames[index].timeUs) / 2;
          const midpoint = midpointTrapFrame(trapFrames[index - 1], trapFrames[index], index, finalMotionPlan.trajectories, normalized, originalMidpointTime);
          midpoint.timeUs = trapFrames[index - 1].timeUs + normalized.motionConfig.framePeriodUs;
          insertTrajectoryMidpoint(finalMotionPlan.trajectories, midpoint, currentTime, normalized.motionConfig.framePeriodUs);
          finalMotionPlan.makespanUs += normalized.motionConfig.framePeriodUs;
          const regenerated = sampleTrapFrames(normalized, finalMotionPlan.trajectories);
          trapFrames.splice(index, trapFrames.length - index, ...regenerated.slice(index));
          insertedFrames += 1;
          solver.rollbackToPreviousAcceptedFrame();
          index -= 1;
          continue;
        }
        throw compilationFailure("FRAME_QUALITY_REJECTED", `SLM frame ${index} failed quality gates`, [
          {
            code: "FRAME_QUALITY_REJECTED",
            stage: "SOLVING_SLM_FRAMES",
            message: `Frame ${index} did not pass configured quality gates`,
            frameIndex: index
          }
        ]);
      }
      try {
        await trapFrameStore.append(frame);
        await slmFrameStore.append(solve.pixels);
      } catch (error) {
        throw new SlmError("STORAGE_ERROR", `Unable to store SLM frame ${index}`, {
          stage: "WRITING_OUTPUT",
          retryable: true,
          cause: error
        });
      }
      solver.commitFrameState();
      const descriptor = frameDescriptor(
        frame,
        solve.pixels,
        normalized.hologramConfig.width,
        normalized.hologramConfig.height,
        normalized.hologramConfig.format,
        byteOffset
      );
      byteOffset += BigInt(descriptor.byteLength);
      slmFrameDescriptors.push(descriptor);
      frameMetrics.push({ ...solve.metrics, refinementCount: retry });
      progress({ stage: "SOLVING_SLM_FRAMES", completed: index + 1, total: trapFrames.length, frameIndex: index });
    }
    this.checkAbort(options.signal);
    progress({ stage: "VALIDATING_SEQUENCE", completed: 0, total: 1 });
    trapValidation = validateTrapFrames(trapFrames, normalized, false);
    if (!trapValidation.accepted) {
      throw compilationFailure("COLLISION_VALIDATION_FAILED", "Refined trap frames failed validation", trapValidation.errors);
    }
    const validation = validateCompleteSequence(
      assignments,
      finalMotionPlan,
      trapFrames,
      slmFrameDescriptors,
      frameMetrics,
      normalized,
      trapValidation
    );
    if (!validation.accepted) {
      throw compilationFailure("FRAME_QUALITY_REJECTED", "Complete sequence validation failed", validation.errors);
    }
    progress({ stage: "VALIDATING_SEQUENCE", completed: 1, total: 1 });
    progress({ stage: "WRITING_OUTPUT", completed: 0, total: 1 });
    const manifest = makeManifest(request, normalized, assignments, finalMotionPlan, trapFrames, slmFrameDescriptors, validation, this.config, assignmentAttempts);
    progress({ stage: "WRITING_OUTPUT", completed: 1, total: 1 });
    return {
      manifest,
      assignment: assignments,
      trajectories: finalMotionPlan.trajectories,
      trapFrameStore,
      slmFrameStore,
      slmFrameDescriptors,
      frameMetrics,
      validation
    };
  }
  async planOnly(request) {
    this.assertUsable();
    const normalized = normalizeAndValidate(request, this.config);
    const penalties = /* @__PURE__ */ new Map();
    let lastError;
    for (let attempt = 0; attempt <= normalized.assignmentConfig.maxAssignmentRetries; attempt += 1) {
      const assignments = assignAtoms(normalized, penalties);
      try {
        return planMotion(normalized, assignments);
      } catch (error) {
        lastError = error;
        if (!(error instanceof SlmError) || error.code !== "PATH_NOT_FOUND" && error.code !== "COLLISION_VALIDATION_FAILED" || attempt === normalized.assignmentConfig.maxAssignmentRetries) throw error;
        for (const assignment of assignments) if (assignment.targetSiteId !== null) {
          const key = `${assignment.atomId}:${assignment.targetSiteId}`;
          penalties.set(key, (penalties.get(key) ?? 0) + normalized.assignmentConfig.conflictPenalty);
        }
      }
    }
    throw lastError;
  }
  async solveTrapFrames(frames) {
    this.assertUsable();
    const calibration = this.config.calibration ?? Object.values(this.config.calibrations ?? {})[0];
    if (!calibration) throw new SlmError("CALIBRATION_MISMATCH", "solveTrapFrames requires a calibration package", { stage: "VALIDATING" });
    const normalized = normalizeAndValidate({
      initialAtoms: [],
      targetSites: [],
      calibration
    }, this.config);
    const trapFrameStore = new MemoryFrameStore();
    const slmFrameStore = new MemoryFrameStore();
    const frameMetrics = [];
    const descriptors = [];
    const solver = new SequentialWgsSolver(calibration, normalized.hologramConfig);
    let byteOffset = 0n;
    for await (const frame of frames) {
      const result = solver.solveSequentialFrame(frame);
      if (!result.metrics.accepted) throw new SlmError("FRAME_QUALITY_REJECTED", `Frame ${frame.frameIndex} failed quality gates`, { stage: "SOLVING_SLM_FRAMES", retryable: true });
      trapFrameStore.append(frame);
      slmFrameStore.append(result.pixels);
      solver.commitFrameState();
      const descriptor = frameDescriptor(frame, result.pixels, normalized.hologramConfig.width, normalized.hologramConfig.height, normalized.hologramConfig.format, byteOffset);
      byteOffset += BigInt(descriptor.byteLength);
      descriptors.push(descriptor);
      frameMetrics.push(result.metrics);
    }
    const validation = validateTrapFrames(trapFrameStore.toArray(), normalized);
    if (!validation.accepted) throw compilationFailure("COLLISION_VALIDATION_FAILED", "Trap frames failed validation", validation.errors);
    const manifest = {
      formatVersion: "1.1",
      creationTimestamp: this.config.creationTimestamp ?? "1970-01-01T00:00:00.000Z",
      compilerVersion: this.config.compilerVersion ?? "0.1.0",
      wasmBuildId: this.config.wasmBuildId ?? "typescript-reference",
      inputHash: hashValue(descriptors),
      calibrationId: calibration.manifest.calibrationId,
      calibrationHash: calibrationHash(calibration),
      coordinateConvention: calibration.manifest.coordinateConvention ?? "+x right, +y up",
      atomCount: 0,
      targetCount: 0,
      trapCount: trapFrameStore.length > 0 ? trapFrameStore.get(0).traps.length : 0,
      frameCount: descriptors.length,
      framePeriodUs: normalized.motionConfig.framePeriodUs,
      assignmentCost: 0,
      assignmentAttempts: 1,
      plannerBackend: "none",
      plannerParameters: {},
      wgsBackend: "scalar-phase-locked-wgs",
      wgsParameters: {
        targetPhaseMode: normalized.hologramConfig.targetPhaseMode,
        firstFrameIterations: normalized.hologramConfig.firstFrameIterations,
        subsequentFrameIterations: normalized.hologramConfig.subsequentFrameIterations,
        gamma: normalized.hologramConfig.gamma,
        epsilon: normalized.hologramConfig.epsilon
      },
      outputWidth: normalized.hologramConfig.width,
      outputHeight: normalized.hologramConfig.height,
      pixelFormat: normalized.hologramConfig.format,
      deterministicSeed: normalized.hologramConfig.deterministicSeed,
      checksums: {
        frameDescriptors: hashValue(descriptors),
        slmFrames: hashValue(descriptors.map((descriptor) => descriptor.crc32)),
        trapFrames: hashValue(trapFrameStore.toArray())
      },
      validationStatus: "accepted"
    };
    return { manifest, assignment: [], trajectories: [], trapFrameStore, slmFrameStore, slmFrameDescriptors: descriptors, frameMetrics, validation };
  }
  dispose() {
    this.disposed = true;
  }
  reset() {
    if (this.disposed) throw new SlmError("INVALID_ARGUMENT", "Compiler has been disposed", { stage: "CREATED" });
    this.cancelled = false;
  }
  assertUsable() {
    if (this.disposed) throw new SlmError("INVALID_ARGUMENT", "Compiler has been disposed", { stage: "CREATED" });
    if (this.cancelled) throw new SlmError("INVALID_ARGUMENT", "Compiler was cancelled; call reset before reuse", { stage: "CANCELLED" });
  }
  checkAbort(signal) {
    try {
      checkAbort(signal);
    } catch (error) {
      if (error instanceof SlmError && error.code === "CANCELLED") this.cancelled = true;
      throw error;
    }
  }
};
function validateCompleteSequence(assignments, motionPlan, trapFrames, descriptors, metrics, input, trapValidation) {
  const errors = [...trapValidation.errors];
  const warnings = [...trapValidation.warnings];
  const continuous = validateContinuousTrajectories(motionPlan.trajectories, input);
  errors.push(...continuous.errors);
  warnings.push(...continuous.warnings);
  const requiredTargets = input.targetSites.filter((target) => target.required);
  const assignedTargetIds = assignments.filter((assignment) => assignment.targetSiteId !== null).map((assignment) => assignment.targetSiteId);
  if (new Set(assignedTargetIds).size !== assignedTargetIds.length || requiredTargets.some((target) => !assignedTargetIds.includes(target.siteId))) {
    errors.push({ code: "ASSIGNMENT_INFEASIBLE", stage: "VALIDATING_SEQUENCE", message: "Required targets do not have exactly one assigned atom" });
  }
  if (trapFrames.length !== descriptors.length || trapFrames.length !== metrics.length) {
    errors.push({ code: "FRAME_COUNT_MISMATCH", stage: "VALIDATING_SEQUENCE", message: "Each trap frame must have one solved SLM frame" });
  }
  for (const assignment of assignments) {
    if (assignment.targetSiteId === null) continue;
    const target = input.targetSites.find((candidate) => candidate.siteId === assignment.targetSiteId);
    const trajectory = motionPlan.trajectories.find((candidate) => candidate.atomId === assignment.atomId);
    if (!target || !trajectory) continue;
    const final = trajectory.waypoints.at(-1);
    const tolerance = Math.max(1e-6, input.plannerConfig.geometricMarginUm * 0.01);
    if (Math.hypot(final.xUm - target.xUm, final.yUm - target.yUm) > tolerance) {
      errors.push({ code: "FINAL_POSITION_MISMATCH", stage: "VALIDATING_SEQUENCE", message: "Trajectory final position does not match target", atomIds: [assignment.atomId], targetSiteIds: [target.siteId], configured: tolerance, measured: Math.hypot(final.xUm - target.xUm, final.yUm - target.yUm) });
    }
    const finalTrap = trapFrames.at(-1)?.traps.find((trap) => trap.trapId === trajectory.trapId);
    const expectedIntensity = target.finalTrapIntensity ?? input.motionConfig.defaultTrapIntensity;
    if (!finalTrap || finalTrap.atomId !== assignment.atomId || finalTrap.intensity <= 0 || Math.abs(finalTrap.intensity - expectedIntensity) > 1e-6) {
      errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "VALIDATING_SEQUENCE", message: "A selected atom was not retained in its final trap", atomIds: [assignment.atomId], targetSiteIds: [target.siteId] });
    }
  }
  for (const metric of metrics) {
    if (!metric.numericalValid || !metric.accepted) errors.push({ code: "FRAME_QUALITY_REJECTED", stage: "VALIDATING_SEQUENCE", message: `Frame ${metric.frameIndex} is not accepted`, frameIndex: metric.frameIndex });
    else if (!metric.converged) warnings.push({ code: "WGS_NOT_CONVERGED", stage: "VALIDATING_SEQUENCE", message: `Frame ${metric.frameIndex} exhausted its WGS budget without convergence`, frameIndex: metric.frameIndex });
  }
  return {
    accepted: errors.length === 0,
    errors,
    warnings,
    minimumAtomSeparationUm: Math.min(trapValidation.minimumAtomSeparationUm, continuous.minimumAtomSeparationUm, motionPlan.minimumValidatedSeparationUm),
    maximumSpeedUmPerUs: continuous.maximumSpeedUmPerUs,
    maximumAccelerationUmPerUs2: continuous.maximumAccelerationUmPerUs2,
    maximumJerkUmPerUs3: continuous.maximumJerkUmPerUs3,
    frameCount: trapFrames.length
  };
}
function makeManifest(request, normalized, assignments, motionPlan, trapFrames, descriptors, validation, compilerConfig, assignmentAttempts) {
  return {
    formatVersion: "1.1",
    creationTimestamp: compilerConfig.creationTimestamp ?? "1970-01-01T00:00:00.000Z",
    compilerVersion: compilerConfig.compilerVersion ?? "0.1.0",
    wasmBuildId: compilerConfig.wasmBuildId ?? "typescript-reference",
    inputHash: inputHash(request, normalized),
    calibrationId: normalized.calibration.manifest.calibrationId,
    calibrationHash: calibrationHash(normalized.calibration),
    coordinateConvention: normalized.calibration.manifest.coordinateConvention ?? "+x right, +y up",
    atomCount: normalized.initialAtoms.length,
    targetCount: normalized.targetSites.length,
    trapCount: motionPlan.trajectories.length + normalized.staticTraps.length,
    frameCount: trapFrames.length,
    framePeriodUs: normalized.motionConfig.framePeriodUs,
    assignmentCost: assignmentCost(assignments),
    assignmentAttempts,
    plannerBackend: motionPlan.conflictComponentCount > 0 ? "direct+cbs-reference" : "direct-reference",
    plannerParameters: {
      minimumSeparationUm: normalized.plannerConfig.minimumSeparationUm,
      gridResolutionUm: normalized.plannerConfig.gridResolutionUm,
      waitCount: motionPlan.waitCount,
      detourCount: motionPlan.detourCount
    },
    wgsBackend: "scalar-phase-locked-wgs",
    wgsParameters: {
      targetPhaseMode: normalized.hologramConfig.targetPhaseMode,
      firstFrameIterations: normalized.hologramConfig.firstFrameIterations,
      subsequentFrameIterations: normalized.hologramConfig.subsequentFrameIterations,
      gamma: normalized.hologramConfig.gamma,
      epsilon: normalized.hologramConfig.epsilon
    },
    outputWidth: normalized.hologramConfig.width,
    outputHeight: normalized.hologramConfig.height,
    pixelFormat: normalized.hologramConfig.format,
    deterministicSeed: normalized.hologramConfig.deterministicSeed,
    checksums: {
      frameDescriptors: hashValue(descriptors),
      slmFrames: hashValue(descriptors.map((descriptor) => descriptor.crc32)),
      trapFrames: hashValue(trapFrames),
      validation: hashValue(validation)
    },
    validationStatus: validation.accepted ? "accepted" : "rejected"
  };
}
function compilationFailure(code, message, issues) {
  return new SlmCompilationError(code, message, issues, { stage: "VALIDATING_SEQUENCE", retryable: true });
}
function midpointTrapFrame(previous, current, frameIndex, trajectories, input, originalMidpointTime) {
  const currentById = new Map(current.traps.map((trap) => [trap.trapId, trap]));
  const traps = previous.traps.map((first) => {
    const second = currentById.get(first.trapId) ?? first;
    const trajectory = trajectories.find((candidate) => candidate.trapId === first.trapId);
    if (trajectory) {
      const point = sampleTrajectory(trajectory, originalMidpointTime);
      const intensity = sampleTrajectoryIntensity(trajectory, originalMidpointTime, input);
      return {
        trapId: first.trapId,
        atomId: intensity <= 1e-9 ? null : trajectory.atomId,
        xUm: point.xUm,
        yUm: point.yUm,
        intensity,
        targetPhaseRad: (first.targetPhaseRad + second.targetPhaseRad) / 2,
        flags: (trajectory.moving ? 1 : 0) | (intensity <= 1e-9 ? 2 : 0)
      };
    }
    return {
      trapId: first.trapId,
      atomId: first.atomId ?? second.atomId,
      xUm: (first.xUm + second.xUm) / 2,
      yUm: (first.yUm + second.yUm) / 2,
      intensity: (first.intensity + second.intensity) / 2,
      targetPhaseRad: (first.targetPhaseRad + second.targetPhaseRad) / 2,
      flags: (distanceSquared2(first, second) > 1e-18 ? 1 : 0) | (first.atomId === null && second.atomId === null && first.intensity <= 0 && second.intensity <= 0 ? 2 : 0) | first.flags & second.flags & 4
    };
  });
  return {
    frameIndex,
    timeUs: Math.floor((previous.timeUs + current.timeUs) / 2),
    traps
  };
}
function distanceSquared2(first, second) {
  return (first.xUm - second.xUm) ** 2 + (first.yUm - second.yUm) ** 2;
}
function insertTrajectoryMidpoint(trajectories, midpoint, originalCurrentTime, period) {
  for (const trajectory of trajectories) {
    const state = midpoint.traps.find((trap) => trap.trapId === trajectory.trapId);
    if (!state) continue;
    const shifted = trajectory.waypoints.map((waypoint) => ({
      ...waypoint,
      arrivalTimeUs: waypoint.arrivalTimeUs >= originalCurrentTime ? waypoint.arrivalTimeUs + period : waypoint.arrivalTimeUs
    }));
    const inserted = {
      xUm: state.xUm,
      yUm: state.yUm,
      arrivalTimeUs: midpoint.timeUs
    };
    const position = shifted.findIndex((waypoint) => waypoint.arrivalTimeUs >= inserted.arrivalTimeUs);
    shifted.splice(position < 0 ? shifted.length : position, 0, inserted);
    trajectory.waypoints = shifted;
    trajectory.endTimeUs = Math.max(trajectory.endTimeUs + (trajectory.endTimeUs >= originalCurrentTime ? period : 0), inserted.arrivalTimeUs);
  }
}
export {
  ASSIGNMENT_INFINITY,
  DEFAULT_ASSIGNMENT_CONFIG,
  DEFAULT_HOLOGRAM_CONFIG,
  DEFAULT_MOTION_CONFIG,
  DEFAULT_PLANNER_CONFIG,
  MappedFrameStore,
  MemoryFrameStore,
  SequentialWgsSolver,
  SlmCompilationError,
  SlmError,
  SlmSequenceCompiler,
  TAU,
  WgsSolver,
  angularDistance,
  asNumberArray,
  assertFinite,
  assignAtoms,
  assignmentCost,
  atomTrapIds,
  boundingBox,
  buildAssignmentCostMatrix,
  buildDirectPaths,
  calibrationHash,
  checkAbort,
  clamp,
  cloneComplexField,
  cloneTyped,
  concatenateSlmFrames,
  connectedConflictComponents,
  crc32,
  createComplexField,
  createSequencePackage,
  createSequencePackageAsync,
  decodeTrapFrames,
  distance,
  distancePointToPolygon,
  distancePointToSegment,
  distanceSquared,
  encodeTrapFrames,
  exportSequence,
  fft1d,
  fft2d,
  fieldPower,
  findPathConflicts,
  frameDescriptor,
  generateTrapFrames,
  hashString,
  hashValue,
  hungarianSolve,
  inputHash,
  integerOr,
  lerpPoint,
  minimumJerk,
  minimumJerkDerivative,
  minimumJerkSecondDerivative,
  minimumJerkThirdDerivative,
  minimumSegmentDuration,
  normalizeAndValidate,
  normalizeInput,
  normalizeLatticeOrAtoms,
  numberOr,
  orientation,
  parameterizeTrajectories,
  pathClearOfStaticAtoms,
  pathIntersectsForbiddenRegion,
  planMotion,
  planPaths,
  pointClearOfStaticAtoms,
  pointInForbiddenRegion,
  pointInPolygon,
  regionPolygon,
  sampleComplex,
  sampleTrajectory,
  sampleTrajectoryIntensity,
  sampleTrapFrames,
  scatterComplex,
  scatterComplexAdjoint,
  segmentDistance,
  segmentIntersectsForbiddenRegion,
  segmentPoint,
  segmentsIntersect,
  smoothstep5,
  solveAssignment,
  solveHologramFrame,
  stableStringify,
  targetPhaseForTrap,
  trapPhaseMap,
  validateCalibrationManifest,
  validateCalibrationPackage,
  validateContinuousTrajectories,
  validateTrapFrames,
  verifySequencePackage,
  verifySequencePackageAsync,
  verifySequencePackageFiles,
  wrapPhase
};
