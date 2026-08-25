import { SlmError } from "./errors.js";
import {
  distance,
  distancePointToSegment,
  pathIntersectsForbiddenRegion,
  pointInForbiddenRegion,
} from "./geometry.js";
import type {
  AtomAssignment,
  AtomTrajectory,
  NormalizedInput,
  PlannedPath,
  Point2D,
  SequenceValidationReport,
  TrapFrame,
  TrapState,
  TrajectoryWaypoint,
  ValidationIssue,
} from "./types.js";
import { clamp, hashString, smoothstep5 } from "./util.js";

const EPSILON = 1e-9;

export function minimumJerk(s: number): number {
  return smoothstep5(s);
}

export function minimumJerkDerivative(s: number): number {
  const t = clamp(s, 0, 1);
  return 30 * t * t * (t - 1) * (t - 1);
}

export function minimumJerkSecondDerivative(s: number): number {
  const t = clamp(s, 0, 1);
  return 60 * t - 180 * t * t + 120 * t * t * t;
}

export function minimumJerkThirdDerivative(s: number): number {
  const t = clamp(s, 0, 1);
  return 60 - 360 * t + 360 * t * t;
}

export function parameterizeTrajectories(input: NormalizedInput, paths: PlannedPath[]): AtomTrajectory[] {
  const commonMoveDuration = Math.max(
    input.motionConfig.framePeriodUs,
    ...paths.flatMap((path) => path.waypointsUm.slice(1).map((point, index) => {
      const previous = path.waypointsUm[index]!;
      const length = distance(previous, point);
      return length <= EPSILON ? input.motionConfig.framePeriodUs : minimumSegmentDuration(length, input, input.plannerConfig.planningTickUs);
    })),
  );
  const trajectories = paths.map((path) => {
    const atom = input.initialAtoms.find((candidate) => candidate.atomId === path.atomId);
    if (!atom) throw new SlmError("INTERNAL_ERROR", `Unknown atom ${path.atomId} in planned path`, { stage: "PARAMETERIZING" });
    const assignment = findAssignment(input, path);
    const initialIntensity = atom.initialTrapIntensity;
    const finalIntensity = targetIntensity(input, assignment, initialIntensity);
    const moving = path.waypointsUm.some((point, index) => index > 0 && distance(point, path.waypointsUm[index - 1]!) > EPSILON);
    const waypoints: TrajectoryWaypoint[] = [];
    const basePreDwell = Math.max(input.motionConfig.preMoveDwellUs, input.motionConfig.minDwellBeforeMoveUs);
    const boostRamp = moving
      ? intensityRampDuration(Math.abs(input.motionConfig.movingTrapIntensity - initialIntensity), input, input.motionConfig.minDwellBeforeMoveUs)
      : 0;
    const preDwell = roundUp(basePreDwell + boostRamp, input.motionConfig.framePeriodUs);
    const scheduledStart = roundUp(Math.max(0, path.startTimeUs ?? 0), input.motionConfig.framePeriodUs);
    const ticks = path.discreteTicks.length === path.waypointsUm.length
      ? path.discreteTicks
      : path.waypointsUm.map((_, index) => index);
    let cursor = 0;
    const firstPoint = path.waypointsUm[0]!;
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
      const previous = path.waypointsUm[index - 1]!;
      const current = path.waypointsUm[index]!;
      const tickDelta = Math.max(1, (ticks[index] ?? index) - (ticks[index - 1] ?? index - 1));
      const requestedDuration = tickDelta * input.plannerConfig.planningTickUs;
      const length = distance(previous, current);
      const duration = roundUp(
        Math.max(requestedDuration, commonMoveDuration * tickDelta),
        input.motionConfig.framePeriodUs,
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
      finalIntensity,
    } satisfies AtomTrajectory;
  });

  const finalTime = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  // Keep the time domain common so a consumer can replay every trap with one clock.
  return trajectories.map((trajectory) => {
    if (trajectory.endTimeUs < finalTime) {
      const last = trajectory.waypoints.at(-1)!;
      trajectory.waypoints.push({ xUm: last.xUm, yUm: last.yUm, arrivalTimeUs: finalTime });
      trajectory.endTimeUs = finalTime;
    }
    return trajectory;
  });
}

export function minimumSegmentDuration(length: number, input: NormalizedInput, requestedDuration = 0): number {
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

export function sampleTrajectory(trajectory: AtomTrajectory, timeUs: number): Point2D {
  const waypoints = trajectory.waypoints;
  if (waypoints.length === 0) return { xUm: 0, yUm: 0 };
  if (timeUs <= waypoints[0]!.arrivalTimeUs) return pointOf(waypoints[0]!);
  const last = waypoints.length - 1;
  if (timeUs >= waypoints[last]!.arrivalTimeUs) return pointOf(waypoints[last]!);
  for (let index = 0; index < last; index += 1) {
    const start = waypoints[index]!;
    const end = waypoints[index + 1]!;
    if (timeUs <= end.arrivalTimeUs) {
      const duration = end.arrivalTimeUs - start.arrivalTimeUs;
      const fraction = duration <= 0 ? 1 : (timeUs - start.arrivalTimeUs) / duration;
      const q = minimumJerk(fraction);
      return {
        xUm: start.xUm + (end.xUm - start.xUm) * q,
        yUm: start.yUm + (end.yUm - start.yUm) * q,
      };
    }
  }
  return pointOf(waypoints[last]!);
}

export function sampleTrajectoryIntensity(
  trajectory: AtomTrajectory,
  timeUs: number,
  input: NormalizedInput,
): number {
  const initial = trajectory.initialIntensity ?? input.motionConfig.defaultTrapIntensity;
  const final = trajectory.finalIntensity ?? initial;
  if (!trajectory.moving) {
    const settleDuration = intensityRampDuration(Math.abs(final - initial), input, input.motionConfig.postMoveSettleUs);
    const settled = smoothstep5((timeUs - trajectory.endTimeUs) / settleDuration);
    const settledIntensity = initial + (final - initial) * settled;
    if (trajectory.disposition === "RELEASE" ||
        (trajectory.disposition === "PARK" && input.assignmentConfig.extraAtomPolicy === "PARK_AND_RELEASE")) {
      const releaseStart = trajectory.endTimeUs + settleDuration;
      const releaseDuration = intensityRampDuration(final, input);
      return settledIntensity * (1 - smoothstep5((timeUs - releaseStart) / releaseDuration));
    }
    return settledIntensity;
  }
  const firstMovingIndex = findFirstMovingSegment(trajectory);
  const firstMovingStart = firstMovingIndex < 0 ? 0 : trajectory.waypoints[firstMovingIndex]!.arrivalTimeUs;
  const lastMovingIndex = findLastMovingSegment(trajectory);
  const lastMovingEnd = lastMovingIndex < 0 ? trajectory.endTimeUs : trajectory.waypoints[lastMovingIndex + 1]!.arrivalTimeUs;
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

export function sampleTrapFrames(
  input: NormalizedInput,
  trajectories: AtomTrajectory[],
  seed = input.hologramConfig.deterministicSeed,
): TrapFrame[] {
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
      input.motionConfig.postMoveSettleUs,
    );
    const boostStart = trajectory.moving ? firstMovingStartTime(trajectory) : movementEnd;
    const settleStart = trajectory.moving ? Math.max(movementEnd, boostStart) : movementEnd;
    const release = trajectory.disposition === "RELEASE" ||
      (trajectory.disposition === "PARK" && input.assignmentConfig.extraAtomPolicy === "PARK_AND_RELEASE")
      ? intensityRampDuration(final, input)
      : 0;
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
    targetPhaseRad: phases.get(trap.trapId)!,
    flags: trap.containsAtom ? 4 : 0,
  }));
  const frames: TrapFrame[] = [];
  for (let time = 0, frameIndex = 0; time <= totalTime; time += period, frameIndex += 1) {
    const dynamicStates: TrapState[] = trajectories.map((trajectory) => {
      const point = sampleTrajectory(trajectory, time);
      const released = sampleTrajectoryIntensity(trajectory, time, input) <= EPSILON;
      return {
        trapId: trajectory.trapId,
        atomId: released ? null : trajectory.atomId,
        xUm: point.xUm,
        yUm: point.yUm,
        intensity: Math.max(0, sampleTrajectoryIntensity(trajectory, time, input)),
        targetPhaseRad: phases.get(trajectory.trapId)!,
        flags: (trajectory.moving && !released ? 1 : 0) | (released ? 2 : 0),
      };
    });
    const traps = [...staticStates, ...dynamicStates].sort((a, b) => a.trapId - b.trapId);
    frames.push({ frameIndex, timeUs: time, traps });
  }
  return frames;
}

export function validateTrapFrames(
  frames: TrapFrame[],
  input: NormalizedInput,
  checkMotionLimits = true,
): SequenceValidationReport {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  let minimumSeparation = Number.POSITIVE_INFINITY;
  let maximumSpeed = 0;
  let maximumAcceleration = 0;
  let maximumJerk = 0;
  const previousByTrap = new Map<number, TrapState>();
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
    const currentIds = new Set<number>();
    const currentAtomIds = new Set<number>();
    for (const [trapId, previous] of previousByTrap) {
      if (!frame.traps.some((trap) => trap.trapId === trapId)) {
        if (previous.intensity > EPSILON) {
          errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAP_FRAMES", message: "A live trap disappeared without a zero-intensity ramp", frameIndex });
        }
        previousByTrap.delete(trapId);
      }
    }
    for (let trapIndex = 0; trapIndex < frame.traps.length; trapIndex += 1) {
      const trap = frame.traps[trapIndex]!;
      if (trapIndex > 0 && frame.traps[trapIndex - 1]!.trapId >= trap.trapId) {
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
      const a = frame.traps[first]!;
      if (a.intensity <= EPSILON) continue;
      for (let second = first + 1; second < frame.traps.length; second += 1) {
        const b = frame.traps[second]!;
        if (b.intensity <= EPSILON) continue;
        const separation = distance(a, b);
        minimumSeparation = Math.min(minimumSeparation, separation);
        const atomA = input.initialAtoms.find((atom) => atom.atomId === a.atomId);
        const atomB = input.initialAtoms.find((atom) => atom.atomId === b.atomId);
        const safe = input.plannerConfig.minimumSeparationUm +
          input.plannerConfig.kSigma * ((atomA?.localizationSigmaUm ?? 0) + (atomB?.localizationSigmaUm ?? 0)) +
          input.plannerConfig.geometricMarginUm;
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
    frameCount: frames.length,
  };
}

export function validateContinuousTrajectories(
  trajectories: AtomTrajectory[],
  input: NormalizedInput,
  checkMotionLimits = true,
): SequenceValidationReport {
  const errors: ValidationIssue[] = [];
  const warnings: ValidationIssue[] = [];
  let minimumSeparation = Number.POSITIVE_INFINITY;
  let maximumSpeed = 0;
  let maximumAcceleration = 0;
  let maximumJerk = 0;
  const maxTime = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  const staticPoints = [
    ...input.initialAtoms.filter((atom) => !atom.movable),
    ...input.staticTraps.filter((trap) => trap.containsAtom),
  ];
  for (const trajectory of trajectories) {
    for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
      const start = trajectory.waypoints[index]!;
      const end = trajectory.waypoints[index + 1]!;
      const length = distance(start, end);
      const duration = end.arrivalTimeUs - start.arrivalTimeUs;
      if (duration <= 0) {
        errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAJECTORIES", message: "Trajectory times are not increasing", atomIds: [trajectory.atomId] });
        continue;
      }
      maximumSpeed = Math.max(maximumSpeed, length * 1.875 / duration);
      maximumAcceleration = Math.max(maximumAcceleration, length * 5.774 / (duration * duration));
      maximumJerk = Math.max(maximumJerk, length * 60 / (duration * duration * duration));
      if (checkMotionLimits && (maximumSpeed > input.motionConfig.maxVelocityUmPerUs + EPSILON ||
          maximumAcceleration > input.motionConfig.maxAccelerationUmPerUs2 + EPSILON ||
          maximumJerk > input.motionConfig.maxJerkUmPerUs3 + EPSILON)) {
        errors.push({ code: "MOTION_LIMIT_VIOLATION", stage: "TRAJECTORIES", message: "Trajectory exceeds a configured motion limit", atomIds: [trajectory.atomId] });
      }
      if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([pointOf(start), pointOf(end)], [region]))) {
        errors.push({ code: "FORBIDDEN_REGION", stage: "TRAJECTORIES", message: "Trajectory segment intersects a forbidden region", atomIds: [trajectory.atomId] });
      }
    }
    const ownAtom = input.initialAtoms.find((atom) => atom.atomId === trajectory.atomId);
    const ownStaticTrap = trajectory.staticTrap ? input.staticTraps.find((trap) => trap.trapId === trajectory.trapId) : undefined;
    if (ownStaticTrap && trajectory.waypoints.some((waypoint) =>
      Math.abs(waypoint.xUm - ownStaticTrap.xUm) > EPSILON || Math.abs(waypoint.yUm - ownStaticTrap.yUm) > EPSILON)) {
      errors.push({ code: "ILLEGAL_LIFECYCLE", stage: "TRAJECTORIES", message: "A static trap trajectory moved", atomIds: [trajectory.atomId] });
    }
    const ownStaticPoints = staticPoints.filter((point) => {
      if (ownAtom && point.xUm === ownAtom.xUm && point.yUm === ownAtom.yUm && !ownAtom.movable) return false;
      if (ownStaticTrap && point.xUm === ownStaticTrap.xUm && point.yUm === ownStaticTrap.yUm) return false;
      return true;
    });
    const staticClearance = input.plannerConfig.minimumSeparationUm +
      input.plannerConfig.kSigma * (ownAtom?.localizationSigmaUm ?? 0) +
      input.plannerConfig.geometricMarginUm;
    for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
      const start = trajectory.waypoints[index]!;
      const end = trajectory.waypoints[index + 1]!;
      const segment = { start: pointOf(start), end: pointOf(end) };
      if (ownStaticPoints.some((point) => distancePointToSegment(point, segment) < staticClearance - EPSILON)) {
        errors.push({ code: "COLLISION_VALIDATION_FAILED", stage: "TRAJECTORIES", message: "Trajectory intersects a static occupied trap", atomIds: [trajectory.atomId], configured: staticClearance });
      }
    }
  }
  const atomsById = new Map(input.initialAtoms.map((atom) => [atom.atomId, atom]));
  for (let first = 0; first < trajectories.length; first += 1) {
    for (let second = first + 1; second < trajectories.length; second += 1) {
      const a = trajectories[first]!;
      const b = trajectories[second]!;
      const atomA = atomsById.get(a.atomId);
      const atomB = atomsById.get(b.atomId);
      const safe = atomA && atomB
        ? input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * (atomA.localizationSigmaUm + atomB.localizationSigmaUm) + input.plannerConfig.geometricMarginUm
        : input.plannerConfig.minimumSeparationUm + input.plannerConfig.geometricMarginUm;
      const breakpoints = new Set<number>([0, maxTime]);
      a.waypoints.forEach((waypoint) => breakpoints.add(waypoint.arrivalTimeUs));
      b.waypoints.forEach((waypoint) => breakpoints.add(waypoint.arrivalTimeUs));
      const sorted = [...breakpoints].sort((x, y) => x - y);
      for (let interval = 0; interval + 1 < sorted.length; interval += 1) {
        const startTime = sorted[interval]!;
        const endTime = sorted[interval + 1]!;
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
    frameCount: 0,
  };
}

function validatePairInterval(
  first: AtomTrajectory,
  second: AtomTrajectory,
  startTime: number,
  endTime: number,
  safe: number,
  maxDepth: number,
): { accepted: boolean; minimum: number } {
  let minimum = Number.POSITIVE_INFINITY;
  const firstSpeed = maximumTrajectorySpeed(first);
  const secondSpeed = maximumTrajectorySpeed(second);
  const relativeSpeed = firstSpeed + secondSpeed;
  const visit = (left: number, right: number, depth: number): boolean => {
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

function maximumTrajectorySpeed(trajectory: AtomTrajectory): number {
  let maximum = 0;
  for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
    const start = trajectory.waypoints[index]!;
    const end = trajectory.waypoints[index + 1]!;
    const duration = end.arrivalTimeUs - start.arrivalTimeUs;
    if (duration > 0) maximum = Math.max(maximum, distance(start, end) * 1.875 / duration);
  }
  return maximum;
}

export function targetPhaseForTrap(trapId: number, seed: number): number {
  const hash = Number.parseInt(hashString(`${seed}:${trapId}`), 16) >>> 0;
  return ((hash / 0xffffffff) * Math.PI * 2) - Math.PI;
}

export function trapPhaseMap(input: NormalizedInput, trajectories: AtomTrajectory[], seed: number): Map<number, number> {
  const phases = new Map<number, number>();
  for (const trap of input.staticTraps) phases.set(trap.trapId, targetPhaseForTrap(trap.trapId, seed));
  for (const trajectory of trajectories) phases.set(trajectory.trapId, targetPhaseForTrap(trajectory.trapId, seed));
  return phases;
}

export const generateTrapFrames = sampleTrapFrames;

function findAssignment(input: NormalizedInput, path: PlannedPath): AtomAssignment | undefined {
  if (path.goalSiteId === null) return undefined;
  const targetIndex = input.targetSites.findIndex((target) => target.siteId === path.goalSiteId);
  if (targetIndex < 0) return undefined;
  return {
    atomId: path.atomId,
    sourceIndex: input.initialAtoms.findIndex((atom) => atom.atomId === path.atomId),
    targetSiteId: path.goalSiteId,
    targetIndex,
    disposition: path.disposition,
    assignmentCost: 0,
  };
}

function targetIntensity(input: NormalizedInput, assignment: AtomAssignment | undefined, initial: number): number {
  if (!assignment || assignment.targetIndex === null) return initial;
  return input.targetSites[assignment.targetIndex]?.finalTrapIntensity ?? initial;
}

function pointOf(waypoint: Point2D): Point2D {
  return { xUm: waypoint.xUm, yUm: waypoint.yUm };
}

function roundUp(value: number, period: number): number {
  if (value <= 0) return 0;
  return Math.ceil(value / period - 1e-12) * period;
}

function intensityRampDuration(delta: number, input: NormalizedInput, minimum = 0): number {
  const period = input.motionConfig.framePeriodUs;
  const maximumDelta = input.motionConfig.maxIntensityChangePerFrame;
  const requiredFrames = Number.isFinite(maximumDelta) && maximumDelta > 0
    ? Math.ceil(1.875 * Math.abs(delta) / maximumDelta)
    : 1;
  return Math.max(period, roundUp(minimum, period), requiredFrames * period);
}

function maximumPerFrameDisplacement(length: number, duration: number, period: number): number {
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

function findFirstMovingSegment(trajectory: AtomTrajectory): number {
  for (let index = 0; index + 1 < trajectory.waypoints.length; index += 1) {
    if (distance(trajectory.waypoints[index]!, trajectory.waypoints[index + 1]!) > EPSILON) return index;
  }
  return -1;
}

function firstMovingStartTime(trajectory: AtomTrajectory): number {
  const index = findFirstMovingSegment(trajectory);
  return index < 0 ? trajectory.startTimeUs : trajectory.waypoints[index]!.arrivalTimeUs;
}

function findLastMovingSegment(trajectory: AtomTrajectory): number {
  for (let index = trajectory.waypoints.length - 2; index >= 0; index -= 1) {
    if (distance(trajectory.waypoints[index]!, trajectory.waypoints[index + 1]!) > EPSILON) return index;
  }
  return -1;
}

function lastMovingEndTime(trajectory: AtomTrajectory): number {
  const index = findLastMovingSegment(trajectory);
  return index < 0 ? trajectory.endTimeUs : trajectory.waypoints[index + 1]!.arrivalTimeUs;
}

function trajectoriesFromFrames(frames: TrapFrame[], input?: NormalizedInput): AtomTrajectory[] {
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
    endTimeUs: frames.at(-1)!.timeUs,
    moving: true,
    staticTrap: input?.staticTraps.some((candidate) => candidate.trapId === trap.trapId) ?? false,
  }));
}
