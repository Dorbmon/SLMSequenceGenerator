import { SlmCompilationError, SlmError, checkAbort } from "./errors.js";
import { assignAtoms, assignmentCost } from "./assignment.js";
import { MemoryFrameStore } from "./frame-store.js";
import {
  frameDescriptor,
  SequentialWgsSolver,
  WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN,
  WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN,
  WGS_SOFT_PHASE_PRECOMPENSATION_GAIN,
} from "./hologram.js";
import { calibrationHash, inputHash, normalizeAndValidate } from "./normalization.js";
import { planMotion } from "./planning.js";
import { sampleTrajectory, sampleTrajectoryIntensity, sampleTrapFrames, validateContinuousTrajectories, validateTrapFrames } from "./trajectory.js";
import type {
  AtomAssignment,
  AtomTrajectory,
  CompileOptions,
  CompileProgress,
  CompiledSequenceHandle,
  CompilerConfig,
  FrameMetrics,
  MotionPlan,
  NormalizedInput,
  RearrangementRequest,
  SequenceManifest,
  SequenceValidationReport,
  SequentialHologramBackend,
  SlmFrameDescriptor,
  TrapFrame,
} from "./types.js";
import { hashValue } from "./util.js";
import { WASM_CORE_BUILD_ID } from "./wasm-core.js";

export class SlmSequenceCompiler {
  private disposed = false;
  private cancelled = false;
  private readonly config: CompilerConfig;

  constructor(config: CompilerConfig = {}) {
    this.config = { ...config };
  }

  static async create(config: CompilerConfig = {}): Promise<SlmSequenceCompiler> {
    return new SlmSequenceCompiler(config);
  }

  async compileRearrangement(
    request: RearrangementRequest,
    options: CompileOptions = {},
  ): Promise<CompiledSequenceHandle> {
    this.assertUsable();
    const progress = (value: CompileProgress) => options.onProgress?.(value);
    this.checkAbort(options.signal);
    progress({ stage: "VALIDATING", completed: 0, total: 1, message: "Normalizing inputs and calibration" });
    const normalized = normalizeAndValidate(request, this.config);
    progress({ stage: "VALIDATING", completed: 1, total: 1 });
    this.checkAbort(options.signal);

    progress({ stage: "ASSIGNING", completed: 0, total: 1 });
    const conflictPenalties = new Map<string, number>();
    let assignments: AtomAssignment[] = [];
    let motionPlan: MotionPlan | undefined;
    let assignmentAttempts = 0;
    let lastPlanningError: unknown;
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
        if (!(error instanceof SlmError) || (error.code !== "PATH_NOT_FOUND" && error.code !== "COLLISION_VALIDATION_FAILED") || assignmentAttempts >= maximumAttempts) {
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

    const trapFrameStore = new MemoryFrameStore<TrapFrame>();
    const slmFrameStore = options.outputStore ?? new MemoryFrameStore<Uint8Array | Uint16Array>();
    const frameMetrics: FrameMetrics[] = [];
    const slmFrameDescriptors: SlmFrameDescriptor[] = [];
    const solver: SequentialHologramBackend = await (options.hologramSolverFactory
      ? options.hologramSolverFactory(normalized.calibration, normalized.hologramConfig)
      : new SequentialWgsSolver(normalized.calibration, normalized.hologramConfig));
    const wgsBackend = solver.backendId;
    let byteOffset = 0n;
    let insertedFrames = 0;
    progress({ stage: "SOLVING_SLM_FRAMES", completed: 0, total: trapFrames.length });
    try {
      for (let index = 0; index < trapFrames.length; index += 1) {
        this.checkAbort(options.signal);
        const frame = trapFrames[index]!;
        let solve = await solver.solveSequentialFrame(frame);
        let retry = 0;
        // Quality failures may be retried with a larger WGS budget, while the
        // sequential phase/weight state remains anchored to the prior frame.
        while (!solve.metrics.accepted && retry < 2) {
          retry += 1;
          await solver.rollbackToPreviousAcceptedFrame();
          const budget = Math.min(
            normalized.hologramConfig.maxIterations,
            Math.max(solve.metrics.iterations + 1, solve.metrics.iterations * 2),
          );
          solve = await solver.solveSequentialFrame(frame, budget);
        }
        if (!solve.metrics.accepted) {
          if (index > 0 && insertedFrames < normalized.hologramConfig.maxInsertedFrames && trapFrames[index]!.timeUs - trapFrames[index - 1]!.timeUs > 1) {
            const currentTime = trapFrames[index]!.timeUs;
            const originalMidpointTime = (trapFrames[index - 1]!.timeUs + trapFrames[index]!.timeUs) / 2;
            const midpoint = midpointTrapFrame(trapFrames[index - 1]!, trapFrames[index]!, index, finalMotionPlan.trajectories, normalized, originalMidpointTime);
            midpoint.timeUs = trapFrames[index - 1]!.timeUs + normalized.motionConfig.framePeriodUs;
            insertTrajectoryMidpoint(finalMotionPlan.trajectories, midpoint, currentTime, normalized.motionConfig.framePeriodUs);
            finalMotionPlan.makespanUs += normalized.motionConfig.framePeriodUs;
            const regenerated = sampleTrapFrames(normalized, finalMotionPlan.trajectories);
            trapFrames.splice(index, trapFrames.length - index, ...regenerated.slice(index));
            insertedFrames += 1;
            await solver.rollbackToPreviousAcceptedFrame();
            index -= 1;
            continue;
          }
          throw compilationFailure("FRAME_QUALITY_REJECTED", `SLM frame ${index} failed quality gates`, [
            {
              code: "FRAME_QUALITY_REJECTED",
              stage: "SOLVING_SLM_FRAMES",
              message: `Frame ${index} did not pass configured quality gates`,
              frameIndex: index,
            },
          ]);
        }
        try {
          await trapFrameStore.append(frame);
          await slmFrameStore.append(solve.pixels);
        } catch (error) {
          throw new SlmError("STORAGE_ERROR", `Unable to store SLM frame ${index}`, {
            stage: "WRITING_OUTPUT",
            retryable: true,
            cause: error,
          });
        }
        await solver.commitFrameState();
        const descriptor = frameDescriptor(
          frame,
          solve.pixels,
          normalized.calibration.manifest.activeWidth,
          normalized.calibration.manifest.activeHeight,
          normalized.hologramConfig.format,
          byteOffset,
        );
        byteOffset += BigInt(descriptor.byteLength);
        slmFrameDescriptors.push(descriptor);
        frameMetrics.push({ ...solve.metrics, refinementCount: retry });
        progress({ stage: "SOLVING_SLM_FRAMES", completed: index + 1, total: trapFrames.length, frameIndex: index });
      }
    } finally {
      await solver.dispose?.();
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
      trapValidation,
    );
    if (!validation.accepted) {
      throw compilationFailure("FRAME_QUALITY_REJECTED", "Complete sequence validation failed", validation.errors);
    }
    progress({ stage: "VALIDATING_SEQUENCE", completed: 1, total: 1 });
    progress({ stage: "WRITING_OUTPUT", completed: 0, total: 1 });
    const manifest = makeManifest(request, normalized, assignments, finalMotionPlan, trapFrames, slmFrameDescriptors, validation, this.config, assignmentAttempts, wgsBackend);
    progress({ stage: "WRITING_OUTPUT", completed: 1, total: 1 });
    return {
      manifest,
      assignment: assignments,
      trajectories: finalMotionPlan.trajectories,
      trapFrameStore,
      slmFrameStore,
      slmFrameDescriptors,
      frameMetrics,
      validation,
    };
  }

  async planOnly(request: RearrangementRequest): Promise<MotionPlan> {
    this.assertUsable();
    const normalized = normalizeAndValidate(request, this.config);
    const penalties = new Map<string, number>();
    let lastError: unknown;
    for (let attempt = 0; attempt <= normalized.assignmentConfig.maxAssignmentRetries; attempt += 1) {
      const assignments = assignAtoms(normalized, penalties);
      try {
        return planMotion(normalized, assignments);
      } catch (error) {
        lastError = error;
        if (!(error instanceof SlmError) || (error.code !== "PATH_NOT_FOUND" && error.code !== "COLLISION_VALIDATION_FAILED") || attempt === normalized.assignmentConfig.maxAssignmentRetries) throw error;
        for (const assignment of assignments) if (assignment.targetSiteId !== null) {
          const key = `${assignment.atomId}:${assignment.targetSiteId}`;
          penalties.set(key, (penalties.get(key) ?? 0) + normalized.assignmentConfig.conflictPenalty);
        }
      }
    }
    throw lastError;
  }

  async solveTrapFrames(frames: AsyncIterable<TrapFrame> | Iterable<TrapFrame>): Promise<CompiledSequenceHandle> {
    this.assertUsable();
    const calibration = this.config.calibration ?? Object.values(this.config.calibrations ?? {})[0];
    if (!calibration) throw new SlmError("CALIBRATION_MISMATCH", "solveTrapFrames requires a calibration package", { stage: "VALIDATING" });
    const normalized = normalizeAndValidate({
      initialAtoms: [],
      targetSites: [],
      calibration,
    }, this.config);
    const trapFrameStore = new MemoryFrameStore<TrapFrame>();
    const slmFrameStore = new MemoryFrameStore<Uint8Array | Uint16Array>();
    const frameMetrics: FrameMetrics[] = [];
    const descriptors: SlmFrameDescriptor[] = [];
    const solver = new SequentialWgsSolver(calibration, normalized.hologramConfig);
    let byteOffset = 0n;
    for await (const frame of frames) {
      const result = solver.solveSequentialFrame(frame);
      if (!result.metrics.accepted) throw new SlmError("FRAME_QUALITY_REJECTED", `Frame ${frame.frameIndex} failed quality gates`, { stage: "SOLVING_SLM_FRAMES", retryable: true });
      trapFrameStore.append(frame);
      slmFrameStore.append(result.pixels);
      solver.commitFrameState();
      const descriptor = frameDescriptor(frame, result.pixels, calibration.manifest.activeWidth, calibration.manifest.activeHeight, normalized.hologramConfig.format, byteOffset);
      byteOffset += BigInt(descriptor.byteLength);
      descriptors.push(descriptor);
      frameMetrics.push(result.metrics);
    }
    const validation = validateTrapFrames(trapFrameStore.toArray(), normalized);
    if (!validation.accepted) throw compilationFailure("COLLISION_VALIDATION_FAILED", "Trap frames failed validation", validation.errors);
    const manifest: SequenceManifest = {
      formatVersion: "1.1",
      creationTimestamp: this.config.creationTimestamp ?? "1970-01-01T00:00:00.000Z",
      compilerVersion: this.config.compilerVersion ?? "0.1.0",
      wasmBuildId: this.config.wasmBuildId ?? WASM_CORE_BUILD_ID,
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
      wgsBackend: solver.backendId,
      wgsParameters: {
        targetPhaseMode: normalized.hologramConfig.targetPhaseMode,
        firstFrameIterations: normalized.hologramConfig.firstFrameIterations,
        subsequentFrameIterations: normalized.hologramConfig.subsequentFrameIterations,
        fftWidth: normalized.hologramConfig.width,
        fftHeight: normalized.hologramConfig.height,
        gamma: normalized.hologramConfig.gamma,
        effectiveAmplitudeFeedbackGain: Math.min(
          normalized.hologramConfig.gamma,
          WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN,
        ),
        phasePrecompensationGain: normalized.hologramConfig.targetPhaseMode === "SOFT_PHASE_LOCKED_WGS"
          ? WGS_SOFT_PHASE_PRECOMPENSATION_GAIN
          : normalized.hologramConfig.targetPhaseMode === "REFERENCE_WGS"
            ? 0
            : WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN,
        retainBestQuantizedCandidate: true,
        convergenceTolerance: normalized.hologramConfig.convergenceTolerance,
        epsilon: normalized.hologramConfig.epsilon,
      },
      outputWidth: calibration.manifest.activeWidth,
      outputHeight: calibration.manifest.activeHeight,
      pixelFormat: normalized.hologramConfig.format,
      deterministicSeed: normalized.hologramConfig.deterministicSeed,
      checksums: {
        frameDescriptors: hashValue(descriptors),
        slmFrames: hashValue(descriptors.map((descriptor) => descriptor.crc32)),
        trapFrames: hashValue(trapFrameStore.toArray()),
      },
      validationStatus: "accepted",
    };
    return { manifest, assignment: [], trajectories: [], trapFrameStore, slmFrameStore, slmFrameDescriptors: descriptors, frameMetrics, validation };
  }

  dispose(): void {
    this.disposed = true;
  }

  reset(): void {
    if (this.disposed) throw new SlmError("INVALID_ARGUMENT", "Compiler has been disposed", { stage: "CREATED" });
    this.cancelled = false;
  }

  private assertUsable(): void {
    if (this.disposed) throw new SlmError("INVALID_ARGUMENT", "Compiler has been disposed", { stage: "CREATED" });
    if (this.cancelled) throw new SlmError("INVALID_ARGUMENT", "Compiler was cancelled; call reset before reuse", { stage: "CANCELLED" });
  }

  private checkAbort(signal?: AbortSignal): void {
    try {
      checkAbort(signal);
    } catch (error) {
      if (error instanceof SlmError && error.code === "CANCELLED") this.cancelled = true;
      throw error;
    }
  }
}

function validateCompleteSequence(
  assignments: AtomAssignment[],
  motionPlan: MotionPlan,
  trapFrames: TrapFrame[],
  descriptors: SlmFrameDescriptor[],
  metrics: FrameMetrics[],
  input: NormalizedInput,
  trapValidation: SequenceValidationReport,
): SequenceValidationReport {
  const errors = [...trapValidation.errors];
  const warnings = [...trapValidation.warnings];
  const continuous = validateContinuousTrajectories(motionPlan.trajectories, input);
  errors.push(...continuous.errors);
  warnings.push(...continuous.warnings);
  const requiredTargets = input.targetSites.filter((target) => target.required);
  const assignedTargetIds = assignments.filter((assignment) => assignment.targetSiteId !== null).map((assignment) => assignment.targetSiteId!);
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
    const final = trajectory.waypoints.at(-1)!;
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
    else if (!metric.converged) warnings.push({
      code: "WGS_NOT_CONVERGED",
      stage: "VALIDATING_SEQUENCE",
      message: `Frame ${metric.frameIndex} retained its best WGS candidate but did not meet ` +
        `the amplitude gate (${metric.maximumRelativeAmplitudeError} <= ${metric.amplitudeConvergenceTolerance}) ` +
        `and phase gate (${metric.maximumTargetPhaseErrorRad} rad <= ${metric.phaseConvergenceToleranceRad} rad)`,
      frameIndex: metric.frameIndex,
    });
  }
  return {
    accepted: errors.length === 0,
    errors,
    warnings,
    minimumAtomSeparationUm: Math.min(trapValidation.minimumAtomSeparationUm, continuous.minimumAtomSeparationUm, motionPlan.minimumValidatedSeparationUm),
    maximumSpeedUmPerUs: continuous.maximumSpeedUmPerUs,
    maximumAccelerationUmPerUs2: continuous.maximumAccelerationUmPerUs2,
    maximumJerkUmPerUs3: continuous.maximumJerkUmPerUs3,
    frameCount: trapFrames.length,
  };
}

function makeManifest(
  request: RearrangementRequest,
  normalized: NormalizedInput,
  assignments: AtomAssignment[],
  motionPlan: MotionPlan,
  trapFrames: TrapFrame[],
  descriptors: SlmFrameDescriptor[],
  validation: SequenceValidationReport,
  compilerConfig: CompilerConfig,
  assignmentAttempts: number,
  wgsBackend: string,
): SequenceManifest {
  return {
    formatVersion: "1.1",
    creationTimestamp: compilerConfig.creationTimestamp ?? "1970-01-01T00:00:00.000Z",
    compilerVersion: compilerConfig.compilerVersion ?? "0.1.0",
    wasmBuildId: compilerConfig.wasmBuildId ?? WASM_CORE_BUILD_ID,
    inputHash: inputHash(request, normalized),
    calibrationId: normalized.calibration.manifest.calibrationId,
    calibrationHash: calibrationHash(normalized.calibration),
    coordinateConvention: normalized.calibration.manifest.coordinateConvention ?? "+x right, +y up",
    atomCount: normalized.initialAtoms.length,
    targetCount: normalized.targetSites.length,
    trapCount: (motionPlan.trajectories.length + normalized.staticTraps.length),
    frameCount: trapFrames.length,
    framePeriodUs: normalized.motionConfig.framePeriodUs,
    assignmentCost: assignmentCost(assignments),
    assignmentAttempts,
    plannerBackend: motionPlan.conflictComponentCount > 0 ? "direct+cbs-reference" : "direct-reference",
    plannerParameters: {
      minimumSeparationUm: normalized.plannerConfig.minimumSeparationUm,
      gridResolutionUm: normalized.plannerConfig.gridResolutionUm,
      waitCount: motionPlan.waitCount,
      detourCount: motionPlan.detourCount,
    },
    wgsBackend,
    wgsParameters: {
      targetPhaseMode: normalized.hologramConfig.targetPhaseMode,
      firstFrameIterations: normalized.hologramConfig.firstFrameIterations,
      subsequentFrameIterations: normalized.hologramConfig.subsequentFrameIterations,
      fftWidth: normalized.hologramConfig.width,
      fftHeight: normalized.hologramConfig.height,
      gamma: normalized.hologramConfig.gamma,
      effectiveAmplitudeFeedbackGain: Math.min(
        normalized.hologramConfig.gamma,
        WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN,
      ),
      phasePrecompensationGain: normalized.hologramConfig.targetPhaseMode === "SOFT_PHASE_LOCKED_WGS"
        ? WGS_SOFT_PHASE_PRECOMPENSATION_GAIN
        : normalized.hologramConfig.targetPhaseMode === "REFERENCE_WGS"
          ? 0
          : WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN,
      retainBestQuantizedCandidate: true,
      convergenceTolerance: normalized.hologramConfig.convergenceTolerance,
      epsilon: normalized.hologramConfig.epsilon,
    },
    outputWidth: normalized.calibration.manifest.activeWidth,
    outputHeight: normalized.calibration.manifest.activeHeight,
    pixelFormat: normalized.hologramConfig.format,
    deterministicSeed: normalized.hologramConfig.deterministicSeed,
    checksums: {
      frameDescriptors: hashValue(descriptors),
      slmFrames: hashValue(descriptors.map((descriptor) => descriptor.crc32)),
      trapFrames: hashValue(trapFrames),
      validation: hashValue(validation),
    },
    validationStatus: validation.accepted ? "accepted" : "rejected",
  };
}

function compilationFailure(code: ConstructorParameters<typeof SlmError>[0], message: string, issues: SequenceValidationReport["errors"]): SlmCompilationError {
  return new SlmCompilationError(code, message, issues, { stage: "VALIDATING_SEQUENCE", retryable: true });
}

function midpointTrapFrame(
  previous: TrapFrame,
  current: TrapFrame,
  frameIndex: number,
  trajectories: AtomTrajectory[],
  input: NormalizedInput,
  originalMidpointTime: number,
): TrapFrame {
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
        flags: (trajectory.moving ? 1 : 0) | (intensity <= 1e-9 ? 2 : 0),
      };
    }
    return {
      trapId: first.trapId,
      atomId: first.atomId ?? second.atomId,
      xUm: (first.xUm + second.xUm) / 2,
      yUm: (first.yUm + second.yUm) / 2,
      intensity: (first.intensity + second.intensity) / 2,
      targetPhaseRad: (first.targetPhaseRad + second.targetPhaseRad) / 2,
      flags: (distanceSquared(first, second) > 1e-18 ? 1 : 0) |
        ((first.atomId === null && second.atomId === null && first.intensity <= 0 && second.intensity <= 0) ? 2 : 0) |
        (first.flags & second.flags & 4),
    };
  });
  return {
    frameIndex,
    timeUs: Math.floor((previous.timeUs + current.timeUs) / 2),
    traps,
  };
}

function distanceSquared(first: { xUm: number; yUm: number }, second: { xUm: number; yUm: number }): number {
  return (first.xUm - second.xUm) ** 2 + (first.yUm - second.yUm) ** 2;
}

function insertTrajectoryMidpoint(
  trajectories: AtomTrajectory[],
  midpoint: TrapFrame,
  originalCurrentTime: number,
  period: number,
): void {
  for (const trajectory of trajectories) {
    const state = midpoint.traps.find((trap) => trap.trapId === trajectory.trapId);
    if (!state) continue;
    const shifted = trajectory.waypoints.map((waypoint) => ({
      ...waypoint,
      arrivalTimeUs: waypoint.arrivalTimeUs >= originalCurrentTime ? waypoint.arrivalTimeUs + period : waypoint.arrivalTimeUs,
    }));
    const inserted = {
      xUm: state.xUm,
      yUm: state.yUm,
      arrivalTimeUs: midpoint.timeUs,
    };
    const position = shifted.findIndex((waypoint) => waypoint.arrivalTimeUs >= inserted.arrivalTimeUs);
    shifted.splice(position < 0 ? shifted.length : position, 0, inserted);
    trajectory.waypoints = shifted;
    trajectory.endTimeUs = Math.max(trajectory.endTimeUs + (trajectory.endTimeUs >= originalCurrentTime ? period : 0), inserted.arrivalTimeUs);
  }
}
