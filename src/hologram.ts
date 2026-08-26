import { SlmError } from "./errors.js";
import {
  createComplexField,
  fft2d,
  type ComplexField,
} from "./fft.js";
import { mapPhysicalPointToDftFrequency, periodicFftDistance } from "./coordinates.js";
import type {
  CalibrationPackage,
  FrameMetrics,
  HologramConfig,
  HologramSolveResult,
  NormalizedInput,
  SlmFrameDescriptor,
  TrapFrame,
  TrapState,
} from "./types.js";
import { angularDistance, clamp, crc32, TAU, wrapPhase } from "./util.js";
import { wasmNudftSampleTargets, wasmNudftSynthesizePhase } from "./wasm-core.js";

// A coherent sum below this fraction has no numerically meaningful phase.
// Both CPU and GPU use the same seeded fallback so float precision cannot send
// the non-convex solver into unrelated solutions at destructive cancellations.
export const WGS_INITIALIZATION_CANCELLATION_RATIO = 1e-3;

// Trap coefficients are strongly coupled after the phase-only projection.
// A large multiplicative WGS step therefore creates a two-cycle even when the
// configured gamma is otherwise reasonable for a full-plane GS solve.
export const WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN = 0.1;
export const WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN = 0.7;
export const WGS_SOFT_PHASE_PRECOMPENSATION_GAIN = 0.2;

interface SolverState {
  phase: Float64Array;
  weights: Map<number, number>;
  targetPhases: Map<number, number>;
  synthesisPhases: Map<number, number>;
  measuredPhases: Map<number, number>;
  measuredIntensities: Map<number, number>;
  codes: Uint8Array | Uint16Array;
  frameIndex: number;
}

interface QuantizedCandidate {
  phase: Float64Array;
  weights: Map<number, number>;
  synthesisPhases: Map<number, number>;
  codes: Uint8Array | Uint16Array;
  measured: { real: number; imag: number }[];
  amplitudes: number[];
  relativeAmplitudeError: number;
  phaseError: number;
  certificateScore: number;
}

export class SequentialWgsSolver {
  readonly backendId = "wasm-exact-nudft-phase-locked-wgs";
  readonly width: number;
  readonly height: number;
  private readonly calibration: CalibrationPackage;
  private readonly config: Required<HologramConfig>;
  private accepted: SolverState | undefined;
  private candidate: SolverState | undefined;

  constructor(calibration: CalibrationPackage, config: HologramConfig | Required<HologramConfig> = {}) {
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
      requireConvergence: config.requireConvergence ?? false,
    };
    this.width = this.config.width;
    this.height = this.config.height;
    this.beginSequence();
  }

  beginSequence(): void {
    this.accepted = undefined;
    this.candidate = undefined;
  }

  solveSequentialFrame(frame: TrapFrame, iterationBudget?: number): HologramSolveResult {
    if (frame.traps.some((trap) => !Number.isFinite(trap.xUm) || !Number.isFinite(trap.yUm) || !Number.isFinite(trap.intensity) || !Number.isFinite(trap.targetPhaseRad))) {
      throw new SlmError("NUMERIC_ERROR", "Trap frame contains a non-finite value", { stage: "SOLVING_SLM_FRAMES" });
    }
    const started = this.config.measureSolveTime ? nowMs() : 0;
    const mappedTargets = frame.traps.map((trap) => this.mapCoordinate(trap));
    const targetX = Float64Array.from(mappedTargets, (target) => target.x);
    const targetY = Float64Array.from(mappedTargets, (target) => target.y);
    const previous = this.accepted;
    const targetPhases = new Map<number, number>();
    const synthesisPhases = new Map<number, number>();
    const weights = new Map<number, number>();
    for (const trap of frame.traps) {
      const previousTargetPhase = previous?.targetPhases.get(trap.trapId);
      const requestedTargetPhase = this.config.targetPhaseMode === "PHASE_INTERPOLATED_WGS"
        ? trap.targetPhaseRad
        : previousTargetPhase ?? trap.targetPhaseRad;
      targetPhases.set(trap.trapId, requestedTargetPhase);
      const previousSynthesisPhase = previous?.synthesisPhases.get(trap.trapId);
      synthesisPhases.set(
        trap.trapId,
        previousSynthesisPhase === undefined
          ? requestedTargetPhase
          : wrapPhase(previousSynthesisPhase + wrapPhase(requestedTargetPhase - (previousTargetPhase ?? requestedTargetPhase))),
      );
      weights.set(trap.trapId, previous?.weights.get(trap.trapId) ?? 1);
    }
    let phase = previous
      ? new Float64Array(previous.phase)
      : this.initializeSuperposition(frame, targetX, targetY);
    const iterations = Math.min(
      Math.max(1, iterationBudget ?? (previous ? this.config.subsequentFrameIterations : this.config.firstFrameIterations)),
      this.config.maxIterations,
    );
    const desired = frame.traps.map((trap) => Math.sqrt(Math.max(0, trap.intensity)));
    const amplitudeTolerance = this.config.convergenceTolerance;
    const phaseTolerance = phaseConvergenceTolerance(this.config);
    const amplitudeGain = Math.min(this.config.gamma, WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN);
    let best: QuantizedCandidate | undefined;
    let performedIterations = 0;
    let evaluated = this.evaluateQuantizedPhase(phase, targetX, targetY);
    const considerCandidate = (): void => {
      const scale = fitAmplitudeScale(evaluated.amplitudes, desired, this.config.epsilon);
      const relativeAmplitudeError = maximumRelativeAmplitudeError(
        evaluated.amplitudes,
        desired,
        scale,
        this.config.epsilon,
      );
      const phaseError = maximumTargetPhaseError(
        frame.traps,
        evaluated.measured,
        targetPhases,
        this.config.targetPhaseMode,
      );
      const score = convergenceCertificateScore(
        relativeAmplitudeError,
        phaseError,
        amplitudeTolerance,
        phaseTolerance,
        this.config.targetPhaseMode,
      );
      if (best && score >= best.certificateScore) return;
      best = {
        phase: new Float64Array(phase),
        weights: new Map(weights),
        synthesisPhases: new Map(synthesisPhases),
        codes: cloneCodes(evaluated.codes),
        measured: evaluated.measured.map((value) => ({ ...value })),
        amplitudes: [...evaluated.amplitudes],
        relativeAmplitudeError,
        phaseError,
        certificateScore: score,
      };
    };
    considerCandidate();

    for (let iteration = 0; iteration < iterations && best!.certificateScore > 1; iteration += 1) {
      performedIterations += 1;
      const scale = fitAmplitudeScale(evaluated.amplitudes, desired, this.config.epsilon);
      for (let index = 0; index < frame.traps.length; index += 1) {
        const trap = frame.traps[index]!;
        if (desired[index]! <= this.config.epsilon) continue;
        const oldWeight = weights.get(trap.trapId) ?? 1;
        const ratio = scale * desired[index]! / (evaluated.amplitudes[index]! + this.config.epsilon);
        weights.set(trap.trapId, clamp(oldWeight * ratio ** amplitudeGain, this.config.minWeight, this.config.maxWeight));
      }
      normalizeWeights(weights);

      if (this.config.targetPhaseMode === "REFERENCE_WGS") {
        frame.traps.forEach((trap, index) => {
          synthesisPhases.set(trap.trapId, Math.atan2(evaluated.measured[index]!.imag, evaluated.measured[index]!.real));
        });
      } else if (iteration === 0 && maximumTargetPhaseError(
        frame.traps,
        evaluated.measured,
        targetPhases,
        this.config.targetPhaseMode,
      ) > phaseTolerance) {
        const phaseGain = this.config.targetPhaseMode === "SOFT_PHASE_LOCKED_WGS"
          ? WGS_SOFT_PHASE_PRECOMPENSATION_GAIN
          : WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN;
        frame.traps.forEach((trap, index) => {
          if (desired[index]! <= this.config.epsilon) return;
          const measuredPhase = Math.atan2(evaluated.measured[index]!.imag, evaluated.measured[index]!.real);
          const requestedPhase = targetPhases.get(trap.trapId) ?? trap.targetPhaseRad;
          const synthesisPhase = synthesisPhases.get(trap.trapId) ?? requestedPhase;
          synthesisPhases.set(trap.trapId, wrapPhase(
            synthesisPhase + phaseGain * wrapPhase(requestedPhase - measuredPhase),
          ));
        });
      }

      const synthesizedAmplitudes = new Float64Array(frame.traps.length);
      const synthesizedPhases = new Float64Array(frame.traps.length);
      for (let index = 0; index < frame.traps.length; index += 1) {
        const trap = frame.traps[index]!;
        synthesizedAmplitudes[index] = (weights.get(trap.trapId) ?? 1) * desired[index]!;
        synthesizedPhases[index] = synthesisPhases.get(trap.trapId) ?? targetPhases.get(trap.trapId) ?? trap.targetPhaseRad;
      }
      wasmNudftSynthesizePhase(
        targetX,
        targetY,
        synthesizedAmplitudes,
        synthesizedPhases,
        this.width,
        this.height,
        phase,
        this.config.epsilon,
        this.config.deterministicSeed,
        false,
      );
      evaluated = this.evaluateQuantizedPhase(phase, targetX, targetY);
      considerCandidate();
    }

    const selected = best!;
    phase = selected.phase;
    const finalWeights = selected.weights;
    const finalSynthesisPhases = selected.synthesisPhases;
    const codes = selected.codes;
    const finalDisplayPhase = this.composeDisplayPhase(phase);
    const finalSpatial = this.spatialField(this.decodeCodes(codes, finalDisplayPhase));
    const finalMeasured = selected.measured;
    const finalAmplitudes = selected.amplitudes;
    const measuredPhases = new Map<number, number>();
    const measuredIntensities = new Map<number, number>();
    frame.traps.forEach((trap, index) => {
      measuredPhases.set(trap.trapId, Math.atan2(finalMeasured[index]!.imag, finalMeasured[index]!.real));
      measuredIntensities.set(trap.trapId, finalAmplitudes[index]! ** 2);
    });
    const maxRelativeError = selected.relativeAmplitudeError;
    const converged = selected.certificateScore <= 1;
    const finalForward = cloneField(finalSpatial);
    fft2d(finalForward, false);
    const metrics = this.evaluateMetrics(
      frame,
      finalForward,
      finalMeasured,
      finalAmplitudes,
      codes,
      previous,
      Math.max(1, performedIterations),
      converged,
      maxRelativeError,
      started,
      targetPhases,
      measuredPhases,
      measuredIntensities,
      finalWeights,
    );
    this.candidate = {
      phase,
      weights: new Map(finalWeights),
      targetPhases: new Map(targetPhases),
      synthesisPhases: new Map(finalSynthesisPhases),
      measuredPhases,
      measuredIntensities,
      codes,
      frameIndex: frame.frameIndex,
    };
    metrics.accepted = passesQualityGates(metrics, this.config.qualityGates) &&
      (!this.config.requireConvergence || metrics.converged);
    return { pixels: this.extractActiveCodes(codes), metrics };
  }

  commitFrameState(): void {
    if (!this.candidate) throw new SlmError("INVALID_ARGUMENT", "No hologram candidate is available to commit", { stage: "SOLVING_SLM_FRAMES" });
    this.accepted = {
      phase: new Float64Array(this.candidate.phase),
      weights: new Map(this.candidate.weights),
      targetPhases: new Map(this.candidate.targetPhases),
      synthesisPhases: new Map(this.candidate.synthesisPhases),
      measuredPhases: new Map(this.candidate.measuredPhases),
      measuredIntensities: new Map(this.candidate.measuredIntensities),
      codes: new (this.candidate.codes.constructor as { new (source: Uint8Array | Uint16Array): Uint8Array | Uint16Array })(this.candidate.codes),
      frameIndex: this.candidate.frameIndex,
    };
    this.candidate = undefined;
  }

  rollbackToPreviousAcceptedFrame(): void {
    this.candidate = undefined;
  }

  get acceptedFrameIndex(): number | undefined {
    return this.accepted?.frameIndex;
  }

  private initializeSuperposition(
    frame: TrapFrame,
    targetX: Float64Array,
    targetY: Float64Array,
  ): Float64Array {
    const phase = new Float64Array(this.width * this.height);
    if (frame.traps.length === 0) return phase;
    const amplitudes = Float64Array.from(frame.traps, (trap) => Math.sqrt(Math.max(0, trap.intensity)));
    const phases = Float64Array.from(frame.traps, (trap) => trap.targetPhaseRad);
    const coherentAmplitude = amplitudes.reduce((sum, amplitude) => sum + amplitude, 0);
    const cancellationThreshold = Math.max(
      this.config.epsilon,
      coherentAmplitude * WGS_INITIALIZATION_CANCELLATION_RATIO,
    );
    wasmNudftSynthesizePhase(
      targetX,
      targetY,
      amplitudes,
      phases,
      this.width,
      this.height,
      phase,
      cancellationThreshold,
      this.config.deterministicSeed,
      true,
    );
    return phase;
  }

  private spatialField(phase: Float64Array): ComplexField {
    const field = createComplexField(this.width, this.height);
    const activePixels = this.calibration.manifest.activeWidth * this.calibration.manifest.activeHeight;
    for (let index = 0; index < phase.length; index += 1) {
      const x = index % this.width;
      const y = Math.floor(index / this.width);
      const amplitude = this.calibrationValueAt(this.calibration.incidentAmplitude, index, x, y, 1, activePixels) *
        this.calibrationValueAt(this.calibration.apertureMask, index, x, y, 1, activePixels);
      field.real[index] = amplitude * Math.cos(phase[index]!);
      field.imag[index] = amplitude * Math.sin(phase[index]!);
    }
    return field;
  }

  private measureTargets(
    field: ComplexField,
    targetX: Float64Array,
    targetY: Float64Array,
  ): { real: number; imag: number }[] {
    const measured = wasmNudftSampleTargets(field.real, field.imag, this.width, this.height, targetX, targetY);
    return Array.from({ length: targetX.length }, (_, index) => ({
      real: measured.real[index]!,
      imag: measured.imag[index]!,
    }));
  }

  /** Evaluate the exact field represented by the exportable display codes. */
  private evaluateQuantizedPhase(
    phase: Float64Array,
    targetX: Float64Array,
    targetY: Float64Array,
  ): {
    codes: Uint8Array | Uint16Array;
    spatial: ComplexField;
    measured: { real: number; imag: number }[];
    amplitudes: number[];
  } {
    const displayPhase = this.composeDisplayPhase(phase);
    const codes = this.quantize(displayPhase);
    const spatial = this.spatialField(this.decodeCodes(codes, displayPhase));
    const measured = this.measureTargets(spatial, targetX, targetY);
    return {
      codes,
      spatial,
      measured,
      amplitudes: measured.map((value) => Math.hypot(value.real, value.imag)),
    };
  }

  private calibrationValueAt(
    value: ArrayLike<number> | number | undefined,
    index: number,
    x: number,
    y: number,
    fallback: number,
    activePixels: number,
  ): number {
    if (value === undefined || typeof value === "number") {
      return this.isActivePixel(x, y) ? (typeof value === "number" ? value : fallback) : 0;
    }
    if (value.length === this.width * this.height) return value[index] ?? fallback;
    if (value.length === activePixels) {
      const { xStart, yStart } = this.activeOrigin();
      if (x < xStart || y < yStart || x >= xStart + this.calibration.manifest.activeWidth || y >= yStart + this.calibration.manifest.activeHeight) {
        return 0;
      }
      const activeIndex = (y - yStart) * this.calibration.manifest.activeWidth + (x - xStart);
      return value[activeIndex] ?? fallback;
    }
    return fallback;
  }

  private activeOrigin(): { xStart: number; yStart: number } {
    return {
      xStart: Math.floor((this.width - this.calibration.manifest.activeWidth) / 2),
      yStart: Math.floor((this.height - this.calibration.manifest.activeHeight) / 2),
    };
  }

  private isActivePixel(x: number, y: number): boolean {
    const { activeWidth, activeHeight } = this.calibration.manifest;
    const { xStart, yStart } = this.activeOrigin();
    return x >= xStart && y >= yStart && x < xStart + activeWidth && y < yStart + activeHeight;
  }

  private extractActiveCodes(codes: Uint8Array | Uint16Array): Uint8Array | Uint16Array {
    const { activeWidth, activeHeight } = this.calibration.manifest;
    const Output = codes.constructor as {
      new(length: number): Uint8Array | Uint16Array;
      new(source: Uint8Array | Uint16Array): Uint8Array | Uint16Array;
    };
    if (activeWidth === this.width && activeHeight === this.height) return new Output(codes);
    const output = new Output(activeWidth * activeHeight);
    const { xStart, yStart } = this.activeOrigin();
    for (let y = 0; y < activeHeight; y += 1) {
      const sourceStart = (yStart + y) * this.width + xStart;
      output.set(codes.subarray(sourceStart, sourceStart + activeWidth), y * activeWidth);
    }
    return output;
  }

  private mapCoordinate(trap: TrapState): { x: number; y: number } {
    return mapPhysicalPointToDftFrequency(trap, this.calibration, this.width, this.height);
  }

  private composeDisplayPhase(phase: Float64Array): Float64Array {
    const display = new Float64Array(phase.length);
    const signs = this.calibration.phaseSigns;
    for (let index = 0; index < phase.length; index += 1) {
      display[index] = wrapPhase(
        phase[index]! +
        (signs?.aberration ?? 1) * calibrationValue(this.calibration.aberrationPhase, index, 0) +
        (signs?.grating ?? 1) * calibrationValue(this.calibration.carrierGrating, index, 0) +
        (signs?.lens ?? 1) * calibrationValue(this.calibration.digitalLens, index, 0),
      );
    }
    return display;
  }

  private quantize(phase: Float64Array): Uint8Array | Uint16Array {
    const maxCode = this.config.format === "UINT8" ? 255 : 65535;
    const result = this.config.format === "UINT8" ? new Uint8Array(phase.length) : new Uint16Array(phase.length);
    for (let index = 0; index < phase.length; index += 1) {
      const target = wrapPhase(phase[index]!);
      let code = inverseLut(target, this.calibration, maxCode);
      code = Math.round(clamp(code, 0, maxCode));
      result[index] = code;
    }
    return result;
  }

  private decodeCodes(codes: Uint8Array | Uint16Array, fallback: Float64Array): Float64Array {
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
            const error = Math.abs(inverse[lutIndex]! - codes[index]!);
            if (error < nearestError) {
              nearestError = error;
              nearest = lutIndex;
            }
          }
          decoded[index] = wrapPhase((nearest / (inverse.length - 1)) * TAU - Math.PI);
        } else {
          decoded[index] = wrapPhase((codes[index]! / maxCode) * TAU - Math.PI);
        }
        continue;
      }
      const position = (codes[index]! / maxCode) * (lut.length - 1);
      const low = Math.floor(position);
      const high = Math.min(lut.length - 1, low + 1);
      const fraction = position - low;
      decoded[index] = wrapPhase((lut[low] ?? fallback[index]!) * (1 - fraction) + (lut[high] ?? fallback[index]!) * fraction);
    }
    return decoded;
  }

  private evaluateMetrics(
    frame: TrapFrame,
    field: ComplexField,
    measured: { real: number; imag: number }[],
    amplitudes: number[],
    codes: Uint8Array | Uint16Array,
    previous: SolverState | undefined,
    iterations: number,
    converged: boolean,
    relativeError: number,
    started: number,
    targetPhases: Map<number, number>,
    measuredPhases: Map<number, number>,
    measuredIntensities: Map<number, number>,
    weights: Map<number, number>,
  ): FrameMetrics {
    const intensities = amplitudes.map((amplitude) => amplitude * amplitude);
    const mean = meanOf(intensities);
    const std = standardDeviation(intensities, mean);
    const totalPower = field.real.reduce((sum, value, index) => sum + value * value + field.imag[index]! * field.imag[index]!, 0);
    const targetPower = intensities.reduce((sum, value) => sum + value, 0);
    const ghost = maximumGhostIntensity(field, frame.traps.map((trap) => this.mapCoordinate(trap)));
    const phaseError = maximumTargetPhaseError(frame.traps, measured, targetPhases, this.config.targetPhaseMode);
    const phaseChange = previous ? maximumMapPhaseChange(previous.measuredPhases, measuredPhases) : 0;
    const codeChange = previous ? maximumCodeChange(previous.codes, codes) : 0;
    const flags: string[] = [];
    if (!converged) flags.push("NOT_CONVERGED");
    if (frame.traps.some((trap) => trap.intensity > this.config.epsilon) && (mean <= this.config.epsilon || targetPower <= this.config.epsilon)) flags.push("ZERO_TARGET_OUTPUT");
    if (!Number.isFinite(relativeError) || !Number.isFinite(totalPower) ||
        ![mean, std, phaseError, phaseChange, codeChange, ...weights.values()].every(Number.isFinite)) flags.push("NUMERIC_ERROR");
    const transitionMinimum = estimateTransitionMinimumIntensity(measuredIntensities, frame, previous, codes, this.config.format);
    return {
      frameIndex: frame.frameIndex,
      timeUs: frame.timeUs,
      iterations,
      converged,
      maximumRelativeAmplitudeError: relativeError,
      amplitudeConvergenceTolerance: this.config.convergenceTolerance,
      phaseConvergenceToleranceRad: phaseConvergenceTolerance(this.config),
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
      flags,
    };
  }
}

export const WgsSolver = SequentialWgsSolver;

export function solveHologramFrame(
  frame: TrapFrame,
  calibration: CalibrationPackage,
  config: HologramConfig | Required<HologramConfig> = {},
): HologramSolveResult {
  const solver = new SequentialWgsSolver(calibration, config);
  const result = solver.solveSequentialFrame(frame);
  if (result.metrics.accepted) solver.commitFrameState();
  return result;
}

function maximumTargetPhaseError(
  traps: TrapState[],
  measured: { real: number; imag: number }[],
  targetPhases: Map<number, number>,
  mode: HologramConfig["targetPhaseMode"],
): number {
  if (traps.length === 0 || mode === "REFERENCE_WGS") return 0;
  return Math.max(...traps.map((trap, index) => angularDistance(Math.atan2(measured[index]!.imag, measured[index]!.real), targetPhases.get(trap.trapId) ?? trap.targetPhaseRad)));
}

function maximumRelativeAmplitudeError(amplitudes: number[], desired: number[], scale: number, epsilon: number): number {
  if (desired.length === 0) return 0;
  return Math.max(...desired.map((value, index) => Math.abs(amplitudes[index]! - scale * value) / (Math.abs(scale * value) + epsilon)));
}

function phaseConvergenceTolerance(config: Required<HologramConfig>): number {
  const codeCount = config.format === "UINT8" ? 256 : 65536;
  return Math.max(config.convergenceTolerance, 1e-3, TAU / codeCount * 1.5);
}

function convergenceCertificateScore(
  relativeAmplitudeError: number,
  phaseError: number,
  amplitudeTolerance: number,
  phaseTolerance: number,
  mode: HologramConfig["targetPhaseMode"],
): number {
  const amplitudeScore = relativeAmplitudeError / amplitudeTolerance;
  if (mode === "REFERENCE_WGS") return amplitudeScore;
  return Math.max(amplitudeScore, phaseError / phaseTolerance);
}

function fitAmplitudeScale(amplitudes: number[], desired: number[], epsilon: number): number {
  let numerator = 0;
  let denominator = 0;
  for (let index = 0; index < desired.length; index += 1) {
    numerator += desired[index]! * amplitudes[index]!;
    denominator += desired[index]! * desired[index]!;
  }
  return denominator > epsilon ? numerator / denominator : 1;
}

function normalizeWeights(weights: Map<number, number>): void {
  if (weights.size === 0) return;
  const mean = [...weights.values()].reduce((sum, value) => sum + value, 0) / weights.size;
  if (mean <= 0 || !Number.isFinite(mean)) return;
  for (const [id, value] of weights) weights.set(id, value / mean);
}

function cloneCodes(codes: Uint8Array | Uint16Array): Uint8Array | Uint16Array {
  return codes instanceof Uint8Array ? new Uint8Array(codes) : new Uint16Array(codes);
}

function passesQualityGates(metrics: FrameMetrics, gates: HologramConfig["qualityGates"]): boolean {
  if (!metrics.numericalValid) return false;
  if (metrics.flags.includes("ZERO_TARGET_OUTPUT")) return false;
  if (gates?.maxIntensityCoefficientOfVariation !== undefined && metrics.targetIntensityCoefficientOfVariation > gates.maxIntensityCoefficientOfVariation) return false;
  if (gates?.minIntensityToMeanRatio !== undefined && metrics.minimumToMeanIntensityRatio < gates.minIntensityToMeanRatio) return false;
  if (gates?.minDiffractionEfficiency !== undefined && metrics.diffractionEfficiency < gates.minDiffractionEfficiency) return false;
  if (gates?.maxGhostIntensity !== undefined && metrics.maximumGhostIntensity > gates.maxGhostIntensity) return false;
  if (gates?.maxTargetPhaseErrorRad !== undefined && metrics.maximumTargetPhaseErrorRad > gates.maxTargetPhaseErrorRad) return false;
  if (gates?.maxPhaseChangeRad !== undefined && metrics.targetPhaseChangeRad > gates.maxPhaseChangeRad) return false;
  if (gates?.maxDisplayCodeChange !== undefined && metrics.displayCodeChange > gates.maxDisplayCodeChange) return false;
  return true;
}

function calibrationValue(value: ArrayLike<number> | number | undefined, index: number, fallback: number): number {
  if (value === undefined) return fallback;
  if (typeof value === "number") return value;
  return value[index] ?? fallback;
}

function inverseLut(phase: number, calibration: CalibrationPackage, maxCode: number): number {
  if (calibration.inversePhaseLut && calibration.inversePhaseLut.length > 1) {
    const position = ((phase + Math.PI) / TAU) * (calibration.inversePhaseLut.length - 1);
    const low = Math.floor(position);
    const high = Math.min(calibration.inversePhaseLut.length - 1, low + 1);
    const fraction = position - low;
    return (calibration.inversePhaseLut[low] ?? 0) * (1 - fraction) + (calibration.inversePhaseLut[high] ?? 0) * fraction;
  }
  if (calibration.phaseResponseLut && calibration.phaseResponseLut.length > 1) {
    const lut = calibration.phaseResponseLut;
    const phaseRange = calibration.manifest.phaseConvention === "ZERO_TO_TWO_PI" ||
      (calibration.manifest.phaseConvention === undefined && lut[0]! >= -1e-9 && lut[lut.length - 1]! > Math.PI)
      ? (phase < 0 ? phase + TAU : phase)
      : phase;
    const increasing = lut[0]! <= lut[lut.length - 1]!;
    let low = 0;
    let high = lut.length - 1;
    while (high - low > 1) {
      const middle = (low + high) >>> 1;
      const before = lut[middle]!;
      if (increasing ? before < phaseRange : before > phaseRange) low = middle;
      else high = middle;
    }
    const bestCode = Math.abs(lut[low]! - phaseRange) <= Math.abs(lut[high]! - phaseRange) ? low : high;
    return bestCode / Math.max(1, lut.length - 1) * maxCode;
  }
  return ((phase + Math.PI) / TAU) * maxCode;
}

function cloneField(field: ComplexField): ComplexField {
  return { real: new Float64Array(field.real), imag: new Float64Array(field.imag), width: field.width, height: field.height };
}

function maximumGhostIntensity(field: ComplexField, targets: { x: number; y: number }[]): number {
  if (targets.length === 0) return 0;
  let maximum = 0;
  for (let y = 0; y < field.height; y += 1) {
    for (let x = 0; x < field.width; x += 1) {
      if (targets.some((target) => Math.hypot(
        periodicFftDistance(target.x, x, field.width),
        periodicFftDistance(target.y, y, field.height),
      ) <= 1.5)) continue;
      const index = y * field.width + x;
      maximum = Math.max(maximum, field.real[index]! ** 2 + field.imag[index]! ** 2);
    }
  }
  return maximum;
}

function maximumMapPhaseChange(previous: Map<number, number>, current: Map<number, number>): number {
  let maximum = 0;
  for (const [id, phase] of current) {
    const old = previous.get(id);
    if (old !== undefined) maximum = Math.max(maximum, angularDistance(old, phase));
  }
  return maximum;
}

function estimateTransitionMinimumIntensity(
  current: Map<number, number>,
  frame: TrapFrame,
  previous: SolverState | undefined,
  codes: Uint8Array | Uint16Array,
  format: "UINT8" | "UINT16",
): number {
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

function maximumCodeChange(previous: Uint8Array | Uint16Array, current: Uint8Array | Uint16Array): number {
  let maximum = 0;
  const length = Math.min(previous.length, current.length);
  for (let index = 0; index < length; index += 1) maximum = Math.max(maximum, Math.abs(previous[index]! - current[index]!));
  return maximum;
}

function meanOf(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values: number[], mean: number): number {
  return values.length === 0 ? 0 : Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length);
}

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

export function frameDescriptor(
  frame: TrapFrame,
  pixels: Uint8Array | Uint16Array,
  width: number,
  height: number,
  format: "UINT8" | "UINT16",
  byteOffset: bigint,
): SlmFrameDescriptor {
  const bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  return {
    frameIndex: frame.frameIndex,
    timeUs: frame.timeUs,
    width,
    height,
    format,
    byteOffset,
    byteLength: pixels.byteLength,
    crc32: crc32(bytes),
  };
}
