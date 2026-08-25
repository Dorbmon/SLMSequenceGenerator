export type NumericArray = ArrayLike<number>;

export interface Point2D {
  xUm: number;
  yUm: number;
}

export interface InitialAtom extends Point2D {
  atomId?: number;
  group?: number;
  movable?: boolean;
  initialTrapIntensity?: number;
  localizationSigmaUm?: number;
}

export interface TargetSite extends Point2D {
  siteId?: number;
  required?: boolean;
  requiredAtomId?: number;
  requiredGroup?: number;
  finalTrapIntensity?: number;
}

export interface StaticTrap extends Point2D {
  trapId: number;
  intensity: number;
  containsAtom: boolean;
  atomId?: number;
}

export interface ForbiddenRegion {
  type: "circle" | "axisAlignedBox" | "polygon";
  coordinates: number[];
  clearanceUm?: number;
}

export interface OccupiedLatticeInput {
  sites: Point2D[];
  occupied: Uint8Array;
}

export type ExtraAtomPolicy =
  | "KEEP"
  | "PARK"
  | "PARK_AND_RELEASE"
  | "RELEASE_IN_PLACE";

export type TargetPhaseMode =
  | "REFERENCE_WGS"
  | "PHASE_LOCKED_WGS"
  | "SOFT_PHASE_LOCKED_WGS"
  | "PHASE_INTERPOLATED_WGS";

export interface AssignmentConfig {
  distanceWeight?: number;
  obstacleWeight?: number;
  staticObstacleWeight?: number;
  groupMismatchPenalty?: number;
  groupMismatchPolicy?: "forbid" | "penalize";
  extraAtomPolicy?: ExtraAtomPolicy;
  parkingSites?: Point2D[];
  stayToleranceUm?: number;
  maxAssignmentRetries?: number;
  conflictPenalty?: number;
}

export interface PlannerConfig {
  minimumSeparationUm?: number;
  kSigma?: number;
  geometricMarginUm?: number;
  duplicatePointToleranceUm?: number;
  duplicatePointPolicy?: "reject" | "merge";
  gridResolutionUm?: number;
  planningTickUs?: number;
  maxSearchTicks?: number;
  maxAStarExpansions?: number;
  maxCbsNodes?: number;
  ecbsLimit?: number;
  maxPriorityRetries?: number;
}

export interface MotionConfig {
  framePeriodUs?: number;
  preMoveDwellUs?: number;
  postMoveSettleUs?: number;
  minDwellBeforeMoveUs?: number;
  maxVelocityUmPerUs?: number;
  maxAccelerationUmPerUs2?: number;
  maxJerkUmPerUs3?: number;
  maxPositionChangePerFrameUm?: number;
  maxIntensityChangePerFrame?: number;
  movingTrapIntensity?: number;
  defaultTrapIntensity?: number;
  maxValidationDepth?: number;
}

export interface CoordinateTransform {
  /** Optional direct transform. It is kept out of JSON manifests by callers. */
  physicalToFft?: (point: Point2D) => Point2D | { x: number; y: number };
  originXUm?: number;
  originYUm?: number;
  scaleX?: number;
  scaleY?: number;
  rotationRad?: number;
  offsetX?: number;
  offsetY?: number;
  /** Affine coefficients: u = a*x + b*y + offsetX, v = c*x + d*y + offsetY. */
  a?: number;
  b?: number;
  c?: number;
  d?: number;
}

export interface CalibrationManifest {
  calibrationId: string;
  revision?: string;
  slmModel?: string;
  serialNumber?: string;
  wavelengthNm: number;
  activeWidth: number;
  activeHeight: number;
  fftWidth?: number;
  fftHeight?: number;
  pixelPitchUm?: number;
  fieldOfViewUm?: { xMinUm: number; xMaxUm: number; yMinUm: number; yMaxUm: number };
  coordinateConvention?: string;
  phaseConvention?: "NEGATIVE_PI_TO_PI" | "ZERO_TO_TWO_PI";
  checksums?: Record<string, string>;
  formatVersion?: string;
}

export interface CalibrationPackage {
  manifest: CalibrationManifest;
  incidentAmplitude?: NumericArray;
  apertureMask?: NumericArray;
  coordinateTransform?: CoordinateTransform;
  aberrationPhase?: NumericArray | number;
  phaseResponseLut?: NumericArray;
  inversePhaseLut?: NumericArray;
  carrierGrating?: NumericArray | number;
  digitalLens?: NumericArray | number;
  /** Explicit signs make composition unambiguous for measured maps. */
  phaseSigns?: {
    aberration?: 1 | -1;
    grating?: 1 | -1;
    lens?: 1 | -1;
  };
}

export interface HologramQualityGates {
  maxIntensityCoefficientOfVariation?: number;
  minIntensityToMeanRatio?: number;
  minDiffractionEfficiency?: number;
  maxGhostIntensity?: number;
  maxTargetPhaseErrorRad?: number;
  maxPhaseChangeRad?: number;
  maxDisplayCodeChange?: number;
}

export interface HologramConfig {
  width?: number;
  height?: number;
  format?: "UINT8" | "UINT16";
  targetPhaseMode?: TargetPhaseMode;
  firstFrameIterations?: number;
  subsequentFrameIterations?: number;
  maxIterations?: number;
  gamma?: number;
  epsilon?: number;
  minWeight?: number;
  maxWeight?: number;
  convergenceTolerance?: number;
  backgroundPolicy?: "PRESERVE" | "ZERO";
  oversampling?: number;
  qualityGates?: HologramQualityGates;
  maxInsertedFrames?: number;
  deterministicSeed?: number;
  measureSolveTime?: boolean;
  requireConvergence?: boolean;
}

export interface CompilerConfig {
  calibration?: CalibrationPackage;
  calibrations?: Record<string, CalibrationPackage>;
  planner?: PlannerConfig;
  assignment?: AssignmentConfig;
  motion?: MotionConfig;
  hologram?: HologramConfig;
  compilerVersion?: string;
  wasmBuildId?: string;
  creationTimestamp?: string;
  simulationMode?: boolean;
  wavelengthNm?: number;
  strictCalibration?: boolean;
  calibrationRequirements?: {
    slmModel?: string;
    serialNumber?: string;
    wavelengthNm?: number;
  };
}

export interface RearrangementRequest {
  initialAtoms: InitialAtom[] | OccupiedLatticeInput;
  targetSites: TargetSite[];
  calibrationId?: string;
  calibration?: CalibrationPackage;
  staticTraps?: StaticTrap[];
  forbiddenRegions?: ForbiddenRegion[];
  plannerConfig?: PlannerConfig;
  motionConfig?: MotionConfig;
  hologramConfig?: HologramConfig;
  assignmentConfig?: AssignmentConfig;
  wavelengthNm?: number;
}

export interface CompileProgress {
  stage:
    | "VALIDATING"
    | "ASSIGNING"
    | "PLANNING"
    | "PARAMETERIZING"
    | "GENERATING_TRAP_FRAMES"
    | "SOLVING_SLM_FRAMES"
    | "VALIDATING_SEQUENCE"
    | "WRITING_OUTPUT";
  completed: number;
  total: number;
  frameIndex?: number;
  message?: string;
}

export interface CompileOptions {
  signal?: AbortSignal;
  onProgress?: (progress: CompileProgress) => void;
  outputStore?: FrameStore<Uint8Array | Uint16Array>;
  /**
   * Optional browser/backend integration point. The core compiler keeps all
   * motion planning and validation unchanged while allowing an asynchronous
   * hologram solver (for example, a GPU-resident WebGPU implementation).
   */
  hologramSolverFactory?: HologramSolverFactory;
}

export interface AtomAssignment {
  atomId: number;
  sourceIndex: number;
  targetSiteId: number | null;
  targetIndex: number | null;
  disposition: "MOVE_TO_TARGET" | "STAY" | "PARK" | "KEEP" | "RELEASE";
  assignmentCost: number;
  parkingSiteIndex?: number;
}

export interface TrajectoryWaypoint extends Point2D {
  arrivalTimeUs: number;
}

export interface AtomTrajectory {
  atomId: number;
  trapId: number;
  targetSiteId: number | null;
  waypoints: TrajectoryWaypoint[];
  startTimeUs: number;
  endTimeUs: number;
  moving: boolean;
  disposition?: AtomAssignment["disposition"];
  initialIntensity?: number;
  finalIntensity?: number;
  staticTrap?: boolean;
}

export interface TrapState {
  trapId: number;
  atomId: number | null;
  xUm: number;
  yUm: number;
  intensity: number;
  targetPhaseRad: number;
  flags: number;
}

export interface TrapFrame {
  frameIndex: number;
  timeUs: number;
  traps: TrapState[];
}

export interface SlmFrameDescriptor {
  frameIndex: number;
  timeUs: number;
  width: number;
  height: number;
  format: "UINT8" | "UINT16";
  byteOffset: bigint;
  byteLength: number;
  crc32: number;
}

export interface FrameStore<T> extends Iterable<T>, AsyncIterable<T> {
  readonly length: number;
  readonly count?: number;
  append(frame: T): void | Promise<void>;
  get(index: number): T | Promise<T>;
  toArray(): T[] | Promise<T[]>;
  clear?(): void | Promise<void>;
}

export interface FrameMetrics {
  frameIndex: number;
  timeUs: number;
  iterations: number;
  converged: boolean;
  targetIntensityMean: number;
  targetIntensityStd: number;
  targetIntensityCoefficientOfVariation: number;
  minimumToMeanIntensityRatio: number;
  diffractionEfficiency: number;
  maximumGhostIntensity: number;
  maximumWgsWeight: number;
  maximumTargetPhaseErrorRad: number;
  targetPhaseChangeRad: number;
  displayCodeChange: number;
  estimatedTransitionMinimumIntensity: number;
  solveTimeMs: number;
  refinementCount: number;
  numericalValid: boolean;
  accepted: boolean;
  flags: string[];
}

export interface HologramSolveResult {
  pixels: Uint8Array | Uint16Array;
  metrics: FrameMetrics;
  descriptor?: SlmFrameDescriptor;
}

export interface SequentialHologramBackend {
  readonly backendId: string;
  solveSequentialFrame(
    frame: TrapFrame,
    iterationBudget?: number,
  ): HologramSolveResult | Promise<HologramSolveResult>;
  commitFrameState(): void | Promise<void>;
  rollbackToPreviousAcceptedFrame(): void | Promise<void>;
  dispose?(): void | Promise<void>;
}

export type HologramSolverFactory = (
  calibration: CalibrationPackage,
  config: Required<HologramConfig>,
) => SequentialHologramBackend | Promise<SequentialHologramBackend>;

export interface SequenceManifest {
  formatVersion: string;
  creationTimestamp: string;
  compilerVersion: string;
  wasmBuildId: string;
  inputHash: string;
  calibrationId: string;
  calibrationHash: string;
  coordinateConvention: string;
  atomCount: number;
  targetCount: number;
  trapCount: number;
  frameCount: number;
  framePeriodUs: number;
  assignmentCost: number;
  assignmentAttempts: number;
  plannerBackend: string;
  plannerParameters: Record<string, number | string | boolean>;
  wgsBackend: string;
  wgsParameters: Record<string, number | string | boolean>;
  outputWidth: number;
  outputHeight: number;
  pixelFormat: "UINT8" | "UINT16";
  deterministicSeed: number;
  checksums: Record<string, string>;
  validationStatus: "accepted" | "rejected";
}

export interface SequenceValidationReport {
  accepted: boolean;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  minimumAtomSeparationUm: number;
  maximumSpeedUmPerUs: number;
  maximumAccelerationUmPerUs2: number;
  maximumJerkUmPerUs3: number;
  frameCount: number;
}

export interface ValidationIssue {
  code: string;
  stage: string;
  message: string;
  atomIds?: number[];
  targetSiteIds?: number[];
  frameIndex?: number;
  configured?: number | string;
  measured?: number | string;
}

export interface CompiledSequence {
  manifest: SequenceManifest;
  assignment: AtomAssignment[];
  trajectories: AtomTrajectory[];
  trapFrameStore: FrameStore<TrapFrame>;
  slmFrameStore: FrameStore<Uint8Array | Uint16Array>;
  slmFrameDescriptors: SlmFrameDescriptor[];
  frameMetrics: FrameMetrics[];
  validation: SequenceValidationReport;
}

export type CompiledSequenceHandle = CompiledSequence;

export interface NormalizedInput {
  initialAtoms: Required<InitialAtom>[];
  targetSites: Required<TargetSite>[];
  staticTraps: Required<StaticTrap>[];
  forbiddenRegions: ForbiddenRegion[];
  calibration: CalibrationPackage;
  plannerConfig: Required<PlannerConfig>;
  assignmentConfig: Required<AssignmentConfig>;
  motionConfig: Required<MotionConfig>;
  hologramConfig: Required<HologramConfig>;
}

export interface PlannedPath {
  atomId: number;
  trapId: number;
  goalSiteId: number | null;
  disposition: AtomAssignment["disposition"];
  waypointsUm: Point2D[];
  discreteTicks: number[];
  startTimeUs?: number;
}

export interface MotionPlan {
  assignment: AtomAssignment[];
  plannedPaths: PlannedPath[];
  trajectories: AtomTrajectory[];
  directPathCount: number;
  conflictComponentCount: number;
  waitCount: number;
  detourCount: number;
  makespanUs: number;
  minimumValidatedSeparationUm: number;
}
