/** A safe rectangle in the focal plane, expressed in micrometres. */
export interface SamplingBounds {
  minXUm: number;
  maxXUm: number;
  minYUm: number;
  maxYUm: number;
}

/** Elliptical region around the undiffracted order in which traps are forbidden. */
export interface ZeroOrderExclusion {
  centerXUm?: number;
  centerYUm?: number;
  radiusXUm: number;
  radiusYUm: number;
}

/**
 * Optionally draw traps from a random sub-window of the safe focal-plane bounds.
 * Fractions are relative to the full safe width/height.
 */
export interface LocalWindowConfig {
  /** Chance of using a local window. Defaults to 1 when this object is supplied. */
  probability?: number;
  minWidthFraction: number;
  minHeightFraction: number;
  maxWidthFraction?: number;
  maxHeightFraction?: number;
}

export interface PositionSamplingConfig {
  bounds: SamplingBounds;
  minimumSpacingXUm: number;
  minimumSpacingYUm: number;
  zeroOrderExclusion?: ZeroOrderExclusion;
  localWindow?: LocalWindowConfig;
  /** Maximum rejected candidates per requested point. Defaults to 256. */
  maxAttemptsPerPoint?: number;
}

export interface TrapCountScheduleConfig {
  totalSamples: number;
  masterSeed: number;
  minTrapCount?: number;
  maxTrapCount?: number;
  /** Defaults to logarithmic stratification. */
  distribution?: "log-uniform" | "uniform";
}

export interface DatasetSamplingConfig extends PositionSamplingConfig, TrapCountScheduleConfig {}

export interface SampledTrap {
  trapId: number;
  xUm: number;
  yUm: number;
}

export interface SampleLayout {
  sampleId: number;
  samplingSeed: number;
  trapCount: number;
  /** The actual full or local window used for this sample. */
  bounds: SamplingBounds;
  /** Canonical row-major spatial order (Y, then X), with IDs 1..trapCount. */
  traps: SampledTrap[];
}

const UINT32_SCALE = 0x1_0000_0000;
const DEFAULT_MIN_TRAPS = 1;
const DEFAULT_MAX_TRAPS = 2000;
const DEFAULT_MAX_ATTEMPTS_PER_POINT = 256;

/**
 * Small, reproducible uint32 PRNG based on Mulberry32. It intentionally uses
 * only specified ECMAScript 32-bit integer operations, so Node/Dawn retries and
 * test runs produce the same stream.
 */
export class Uint32Prng {
  private state: number;

  public constructor(seed: number) {
    this.state = requireUint32(seed, "PRNG seed");
  }

  public nextUint32(): number {
    this.state = (this.state + 0x6d2b_79f5) >>> 0;
    let value = this.state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return (value ^ (value >>> 14)) >>> 0;
  }

  /** Uniform on [0, 1), including zero and never including one. */
  public nextFloat(): number {
    return this.nextUint32() / UINT32_SCALE;
  }

  public uniform(minimum: number, maximum: number): number {
    return minimum + (maximum - minimum) * this.nextFloat();
  }
}

/** Mix a uint32 without retaining mutable state. */
function mixUint32(value: number): number {
  value = Math.imul(value ^ (value >>> 16), 0x7feb_352d);
  value = Math.imul(value ^ (value >>> 15), 0x846c_a68b);
  return (value ^ (value >>> 16)) >>> 0;
}

/** A retry-stable seed for one dataset sample. */
export function deriveSampleSeed(masterSeed: number, sampleId: number, attempt = 0): number {
  const seed = requireUint32(masterSeed, "Master seed");
  requireSampleId(sampleId);
  const retry = requireUint32(attempt, "Sampling attempt");
  const id = BigInt(sampleId);
  const low = Number(id & 0xffff_ffffn);
  const high = Number((id >> 32n) & 0xffff_ffffn);
  return mixUint32(
    seed
    ^ mixUint32(low ^ 0x9e37_79b9)
    ^ mixUint32(high ^ 0x85eb_ca6b)
    ^ mixUint32(retry ^ 0xc2b2_ae35),
  );
}

/**
 * Assign an exact logarithmic stratum to a sample. The affine permutation
 * makes adjacent sample IDs non-monotonic while preserving one visit to every
 * stratum. Consequently retrying a sample ID never changes its trap count.
 */
export function sampleTrapCount(sampleId: number, config: TrapCountScheduleConfig): number {
  const totalSamples = requirePositiveInteger(config.totalSamples, "Total sample count");
  if (totalSamples > 0xffff_ffff) throw new Error("Total sample count must fit in uint32");
  requireSampleId(sampleId);
  if (sampleId >= totalSamples) {
    throw new Error(`Sample ID ${sampleId} is outside the configured range 0..${totalSamples - 1}`);
  }
  const masterSeed = requireUint32(config.masterSeed, "Master seed");
  const minimum = requirePositiveInteger(config.minTrapCount ?? DEFAULT_MIN_TRAPS, "Minimum trap count");
  const maximum = requirePositiveInteger(config.maxTrapCount ?? DEFAULT_MAX_TRAPS, "Maximum trap count");
  const distribution = config.distribution ?? "log-uniform";
  if (distribution !== "log-uniform" && distribution !== "uniform") {
    throw new Error(`Unsupported trap-count distribution: ${String(distribution)}`);
  }
  if (maximum < minimum) throw new Error("Maximum trap count must be at least the minimum trap count");
  if (minimum === maximum) return minimum;
  if (totalSamples === 1) {
    return distribution === "uniform"
      ? Math.round((minimum + maximum) / 2)
      : Math.round(Math.sqrt(minimum * maximum));
  }

  const stratum = permuteStratum(sampleId, totalSamples, masterSeed);
  const fraction = stratum / (totalSamples - 1);
  const value = distribution === "uniform"
    ? minimum + fraction * (maximum - minimum)
    : Math.exp(Math.log(minimum) + fraction * (Math.log(maximum) - Math.log(minimum)));
  return Math.max(minimum, Math.min(maximum, Math.round(value)));
}

/**
 * Generate Poisson-like random focal-plane positions with an anisotropic
 * exclusion ellipse. A spatial hash limits neighbour checks to nearby cells.
 */
export function sampleTrapPositions(
  trapCount: number,
  samplingSeed: number,
  config: PositionSamplingConfig,
): { bounds: SamplingBounds; traps: SampledTrap[] } {
  const count = requirePositiveInteger(trapCount, "Trap count");
  const seed = requireUint32(samplingSeed, "Sampling seed");
  const safeBounds = validateBounds(config.bounds, "Safe bounds");
  const spacingX = requirePositiveFinite(config.minimumSpacingXUm, "Minimum X spacing");
  const spacingY = requirePositiveFinite(config.minimumSpacingYUm, "Minimum Y spacing");
  const attemptsPerPoint = requirePositiveInteger(
    config.maxAttemptsPerPoint ?? DEFAULT_MAX_ATTEMPTS_PER_POINT,
    "Maximum attempts per point",
  );
  const zeroOrder = validateZeroOrderExclusion(config.zeroOrderExclusion);
  const rng = new Uint32Prng(seed);
  const bounds = chooseSamplingWindow(safeBounds, count, spacingX, spacingY, config.localWindow, rng);
  const accepted: MutablePoint[] = [];
  const cells = new Map<string, MutablePoint[]>();
  const maximumAttempts = Math.max(1024, count * attemptsPerPoint);

  for (let attempt = 0; accepted.length < count && attempt < maximumAttempts; attempt += 1) {
    const candidate: MutablePoint = {
      // Positions cross the WebGPU/HDF5 boundary as float32. Quantize before
      // testing exclusions so the stored coordinates retain every guarantee.
      xUm: Math.fround(rng.uniform(bounds.minXUm, bounds.maxXUm)),
      yUm: Math.fround(rng.uniform(bounds.minYUm, bounds.maxYUm)),
      acceptanceIndex: accepted.length,
    };
    if (
      candidate.xUm < bounds.minXUm || candidate.xUm > bounds.maxXUm
      || candidate.yUm < bounds.minYUm || candidate.yUm > bounds.maxYUm
    ) continue;
    if (insideZeroOrder(candidate, zeroOrder)) continue;

    const cellX = Math.floor(candidate.xUm / spacingX);
    const cellY = Math.floor(candidate.yUm / spacingY);
    if (hasTooCloseNeighbour(candidate, cellX, cellY, spacingX, spacingY, cells)) continue;

    accepted.push(candidate);
    const key = cellKey(cellX, cellY);
    const bucket = cells.get(key);
    if (bucket) bucket.push(candidate);
    else cells.set(key, [candidate]);
  }

  if (accepted.length !== count) {
    throw new Error(
      `Could place only ${accepted.length} of ${count} traps after ${maximumAttempts} candidates; `
      + "increase the safe bounds or attempt budget, or reduce the minimum spacing/local-window density",
    );
  }

  accepted.sort((first, second) => (
    first.yUm - second.yUm
    || first.xUm - second.xUm
    || first.acceptanceIndex - second.acceptanceIndex
  ));
  const traps = accepted.map((point, index) => ({
    trapId: index + 1,
    xUm: point.xUm,
    yUm: point.yUm,
  }));
  return { bounds, traps };
}

export function generateSampleLayout(sampleId: number, config: DatasetSamplingConfig, attempt = 0): SampleLayout {
  const trapCount = sampleTrapCount(sampleId, config);
  const samplingSeed = deriveSampleSeed(config.masterSeed, sampleId, attempt);
  const sampled = sampleTrapPositions(trapCount, samplingSeed, config);
  return {
    sampleId,
    samplingSeed,
    trapCount,
    bounds: sampled.bounds,
    traps: sampled.traps,
  };
}

interface MutablePoint {
  xUm: number;
  yUm: number;
  acceptanceIndex: number;
}

interface ValidatedZeroOrder {
  centerXUm: number;
  centerYUm: number;
  radiusXUm: number;
  radiusYUm: number;
}

function chooseSamplingWindow(
  safe: SamplingBounds,
  count: number,
  spacingX: number,
  spacingY: number,
  local: LocalWindowConfig | undefined,
  rng: Uint32Prng,
): SamplingBounds {
  if (!local) return { ...safe };
  const probability = requireProbability(local.probability ?? 1, "Local-window probability");
  const minWidthFraction = requireFraction(local.minWidthFraction, "Minimum local-window width fraction");
  const minHeightFraction = requireFraction(local.minHeightFraction, "Minimum local-window height fraction");
  const maxWidthFraction = requireFraction(local.maxWidthFraction ?? 1, "Maximum local-window width fraction");
  const maxHeightFraction = requireFraction(local.maxHeightFraction ?? 1, "Maximum local-window height fraction");
  if (maxWidthFraction < minWidthFraction || maxHeightFraction < minHeightFraction) {
    throw new Error("Local-window maximum fractions must be at least their minimum fractions");
  }
  if (rng.nextFloat() >= probability) return { ...safe };

  let widthFraction = rng.uniform(minWidthFraction, maxWidthFraction);
  let heightFraction = rng.uniform(minHeightFraction, maxHeightFraction);
  const fullWidth = safe.maxXUm - safe.minXUm;
  const fullHeight = safe.maxYUm - safe.minYUm;

  // Avoid selecting a clearly impossible dense sub-window. This is only a
  // conservative adjustment; rejection sampling remains the final authority.
  const requestedAreaFraction = Math.min(1, count * spacingX * spacingY * 1.2 / (fullWidth * fullHeight));
  if (widthFraction * heightFraction < requestedAreaFraction) {
    const expansion = Math.sqrt(requestedAreaFraction / (widthFraction * heightFraction));
    widthFraction = Math.min(maxWidthFraction, widthFraction * expansion);
    heightFraction = Math.min(maxHeightFraction, heightFraction * expansion);
    if (widthFraction * heightFraction < requestedAreaFraction) {
      widthFraction = Math.min(maxWidthFraction, requestedAreaFraction / heightFraction);
      heightFraction = Math.min(maxHeightFraction, requestedAreaFraction / widthFraction);
    }
  }

  const width = fullWidth * widthFraction;
  const height = fullHeight * heightFraction;
  const minXUm = rng.uniform(safe.minXUm, safe.maxXUm - width);
  const minYUm = rng.uniform(safe.minYUm, safe.maxYUm - height);
  return {
    minXUm,
    maxXUm: minXUm + width,
    minYUm,
    maxYUm: minYUm + height,
  };
}

function hasTooCloseNeighbour(
  candidate: MutablePoint,
  cellX: number,
  cellY: number,
  spacingX: number,
  spacingY: number,
  cells: ReadonlyMap<string, readonly MutablePoint[]>,
): boolean {
  for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
    for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
      const neighbours = cells.get(cellKey(cellX + offsetX, cellY + offsetY));
      if (!neighbours) continue;
      for (const neighbour of neighbours) {
        const dx = (candidate.xUm - neighbour.xUm) / spacingX;
        const dy = (candidate.yUm - neighbour.yUm) / spacingY;
        if (dx * dx + dy * dy < 1) return true;
      }
    }
  }
  return false;
}

function insideZeroOrder(point: Pick<MutablePoint, "xUm" | "yUm">, zeroOrder: ValidatedZeroOrder | undefined): boolean {
  if (!zeroOrder) return false;
  const dx = (point.xUm - zeroOrder.centerXUm) / zeroOrder.radiusXUm;
  const dy = (point.yUm - zeroOrder.centerYUm) / zeroOrder.radiusYUm;
  return dx * dx + dy * dy <= 1;
}

function validateZeroOrderExclusion(value: ZeroOrderExclusion | undefined): ValidatedZeroOrder | undefined {
  if (!value) return undefined;
  return {
    centerXUm: requireFinite(value.centerXUm ?? 0, "Zero-order X centre"),
    centerYUm: requireFinite(value.centerYUm ?? 0, "Zero-order Y centre"),
    radiusXUm: requirePositiveFinite(value.radiusXUm, "Zero-order X radius"),
    radiusYUm: requirePositiveFinite(value.radiusYUm, "Zero-order Y radius"),
  };
}

function permuteStratum(index: number, count: number, seed: number): number {
  let multiplier = (mixUint32(seed ^ 0xa511_e9b3) % count) || 1;
  while (greatestCommonDivisor(multiplier, count) !== 1) {
    multiplier += 1;
    if (multiplier >= count) multiplier = 1;
  }
  const offset = mixUint32(seed ^ 0x63d8_3595) % count;
  return Number((BigInt(multiplier) * BigInt(index) + BigInt(offset)) % BigInt(count));
}

function greatestCommonDivisor(first: number, second: number): number {
  while (second !== 0) {
    const remainder = first % second;
    first = second;
    second = remainder;
  }
  return first;
}

function cellKey(x: number, y: number): string {
  return `${x}:${y}`;
}

function validateBounds(value: SamplingBounds, label: string): SamplingBounds {
  const bounds = {
    minXUm: requireFinite(value.minXUm, `${label} minimum X`),
    maxXUm: requireFinite(value.maxXUm, `${label} maximum X`),
    minYUm: requireFinite(value.minYUm, `${label} minimum Y`),
    maxYUm: requireFinite(value.maxYUm, `${label} maximum Y`),
  };
  if (bounds.maxXUm <= bounds.minXUm || bounds.maxYUm <= bounds.minYUm) {
    throw new Error(`${label} must have positive width and height`);
  }
  return bounds;
}

function requireFinite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite`);
  return value;
}

function requirePositiveFinite(value: number, label: string): number {
  requireFinite(value, label);
  if (value <= 0) throw new Error(`${label} must be greater than zero`);
  return value;
}

function requirePositiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive safe integer`);
  return value;
}

function requireUint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new Error(`${label} must be a uint32 integer`);
  }
  return value >>> 0;
}

function requireSampleId(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("Sample ID must be a non-negative safe integer");
}

function requireProbability(value: number, label: string): number {
  requireFinite(value, label);
  if (value < 0 || value > 1) throw new Error(`${label} must be between zero and one`);
  return value;
}

function requireFraction(value: number, label: string): number {
  requirePositiveFinite(value, label);
  if (value > 1) throw new Error(`${label} must be no greater than one`);
  return value;
}
