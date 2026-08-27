export const SAMPLE_PAYLOAD_MAGIC = "SLMD";
export const SAMPLE_PAYLOAD_VERSION = 1;
export const SAMPLE_PAYLOAD_HEADER_BYTES = 64;

const CRC32_TABLE = createCrc32Table();

export interface SamplePayloadInput {
  sampleId: number | bigint;
  samplingSeed: number;
  width: number;
  height: number;
  /** Flattened [trap_count, 2] X/Y positions, in micrometres. */
  positions: Float32Array;
  /** One measured focal-plane phase per position, in radians. */
  measuredPhases: Float32Array;
  /** One stable trap ID per position. */
  trapIds: Uint32Array;
  /** Exported uint8 SLM codes (logical or raw, per frameMode) in row-major order. */
  frame: Uint8Array;
  /** JSON-serializable convergence and solve metadata. */
  metrics: unknown;
}

/** IEEE 802.3/ZIP CRC-32 of the exact exported frame bytes. */
export function crc32(bytes: Uint8Array): number {
  let crc = 0xffff_ffff;
  for (const byte of bytes) {
    crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ byte) & 0xff]!;
  }
  return (crc ^ 0xffff_ffff) >>> 0;
}

/**
 * Encode one HTTP POST body using the versioned little-endian dataset wire
 * format shared with generate.py. HTTP Content-Length supplies framing; there
 * is deliberately no outer length prefix.
 *
 * Header: <4sHHQ12I (64 bytes), followed by positions f32, measured phases
 * f32, trap IDs u32, frame u8, and UTF-8 metrics JSON.
 */
export function encodeSamplePayload(input: SamplePayloadInput): Uint8Array {
  const sampleId = requireUint64(input.sampleId, "Sample ID");
  const samplingSeed = requireUint32(input.samplingSeed, "Sampling seed");
  const width = requirePositiveUint32(input.width, "Frame width");
  const height = requirePositiveUint32(input.height, "Frame height");
  const trapCount = input.measuredPhases.length;
  if (trapCount > 0xffff_ffff) throw new Error("Trap count must fit in uint32");
  if (input.positions.length !== trapCount * 2) {
    throw new Error(`Positions length ${input.positions.length} does not equal trap_count * 2 (${trapCount * 2})`);
  }
  if (input.trapIds.length !== trapCount) {
    throw new Error(`Trap ID length ${input.trapIds.length} does not equal trap_count (${trapCount})`);
  }
  const expectedFrameBytes = width * height;
  if (!Number.isSafeInteger(expectedFrameBytes) || input.frame.byteLength !== expectedFrameBytes) {
    throw new Error(`Frame byte length ${input.frame.byteLength} does not equal width * height (${expectedFrameBytes})`);
  }

  const metricsJson = stringifyMetrics(input.metrics);
  const metricsBytes = new TextEncoder().encode(metricsJson);
  const positionsBytes = input.positions.length * Float32Array.BYTES_PER_ELEMENT;
  const phasesBytes = input.measuredPhases.length * Float32Array.BYTES_PER_ELEMENT;
  const trapIdsBytes = input.trapIds.length * Uint32Array.BYTES_PER_ELEMENT;
  requireUint32(positionsBytes, "Positions byte length");
  requireUint32(phasesBytes, "Measured-phases byte length");
  requireUint32(trapIdsBytes, "Trap-IDs byte length");
  requireUint32(input.frame.byteLength, "Frame byte length");
  requireUint32(metricsBytes.byteLength, "Metrics byte length");

  const totalBytes = SAMPLE_PAYLOAD_HEADER_BYTES
    + positionsBytes
    + phasesBytes
    + trapIdsBytes
    + input.frame.byteLength
    + metricsBytes.byteLength;
  if (!Number.isSafeInteger(totalBytes)) throw new Error("Encoded sample payload is too large");
  const payload = new Uint8Array(totalBytes);
  const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
  writeAsciiMagic(payload);
  view.setUint16(4, SAMPLE_PAYLOAD_VERSION, true);
  view.setUint16(6, SAMPLE_PAYLOAD_HEADER_BYTES, true);
  view.setBigUint64(8, sampleId, true);
  view.setUint32(16, trapCount, true);
  view.setUint32(20, samplingSeed, true);
  view.setUint32(24, crc32(input.frame), true);
  view.setUint32(28, width, true);
  view.setUint32(32, height, true);
  view.setUint32(36, positionsBytes, true);
  view.setUint32(40, phasesBytes, true);
  view.setUint32(44, trapIdsBytes, true);
  view.setUint32(48, input.frame.byteLength, true);
  view.setUint32(52, metricsBytes.byteLength, true);
  view.setUint32(56, 0, true); // flags
  view.setUint32(60, 0, true); // reserved

  let offset = SAMPLE_PAYLOAD_HEADER_BYTES;
  for (const value of input.positions) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  for (const value of input.measuredPhases) {
    view.setFloat32(offset, value, true);
    offset += 4;
  }
  for (const value of input.trapIds) {
    view.setUint32(offset, value, true);
    offset += 4;
  }
  payload.set(input.frame, offset);
  offset += input.frame.byteLength;
  payload.set(metricsBytes, offset);
  return payload;
}

function stringifyMetrics(metrics: unknown): string {
  let result: string | undefined;
  try {
    result = JSON.stringify(metrics);
  } catch (error) {
    throw new Error(`Metrics are not JSON-serializable: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (result === undefined) throw new Error("Metrics must be JSON-serializable");
  return result;
}

function writeAsciiMagic(target: Uint8Array): void {
  target[0] = 0x53; // S
  target[1] = 0x4c; // L
  target[2] = 0x4d; // M
  target[3] = 0x44; // D
}

function createCrc32Table(): Uint32Array {
  const table = new Uint32Array(256);
  for (let index = 0; index < table.length; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = (value >>> 1) ^ (0xedb8_8320 & -(value & 1));
    }
    table[index] = value >>> 0;
  }
  return table;
}

function requireUint64(value: number | bigint, label: string): bigint {
  let integer: bigint;
  if (typeof value === "bigint") integer = value;
  else {
    if (!Number.isSafeInteger(value)) throw new Error(`${label} must be a safe integer or bigint`);
    integer = BigInt(value);
  }
  if (integer < 0n || integer > 0xffff_ffff_ffff_ffffn) throw new Error(`${label} must fit in uint64`);
  return integer;
}

function requireUint32(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0 || value > 0xffff_ffff) throw new Error(`${label} must fit in uint32`);
  return value;
}

function requirePositiveUint32(value: number, label: string): number {
  requireUint32(value, label);
  if (value === 0) throw new Error(`${label} must be greater than zero`);
  return value;
}
