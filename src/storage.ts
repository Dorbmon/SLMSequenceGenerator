import { crc32, hashString, hashValue, stableStringify } from "./util.js";
import { SlmError } from "./errors.js";
import type {
  CompiledSequence,
  SlmFrameDescriptor,
  TrapFrame,
  TrapState,
} from "./types.js";

export interface SequencePackageFiles {
  "manifest.json": string;
  "assignment.json": string;
  "trajectories.json": string;
  "trap-frames.bin": Uint8Array;
  "trap-frames.index.json": string;
  "slm-frames.bin": Uint8Array;
  "slm-frames.index.json": string;
  "frame-metrics.jsonl": string;
  "validation.json": string;
  "calibration-manifest.json": string;
}

export type SlmFrameDescriptorJson = Omit<SlmFrameDescriptor, "byteOffset"> & { byteOffset: string };

/** Build the deterministic archive entries described in design.md section 20. */
export function createSequencePackage(sequence: CompiledSequence): SequencePackageFiles {
  const slmFrames = syncValue(sequence.slmFrameStore.toArray()) as (Uint8Array | Uint16Array)[];
  const slmBytes = concatenateSlmFrames(slmFrames);
  const trapFrames = syncValue(sequence.trapFrameStore.toArray());
  const trapBytes = encodeTrapFrames(trapFrames);
  const descriptors = sequence.slmFrameDescriptors;
  const trapIndex = trapFrames.map((frame, index) => ({
    frameIndex: frame.frameIndex,
    timeUs: frame.timeUs,
    byteOffset: trapFrameOffset(trapFrames, index),
    byteLength: trapFrameByteLength(frame),
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
    "calibration-manifest.json": stableStringify({ calibrationId: sequence.manifest.calibrationId, calibrationHash: sequence.manifest.calibrationHash }),
  };
}

export async function createSequencePackageAsync(sequence: CompiledSequence): Promise<SequencePackageFiles> {
  const [slmFrames, trapFrames] = await Promise.all([
    sequence.slmFrameStore.toArray(),
    sequence.trapFrameStore.toArray(),
  ]) as [Uint8Array[] | Uint16Array[], TrapFrame[]];
  const slmBytes = concatenateSlmFrames(slmFrames);
  const trapBytes = encodeTrapFrames(trapFrames);
  const trapIndex = trapFrames.map((frame, index) => ({
    frameIndex: frame.frameIndex,
    timeUs: frame.timeUs,
    byteOffset: trapFrameOffset(trapFrames, index),
    byteLength: trapFrameByteLength(frame),
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
    "calibration-manifest.json": stableStringify({ calibrationId: sequence.manifest.calibrationId, calibrationHash: sequence.manifest.calibrationHash }),
  };
}

export function encodeTrapFrames(frames: TrapFrame[]): Uint8Array {
  const bytesPerState = 4 + 4 + 1 + 8 + 8 + 8 + 8 + 4;
  const bytesPerFrame = 4 + 8 + 4;
  const total = frames.reduce((sum, frame) => sum + bytesPerFrame + frame.traps.length * bytesPerState, 0);
  const output = new Uint8Array(total);
  const view = new DataView(output.buffer);
  let offset = 0;
  for (const frame of frames) {
    if (!Number.isInteger(frame.frameIndex) || frame.frameIndex < 0 || frame.frameIndex > 0xffffffff || !Number.isInteger(frame.timeUs) || frame.timeUs < 0) {
      throw new SlmError("STORAGE_ERROR", "Trap frame index and time must fit the binary format", { stage: "WRITING_OUTPUT" });
    }
    view.setUint32(offset, frame.frameIndex, true);
    offset += 4;
    view.setBigInt64(offset, BigInt(Math.trunc(frame.timeUs)), true);
    offset += 8;
    view.setUint32(offset, frame.traps.length, true);
    offset += 4;
    for (const trap of frame.traps) {
      if (!Number.isInteger(trap.trapId) || trap.trapId < 0 || trap.trapId > 0xffffffff ||
          (trap.atomId !== null && (!Number.isInteger(trap.atomId) || trap.atomId < 0 || trap.atomId > 0xffffffff))) {
        throw new SlmError("STORAGE_ERROR", "Trap identifiers must fit uint32", { stage: "WRITING_OUTPUT" });
      }
      view.setUint32(offset, trap.trapId, true);
      offset += 4;
      // Presence is explicit so uint32 atom id 0xffffffff remains valid.
      view.setUint32(offset, trap.atomId === null || trap.atomId === undefined ? 0 : trap.atomId >>> 0, true);
      offset += 4;
      view.setUint8(offset, trap.atomId === null || trap.atomId === undefined ? 0 : 1);
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

export function decodeTrapFrames(bytes: Uint8Array): TrapFrame[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frames: TrapFrame[] = [];
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
    if (count > 0x1000000 || offset + count * stateBytes > bytes.byteLength) {
      throw new SlmError("STORAGE_ERROR", "Truncated trap-frame state data", { stage: "VALIDATING" });
    }
    const traps: TrapState[] = [];
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

export function concatenateSlmFrames(frames: (Uint8Array | Uint16Array)[]): Uint8Array {
  const byteLength = frames.reduce((sum, frame) => sum + frame.byteLength, 0);
  const output = new Uint8Array(byteLength);
  let offset = 0;
  for (const frame of frames) {
    output.set(new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength), offset);
    offset += frame.byteLength;
  }
  return output;
}

export function verifySequencePackage(
  sequence: CompiledSequence,
  descriptors: ReadonlyArray<SlmFrameDescriptor | SlmFrameDescriptorJson> = sequence.slmFrameDescriptors,
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const normalizedDescriptors = descriptors.map(normalizeDescriptor);
  const stored = sequence.slmFrameStore.toArray();
  if (stored instanceof Promise) throw new TypeError("This frame store is asynchronous; use verifySequencePackageAsync");
  const frames = stored as (Uint8Array | Uint16Array)[];
  if (frames.length !== normalizedDescriptors.length) errors.push("SLM frame count does not match descriptor count");
  for (let index = 0; index < Math.min(frames.length, normalizedDescriptors.length); index += 1) {
    const descriptor = normalizedDescriptors[index]!;
    const bytes = new Uint8Array(frames[index]!.buffer, frames[index]!.byteOffset, frames[index]!.byteLength);
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

export async function verifySequencePackageAsync(
  sequence: CompiledSequence,
  descriptors: ReadonlyArray<SlmFrameDescriptor | SlmFrameDescriptorJson> = sequence.slmFrameDescriptors,
): Promise<{ valid: boolean; errors: string[] }> {
  const [stored, expectedDescriptors] = await Promise.all([
    sequence.slmFrameStore.toArray(),
    Promise.resolve(descriptors.map(normalizeDescriptor)),
  ]) as [(Uint8Array | Uint16Array)[], SlmFrameDescriptor[]];
  const errors: string[] = [];
  if (stored.length !== expectedDescriptors.length) errors.push("SLM frame count does not match descriptor count");
  for (let index = 0; index < Math.min(stored.length, expectedDescriptors.length); index += 1) {
    const descriptor = expectedDescriptors[index]!;
    const bytes = new Uint8Array(stored[index]!.buffer, stored[index]!.byteOffset, stored[index]!.byteLength);
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

export function verifySequencePackageFiles(
  sequence: CompiledSequence,
  files: SequencePackageFiles,
): { valid: boolean; errors: string[] } {
  const result = verifySequencePackage(sequence);
  const errors = [...result.errors];
  let decoded: TrapFrame[];
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
    const entry = trapIndex[index]!;
    if (entry.frameIndex !== decoded[index]!.frameIndex || entry.timeUs !== decoded[index]!.timeUs || entry.byteOffset !== trapFrameOffset(decoded, index) || entry.byteLength !== trapFrameByteLength(decoded[index]!)) {
      errors.push(`Trap-frame index mismatch at frame ${index}`);
    }
  }
  const descriptors = sequence.slmFrameDescriptors;
  const packageDescriptors = parseJsonArray(files["slm-frames.index.json"]);
  if (packageDescriptors.length !== descriptors.length) errors.push("SLM frame index count mismatch");
  for (let index = 0; index < Math.min(packageDescriptors.length, descriptors.length); index += 1) {
    try {
      const packageDescriptor = normalizeDescriptor(packageDescriptors[index] as SlmFrameDescriptorJson);
      const expected = descriptors[index]!;
      if (packageDescriptor.frameIndex !== expected.frameIndex || packageDescriptor.timeUs !== expected.timeUs || packageDescriptor.byteOffset !== expected.byteOffset || packageDescriptor.byteLength !== expected.byteLength || packageDescriptor.crc32 !== expected.crc32) {
        errors.push(`SLM frame index mismatch at frame ${index}`);
      }
    } catch {
      errors.push(`Invalid SLM frame index at frame ${index}`);
    }
  }
  for (let index = 0; index < descriptors.length; index += 1) {
    const descriptor = descriptors[index]!;
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

export const exportSequence = createSequencePackage;

function syncValue<T>(value: T | Promise<T>): T {
  if (value instanceof Promise) {
    throw new TypeError("This frame store is asynchronous; use createSequencePackageAsync");
  }
  return value;
}

function descriptorForJson(descriptor: SlmFrameDescriptor): SlmFrameDescriptorJson {
  return { ...descriptor, byteOffset: descriptor.byteOffset.toString() };
}

function normalizeDescriptor(descriptor: SlmFrameDescriptor | SlmFrameDescriptorJson): SlmFrameDescriptor {
  return { ...descriptor, byteOffset: parseByteOffset(descriptor.byteOffset) };
}

function parseByteOffset(value: bigint | string): bigint {
  if (typeof value === "bigint") return value;
  if (!/^\d+$/.test(value)) throw new TypeError(`Invalid decimal frame offset: ${value}`);
  return BigInt(value);
}

function parseJsonArray(value: string): Record<string, any>[] {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? parsed as Record<string, any>[] : [];
  } catch {
    return [];
  }
}

function trapFrameByteLength(frame: TrapFrame): number {
  return 16 + frame.traps.length * (4 + 4 + 1 + 8 + 8 + 8 + 8 + 4);
}

function trapFrameOffset(frames: TrapFrame[], frameIndex: number): number {
  let offset = 0;
  for (let index = 0; index < frameIndex; index += 1) offset += trapFrameByteLength(frames[index]!);
  return offset;
}
