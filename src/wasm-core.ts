import { SlmError } from "./errors.js";
import { SLM_CORE_WASM_BASE64, SLM_CORE_WASM_SHA256 } from "./wasm-binary.js";

const WASM_PAGE_BYTES = 65_536;
const MAX_WASM32_ADDRESS = 0xffff_ffff;
const EXPECTED_ABI_VERSION = 2;

interface CoreExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  core_abi_version(): number;
  core_fft_1d(
    realPointer: number,
    imagPointer: number,
    scratchRealPointer: number,
    scratchImagPointer: number,
    length: number,
    inverse: number,
  ): number;
  core_fft_2d(
    realPointer: number,
    imagPointer: number,
    width: number,
    height: number,
    workRealPointer: number,
    workImagPointer: number,
    scratchRealPointer: number,
    scratchImagPointer: number,
    inverse: number,
  ): number;
  core_nudft_sample_targets(
    fieldRealPointer: number,
    fieldImagPointer: number,
    width: number,
    height: number,
    targetXPointer: number,
    targetYPointer: number,
    targetCount: number,
    outputRealPointer: number,
    outputImagPointer: number,
  ): number;
  core_nudft_synthesize_phase(
    targetXPointer: number,
    targetYPointer: number,
    targetAmplitudePointer: number,
    targetPhasePointer: number,
    targetCount: number,
    width: number,
    height: number,
    phasePointer: number,
    scratchRealPointer: number,
    scratchImagPointer: number,
    cancellationThreshold: number,
    deterministicSeed: number,
    deterministicFallback: number,
  ): number;
  core_hungarian_workspace_bytes(rows: number, columns: number): number;
  core_hungarian(
    costPointer: number,
    rows: number,
    columns: number,
    assignmentPointer: number,
    workspacePointer: number,
  ): number;
}

export interface WasmCoreInfo {
  backend: "webassembly";
  buildId: string;
  abiVersion: number;
  moduleBytes: number;
}

function decodeBase64(value: string): Uint8Array<ArrayBuffer> {
  const decoded = globalThis.atob(value);
  const bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return bytes;
}

function instantiateCore(): { exports: CoreExports; byteLength: number } {
  try {
    const bytes = decodeBase64(SLM_CORE_WASM_BASE64);
    const module = new WebAssembly.Module(bytes);
    const instance = new WebAssembly.Instance(module, {
      env: {
        atan2: Math.atan2,
        cos: Math.cos,
        sin: Math.sin,
      },
    });
    const exports = instance.exports as CoreExports;
    if (!(exports.memory instanceof WebAssembly.Memory) ||
        typeof exports.core_abi_version !== "function" ||
        exports.core_abi_version() !== EXPECTED_ABI_VERSION) {
      throw new Error("Wasm core ABI mismatch");
    }
    return { exports, byteLength: bytes.byteLength };
  } catch (cause) {
    throw new SlmError("INTERNAL_ERROR", "Unable to initialize the WebAssembly core", {
      stage: "CREATED",
      cause,
    });
  }
}

const instantiated = instantiateCore();
const core = instantiated.exports;
const arenaBase = alignUp(core.memory.buffer.byteLength, 8);

export const WASM_CORE_BUILD_ID = `rust-wasm-core-abi${EXPECTED_ABI_VERSION}-${SLM_CORE_WASM_SHA256.slice(0, 12)}`;

const CORE_INFO: WasmCoreInfo = Object.freeze({
  backend: "webassembly",
  buildId: WASM_CORE_BUILD_ID,
  abiVersion: EXPECTED_ABI_VERSION,
  moduleBytes: instantiated.byteLength,
});

export function getWasmCoreInfo(): WasmCoreInfo {
  return CORE_INFO;
}

export function wasmFft1d(real: Float64Array, imag: Float64Array, inverse: boolean): void {
  const length = real.length;
  if (length === 0) return;
  const arrayBytes = checkedProduct(length, Float64Array.BYTES_PER_ELEMENT);
  const totalBytes = checkedProduct(arrayBytes, 4);
  const base = reserveArena(totalBytes);
  const realPointer = base;
  const imagPointer = realPointer + arrayBytes;
  const scratchRealPointer = imagPointer + arrayBytes;
  const scratchImagPointer = scratchRealPointer + arrayBytes;
  const memory = new Float64Array(core.memory.buffer);
  memory.set(real, realPointer / Float64Array.BYTES_PER_ELEMENT);
  memory.set(imag, imagPointer / Float64Array.BYTES_PER_ELEMENT);
  const status = core.core_fft_1d(
    realPointer,
    imagPointer,
    scratchRealPointer,
    scratchImagPointer,
    length,
    inverse ? 1 : 0,
  );
  assertCoreStatus(status, "FFT-1D");
  real.set(memory.subarray(realPointer / 8, realPointer / 8 + length));
  imag.set(memory.subarray(imagPointer / 8, imagPointer / 8 + length));
}

export function wasmFft2d(
  real: Float64Array,
  imag: Float64Array,
  width: number,
  height: number,
  inverse: boolean,
): void {
  const length = checkedProduct(width, height);
  const fieldBytes = checkedProduct(length, Float64Array.BYTES_PER_ELEMENT);
  const workLength = Math.max(width, height);
  const workBytes = checkedProduct(workLength, Float64Array.BYTES_PER_ELEMENT);
  const totalBytes = checkedSum(checkedProduct(fieldBytes, 2), checkedProduct(workBytes, 4));
  const base = reserveArena(totalBytes);
  const realPointer = base;
  const imagPointer = realPointer + fieldBytes;
  const workRealPointer = imagPointer + fieldBytes;
  const workImagPointer = workRealPointer + workBytes;
  const scratchRealPointer = workImagPointer + workBytes;
  const scratchImagPointer = scratchRealPointer + workBytes;
  const memory = new Float64Array(core.memory.buffer);
  memory.set(real, realPointer / 8);
  memory.set(imag, imagPointer / 8);
  const status = core.core_fft_2d(
    realPointer,
    imagPointer,
    width,
    height,
    workRealPointer,
    workImagPointer,
    scratchRealPointer,
    scratchImagPointer,
    inverse ? 1 : 0,
  );
  assertCoreStatus(status, "FFT-2D");
  real.set(memory.subarray(realPointer / 8, realPointer / 8 + length));
  imag.set(memory.subarray(imagPointer / 8, imagPointer / 8 + length));
}

/** Exact unnormalised 2D DFT values at arbitrary FFT-bin coordinates. */
export function wasmNudftSampleTargets(
  fieldReal: Float64Array,
  fieldImag: Float64Array,
  width: number,
  height: number,
  targetX: Float64Array,
  targetY: Float64Array,
): { real: Float64Array; imag: Float64Array } {
  const pixelCount = checkedProduct(width, height);
  if (fieldReal.length !== pixelCount || fieldImag.length !== pixelCount) {
    throw new SlmError("INVALID_ARGUMENT", "NUDFT field dimensions do not match its buffers", {
      stage: "SOLVING_SLM_FRAMES",
      details: { width, height, realLength: fieldReal.length, imagLength: fieldImag.length },
    });
  }
  if (targetX.length !== targetY.length) {
    throw new SlmError("INVALID_ARGUMENT", "NUDFT target coordinate arrays have different lengths", {
      stage: "SOLVING_SLM_FRAMES",
    });
  }
  const targetCount = targetX.length;
  const outputReal = new Float64Array(targetCount);
  const outputImag = new Float64Array(targetCount);
  if (targetCount === 0) return { real: outputReal, imag: outputImag };

  const fieldBytes = checkedProduct(pixelCount, Float64Array.BYTES_PER_ELEMENT);
  const targetBytes = checkedProduct(targetCount, Float64Array.BYTES_PER_ELEMENT);
  const totalBytes = checkedSum(checkedProduct(fieldBytes, 2), checkedProduct(targetBytes, 4));
  const base = reserveArena(totalBytes);
  const fieldRealPointer = base;
  const fieldImagPointer = fieldRealPointer + fieldBytes;
  const targetXPointer = fieldImagPointer + fieldBytes;
  const targetYPointer = targetXPointer + targetBytes;
  const outputRealPointer = targetYPointer + targetBytes;
  const outputImagPointer = outputRealPointer + targetBytes;
  const memory = new Float64Array(core.memory.buffer);
  memory.set(fieldReal, fieldRealPointer / 8);
  memory.set(fieldImag, fieldImagPointer / 8);
  memory.set(targetX, targetXPointer / 8);
  memory.set(targetY, targetYPointer / 8);
  const status = core.core_nudft_sample_targets(
    fieldRealPointer,
    fieldImagPointer,
    width,
    height,
    targetXPointer,
    targetYPointer,
    targetCount,
    outputRealPointer,
    outputImagPointer,
  );
  assertCoreStatus(status, "NUDFT sampling");
  outputReal.set(memory.subarray(outputRealPointer / 8, outputRealPointer / 8 + targetCount));
  outputImag.set(memory.subarray(outputImagPointer / 8, outputImagPointer / 8 + targetCount));
  return { real: outputReal, imag: outputImag };
}

/** Exact adjoint trap synthesis, writing the resulting phase in place. */
export function wasmNudftSynthesizePhase(
  targetX: Float64Array,
  targetY: Float64Array,
  targetAmplitude: Float64Array,
  targetPhase: Float64Array,
  width: number,
  height: number,
  phase: Float64Array,
  cancellationThreshold: number,
  deterministicSeed: number,
  deterministicFallback: boolean,
): void {
  const targetCount = targetX.length;
  if (targetY.length !== targetCount || targetAmplitude.length !== targetCount || targetPhase.length !== targetCount) {
    throw new SlmError("INVALID_ARGUMENT", "NUDFT target arrays have different lengths", {
      stage: "SOLVING_SLM_FRAMES",
    });
  }
  const pixelCount = checkedProduct(width, height);
  if (phase.length !== pixelCount) {
    throw new SlmError("INVALID_ARGUMENT", "NUDFT phase dimensions do not match its buffer", {
      stage: "SOLVING_SLM_FRAMES",
      details: { width, height, phaseLength: phase.length },
    });
  }
  const targetBytes = checkedProduct(targetCount, Float64Array.BYTES_PER_ELEMENT);
  const fieldBytes = checkedProduct(pixelCount, Float64Array.BYTES_PER_ELEMENT);
  const totalBytes = checkedSum(checkedProduct(targetBytes, 4), checkedProduct(fieldBytes, 3));
  const base = reserveArena(totalBytes);
  const targetXPointer = base;
  const targetYPointer = targetXPointer + targetBytes;
  const targetAmplitudePointer = targetYPointer + targetBytes;
  const targetPhasePointer = targetAmplitudePointer + targetBytes;
  const phasePointer = targetPhasePointer + targetBytes;
  const scratchRealPointer = phasePointer + fieldBytes;
  const scratchImagPointer = scratchRealPointer + fieldBytes;
  const memory = new Float64Array(core.memory.buffer);
  memory.set(targetX, targetXPointer / 8);
  memory.set(targetY, targetYPointer / 8);
  memory.set(targetAmplitude, targetAmplitudePointer / 8);
  memory.set(targetPhase, targetPhasePointer / 8);
  memory.set(phase, phasePointer / 8);
  const status = core.core_nudft_synthesize_phase(
    targetXPointer,
    targetYPointer,
    targetAmplitudePointer,
    targetPhasePointer,
    targetCount,
    width,
    height,
    phasePointer,
    scratchRealPointer,
    scratchImagPointer,
    cancellationThreshold,
    deterministicSeed >>> 0,
    deterministicFallback ? 1 : 0,
  );
  assertCoreStatus(status, "NUDFT synthesis");
  phase.set(memory.subarray(phasePointer / 8, phasePointer / 8 + pixelCount));
}

export function wasmHungarianSolve(costMatrix: number[][]): { assignment: number[]; feasible: boolean } {
  const rows = costMatrix.length;
  const columns = costMatrix[0]?.length ?? 0;
  const matrixLength = checkedProduct(rows, columns);
  const matrixBytes = checkedProduct(matrixLength, Float64Array.BYTES_PER_ELEMENT);
  const assignmentBytes = checkedProduct(rows, Int32Array.BYTES_PER_ELEMENT);
  const workspaceBytes = core.core_hungarian_workspace_bytes(rows, columns) >>> 0;
  if (workspaceBytes === 0) {
    throw new SlmError("INTERNAL_ERROR", "The WebAssembly assignment workspace is too large", {
      stage: "ASSIGNING",
    });
  }
  const assignmentPointer = alignUp(arenaBase + matrixBytes, 8);
  const workspacePointer = alignUp(checkedSum(assignmentPointer, assignmentBytes), 8);
  const end = checkedSum(workspacePointer, workspaceBytes);
  reserveArena(end - arenaBase);
  const memory = new Float64Array(core.memory.buffer);
  let matrixOffset = arenaBase / 8;
  for (const row of costMatrix) {
    memory.set(row, matrixOffset);
    matrixOffset += columns;
  }
  const status = core.core_hungarian(
    arenaBase,
    rows,
    columns,
    assignmentPointer,
    workspacePointer,
  );
  if (status < 0) assertCoreStatus(status, "Hungarian assignment");
  const assignmentView = new Int32Array(core.memory.buffer, assignmentPointer, rows);
  return { assignment: Array.from(assignmentView), feasible: status === 1 };
}

function reserveArena(bytes: number): number {
  const requiredEnd = checkedSum(arenaBase, bytes);
  if (requiredEnd > MAX_WASM32_ADDRESS) {
    throw new SlmError("INVALID_ARGUMENT", "Input exceeds the WebAssembly 32-bit address space", {
      stage: "VALIDATING",
      details: { bytes },
    });
  }
  const currentBytes = core.memory.buffer.byteLength;
  if (requiredEnd > currentBytes) {
    const pages = Math.ceil((requiredEnd - currentBytes) / WASM_PAGE_BYTES);
    try {
      core.memory.grow(pages);
    } catch (cause) {
      throw new SlmError("INTERNAL_ERROR", "Unable to grow WebAssembly linear memory", {
        stage: "CREATED",
        details: { bytes },
        cause,
      });
    }
  }
  return arenaBase;
}

function checkedProduct(first: number, second: number): number {
  const result = first * second;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new SlmError("INVALID_ARGUMENT", "Input size exceeds the supported range", {
      stage: "VALIDATING",
      details: { first, second },
    });
  }
  return result;
}

function checkedSum(first: number, second: number): number {
  const result = first + second;
  if (!Number.isSafeInteger(result) || result < 0) {
    throw new SlmError("INVALID_ARGUMENT", "Input size exceeds the supported range", {
      stage: "VALIDATING",
      details: { first, second },
    });
  }
  return result;
}

function alignUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function assertCoreStatus(status: number, operation: string): void {
  if (status === 0) return;
  throw new SlmError("INTERNAL_ERROR", `${operation} failed in the WebAssembly core`, {
    stage: operation.startsWith("FFT") || operation.startsWith("NUDFT") ? "SOLVING_SLM_FRAMES" : "ASSIGNING",
    details: { status },
  });
}
