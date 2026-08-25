import { SlmError } from "./errors.js";
import { SLM_CORE_WASM_BASE64, SLM_CORE_WASM_SHA256 } from "./wasm-binary.js";

const WASM_PAGE_BYTES = 65_536;
const MAX_WASM32_ADDRESS = 0xffff_ffff;
const EXPECTED_ABI_VERSION = 1;

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
    stage: operation.startsWith("FFT") ? "SOLVING_SLM_FRAMES" : "ASSIGNING",
    details: { status },
  });
}
