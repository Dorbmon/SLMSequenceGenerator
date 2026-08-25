const FILE_HEADER_BYTES = 14;
const DIB_HEADER_BYTES = 40;
const PALETTE_ENTRIES = 256;
const PALETTE_BYTES = PALETTE_ENTRIES * 4;
const PIXEL_OFFSET = FILE_HEADER_BYTES + DIB_HEADER_BYTES + PALETTE_BYTES;
const MAX_DECODED_PIXELS = 16_777_216;

export interface DecodedGrayscaleBmp {
  pixels: Uint8Array;
  width: number;
  height: number;
}

/** Encodes a standards-compatible, uncompressed 8-bit indexed grayscale BMP. */
export function encodeGrayscaleBmp(
  pixels: Uint8Array | Uint16Array,
  width: number,
  height: number,
): Uint8Array {
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new Error("BMP width and height must be positive integers");
  }
  if (pixels.length !== width * height) {
    throw new Error(`BMP pixel count ${pixels.length} does not match ${width} × ${height}`);
  }
  const rowStride = (width + 3) & ~3;
  const imageBytes = rowStride * height;
  const output = new Uint8Array(PIXEL_OFFSET + imageBytes);
  const view = new DataView(output.buffer);

  output[0] = 0x42;
  output[1] = 0x4d;
  view.setUint32(2, output.length, true);
  view.setUint32(10, PIXEL_OFFSET, true);

  view.setUint32(14, DIB_HEADER_BYTES, true);
  view.setInt32(18, width, true);
  view.setInt32(22, height, true);
  view.setUint16(26, 1, true);
  view.setUint16(28, 8, true);
  view.setUint32(30, 0, true);
  view.setUint32(34, imageBytes, true);
  view.setInt32(38, 2835, true);
  view.setInt32(42, 2835, true);
  view.setUint32(46, PALETTE_ENTRIES, true);
  view.setUint32(50, PALETTE_ENTRIES, true);

  for (let value = 0; value < PALETTE_ENTRIES; value += 1) {
    const offset = FILE_HEADER_BYTES + DIB_HEADER_BYTES + value * 4;
    output[offset] = value;
    output[offset + 1] = value;
    output[offset + 2] = value;
    output[offset + 3] = 0;
  }

  const divisor = pixels instanceof Uint16Array ? 257 : 1;
  for (let outputRow = 0; outputRow < height; outputRow += 1) {
    const sourceRow = height - 1 - outputRow;
    const destination = PIXEL_OFFSET + outputRow * rowStride;
    const source = sourceRow * width;
    for (let x = 0; x < width; x += 1) {
      output[destination + x] = Math.round(pixels[source + x]! / divisor);
    }
  }
  return output;
}

/**
 * Decodes an uncompressed 8, 24, or 32-bit Windows BMP into top-down U8
 * grayscale pixels. Indexed images are resolved through their palette so an
 * exported SLM code frame round-trips without browser colour conversion.
 */
export function decodeGrayscaleBmp(source: ArrayBuffer | Uint8Array): DecodedGrayscaleBmp {
  const bytes = source instanceof Uint8Array
    ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
    : new Uint8Array(source);
  if (bytes.byteLength < FILE_HEADER_BYTES + DIB_HEADER_BYTES || bytes[0] !== 0x42 || bytes[1] !== 0x4d) {
    throw new Error("The selected file is not a supported BMP image");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const declaredSize = view.getUint32(2, true);
  const pixelOffset = view.getUint32(10, true);
  const dibBytes = view.getUint32(14, true);
  if (dibBytes < DIB_HEADER_BYTES || FILE_HEADER_BYTES + dibBytes > bytes.byteLength) {
    throw new Error("The BMP has an unsupported or truncated DIB header");
  }
  if (declaredSize !== 0 && declaredSize > bytes.byteLength) {
    throw new Error("The BMP file is truncated");
  }

  const width = view.getInt32(18, true);
  const signedHeight = view.getInt32(22, true);
  const planes = view.getUint16(26, true);
  const bitsPerPixel = view.getUint16(28, true);
  const compression = view.getUint32(30, true);
  if (width <= 0 || signedHeight === 0) throw new Error("The BMP dimensions are invalid");
  if (planes !== 1 || compression !== 0 || ![8, 24, 32].includes(bitsPerPixel)) {
    throw new Error("Only uncompressed 8, 24, or 32-bit BMP frames are supported");
  }
  const height = Math.abs(signedHeight);
  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount) || pixelCount > MAX_DECODED_PIXELS) throw new Error("The BMP dimensions are too large");
  const rowStride = Math.floor((bitsPerPixel * width + 31) / 32) * 4;
  const requiredBytes = pixelOffset + rowStride * height;
  if (pixelOffset < FILE_HEADER_BYTES + dibBytes || requiredBytes > bytes.byteLength) {
    throw new Error("The BMP pixel data is truncated");
  }

  let palette: Uint8Array | undefined;
  if (bitsPerPixel === 8) {
    const colorsUsed = view.getUint32(46, true);
    const entryCount = colorsUsed === 0 ? 256 : Math.min(colorsUsed, 256);
    const paletteOffset = FILE_HEADER_BYTES + dibBytes;
    if (paletteOffset + entryCount * 4 > pixelOffset) throw new Error("The BMP palette is truncated");
    palette = new Uint8Array(256);
    for (let index = 0; index < entryCount; index += 1) {
      const offset = paletteOffset + index * 4;
      palette[index] = grayscale(bytes[offset + 2]!, bytes[offset + 1]!, bytes[offset]!);
    }
  }

  const pixels = new Uint8Array(pixelCount);
  const topDown = signedHeight < 0;
  const bytesPerPixel = bitsPerPixel >>> 3;
  for (let y = 0; y < height; y += 1) {
    const storedRow = topDown ? y : height - 1 - y;
    const rowOffset = pixelOffset + storedRow * rowStride;
    const outputOffset = y * width;
    for (let x = 0; x < width; x += 1) {
      const offset = rowOffset + x * bytesPerPixel;
      pixels[outputOffset + x] = bitsPerPixel === 8
        ? palette![bytes[offset]!]!
        : grayscale(bytes[offset + 2]!, bytes[offset + 1]!, bytes[offset]!);
    }
  }
  return { pixels, width, height };
}

function grayscale(red: number, green: number, blue: number): number {
  return Math.round(0.2126 * red + 0.7152 * green + 0.0722 * blue);
}
