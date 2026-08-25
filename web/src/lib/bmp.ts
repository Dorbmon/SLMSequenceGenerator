const FILE_HEADER_BYTES = 14;
const DIB_HEADER_BYTES = 40;
const PALETTE_ENTRIES = 256;
const PALETTE_BYTES = PALETTE_ENTRIES * 4;
const PIXEL_OFFSET = FILE_HEADER_BYTES + DIB_HEADER_BYTES + PALETTE_BYTES;

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
