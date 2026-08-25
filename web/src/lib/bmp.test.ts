import { describe, expect, it } from "vitest";
import { decodeGrayscaleBmp, encodeGrayscaleBmp } from "./bmp.js";

describe("encodeGrayscaleBmp", () => {
  it("writes an 8-bit indexed BMP with padded bottom-up rows", () => {
    const bmp = encodeGrayscaleBmp(new Uint8Array([
      1, 2, 3,
      4, 5, 6,
    ]), 3, 2);
    const view = new DataView(bmp.buffer);
    expect(String.fromCharCode(bmp[0]!, bmp[1]!)).toBe("BM");
    expect(view.getUint32(2, true)).toBe(1086);
    expect(view.getUint32(10, true)).toBe(1078);
    expect(view.getInt32(18, true)).toBe(3);
    expect(view.getInt32(22, true)).toBe(2);
    expect(view.getUint16(28, true)).toBe(8);
    expect([...bmp.slice(1078, 1086)]).toEqual([4, 5, 6, 0, 1, 2, 3, 0]);
  });

  it("scales UINT16 display codes to the grayscale palette", () => {
    const bmp = encodeGrayscaleBmp(new Uint16Array([0, 32768, 65535, 257]), 4, 1);
    expect([...bmp.slice(1078)]).toEqual([0, 128, 255, 1]);
    expect([...bmp.slice(54 + 255 * 4, 54 + 256 * 4)]).toEqual([255, 255, 255, 0]);
  });

  it("rejects mismatched frame dimensions", () => {
    expect(() => encodeGrayscaleBmp(new Uint8Array(3), 2, 2)).toThrow(/pixel count/i);
  });

  it("round-trips an indexed grayscale SLM frame", () => {
    const source = new Uint8Array([
      0, 17, 255,
      93, 128, 201,
    ]);
    const decoded = decodeGrayscaleBmp(encodeGrayscaleBmp(source, 3, 2));
    expect(decoded.width).toBe(3);
    expect(decoded.height).toBe(2);
    expect(decoded.pixels).toEqual(source);
  });

  it("rejects truncated BMP pixel data", () => {
    const bmp = encodeGrayscaleBmp(new Uint8Array(16), 4, 4);
    expect(() => decodeGrayscaleBmp(bmp.slice(0, -2))).toThrow(/truncated/i);
  });
});
