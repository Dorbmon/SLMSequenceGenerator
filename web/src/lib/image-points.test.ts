import { describe, expect, it } from "vitest";
import {
  detectImagePoints,
  mapImagePointsToField,
  type SpotDetectionOptions,
} from "./image-points.js";

const defaults: SpotDetectionOptions = {
  polarity: "BRIGHT",
  threshold: 0.6,
  minimumAreaPx: 2,
  maximumPoints: 64,
};

describe("target-field image detection", () => {
  it("finds bright connected spots and rejects isolated noise", () => {
    const image = rgbaImage(12, 8, 0);
    fill(image, 12, 2, 2, 2, 2, 255);
    fill(image, 12, 8, 4, 3, 2, 220);
    setPixel(image, 12, 6, 1, 255);

    const result = detectImagePoints(image, 12, 8, defaults);

    expect(result.points).toHaveLength(2);
    expect(result.discardedSmallComponents).toBe(1);
    expect(result.points[0]).toMatchObject({ xPx: 2.5, yPx: 2.5, areaPx: 4, peakSignal: 255 });
    expect(result.points[1]?.xPx).toBeCloseTo(9);
    expect(result.points[1]?.yPx).toBeCloseTo(4.5);
  });

  it("supports dark spots and retains the strongest points when limited", () => {
    const image = rgbaImage(10, 6, 255);
    fill(image, 10, 1, 1, 2, 2, 0);
    fill(image, 10, 7, 3, 2, 2, 80);

    const result = detectImagePoints(image, 10, 6, {
      ...defaults,
      polarity: "DARK",
      maximumPoints: 1,
    });

    expect(result.points).toHaveLength(1);
    expect(result.points[0]?.xPx).toBeCloseTo(1.5);
    expect(result.points[0]?.yPx).toBeCloseTo(1.5);
  });

  it("ignores transparent pixels when detecting dark spots", () => {
    const image = rgbaImage(8, 8, 255);
    for (let index = 0; index < 8; index += 1) image[index * 4 + 3] = 0;
    fill(image, 8, 3, 3, 2, 2, 0);

    const result = detectImagePoints(image, 8, 8, { ...defaults, polarity: "DARK" });

    expect(result.points).toHaveLength(1);
    expect(result.points[0]).toMatchObject({ xPx: 3.5, yPx: 3.5, areaPx: 4 });
  });

  it("maps pixel centers into a centered micrometer field with +y up", () => {
    const points = mapImagePointsToField([
      { xPx: 0, yPx: 0, areaPx: 2, peakSignal: 255, integratedSignal: 10 },
      { xPx: 5, yPx: 4, areaPx: 2, peakSignal: 255, integratedSignal: 10 },
      { xPx: 10, yPx: 8, areaPx: 2, peakSignal: 255, integratedSignal: 10 },
    ], 11, 9, 20, 12);

    expect(points.map(({ xUm, yUm }) => ({ xUm, yUm }))).toEqual([
      { xUm: -10, yUm: 6 },
      { xUm: 0, yUm: 0 },
      { xUm: 10, yUm: -6 },
    ]);
  });
});

function rgbaImage(width: number, height: number, intensity: number): Uint8ClampedArray {
  const image = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < width * height; index += 1) {
    const offset = index * 4;
    image[offset] = intensity;
    image[offset + 1] = intensity;
    image[offset + 2] = intensity;
    image[offset + 3] = 255;
  }
  return image;
}

function fill(
  image: Uint8ClampedArray,
  imageWidth: number,
  x: number,
  y: number,
  width: number,
  height: number,
  intensity: number,
): void {
  for (let offsetY = 0; offsetY < height; offsetY += 1) {
    for (let offsetX = 0; offsetX < width; offsetX += 1) {
      setPixel(image, imageWidth, x + offsetX, y + offsetY, intensity);
    }
  }
}

function setPixel(image: Uint8ClampedArray, width: number, x: number, y: number, intensity: number): void {
  const offset = (y * width + x) * 4;
  image[offset] = intensity;
  image[offset + 1] = intensity;
  image[offset + 2] = intensity;
}
