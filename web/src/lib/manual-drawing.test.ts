import { describe, expect, it } from "vitest";
import {
  appendResolvableDrawingPoints,
  eraseDrawingPoints,
  mapViewportToDrawingField,
  resampleDrawingPath,
} from "./manual-drawing.js";

describe("manual tweezer drawing geometry", () => {
  it("samples straight strokes at a physical distance independent of pointer density", () => {
    const sparse = resampleDrawingPath([
      { xUm: 0, yUm: 0 },
      { xUm: 10, yUm: 0 },
    ], 2.5);
    const dense = resampleDrawingPath([
      { xUm: 0, yUm: 0 },
      { xUm: 1, yUm: 0 },
      { xUm: 4, yUm: 0 },
      { xUm: 7, yUm: 0 },
      { xUm: 10, yUm: 0 },
    ], 2.5);

    expect(sparse).toEqual([
      { xUm: 0, yUm: 0 },
      { xUm: 2.5, yUm: 0 },
      { xUm: 5, yUm: 0 },
      { xUm: 7.5, yUm: 0 },
      { xUm: 10, yUm: 0 },
    ]);
    expect(dense).toEqual(sparse);
  });

  it("carries sampling distance around corners", () => {
    expect(resampleDrawingPath([
      { xUm: 0, yUm: 0 },
      { xUm: 3, yUm: 0 },
      { xUm: 3, yUm: 4 },
    ], 2.5)).toEqual([
      { xUm: 0, yUm: 0 },
      { xUm: 2.5, yUm: 0 },
      { xUm: 3, yUm: 2 },
    ]);
  });

  it("keeps path order while rejecting unresolved crossings and respecting the limit", () => {
    const result = appendResolvableDrawingPoints(
      [{ xUm: 0, yUm: 0 }],
      [
        { xUm: 0.2, yUm: 0.2 },
        { xUm: 2, yUm: 0 },
        { xUm: 4, yUm: 0 },
      ],
      1,
      1,
      3,
    );

    expect(result).toEqual([
      { xUm: 0, yUm: 0 },
      { xUm: 2, yUm: 0 },
      { xUm: 4, yUm: 0 },
    ]);
  });

  it("erases with a physical-radius brush", () => {
    expect(eraseDrawingPoints([
      { xUm: 0, yUm: 0 },
      { xUm: 1, yUm: 1 },
      { xUm: 3, yUm: 0 },
    ], { xUm: 0, yUm: 0 }, 1.5)).toEqual([
      { xUm: 3, yUm: 0 },
    ]);
  });

  it("maps browser coordinates into a centered field with positive Y upward", () => {
    const rectangle = { left: 10, top: 20, width: 800, height: 400 };
    expect(mapViewportToDrawingField(10, 20, rectangle, 80, 40)).toEqual({ xUm: -40, yUm: 20 });
    expect(mapViewportToDrawingField(410, 220, rectangle, 80, 40)).toEqual({ xUm: 0, yUm: 0 });
    expect(mapViewportToDrawingField(810, 420, rectangle, 80, 40)).toEqual({ xUm: 40, yUm: -20 });
  });
});
