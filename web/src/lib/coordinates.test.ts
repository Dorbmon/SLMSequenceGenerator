import { describe, expect, it } from "vitest";
import { nextPointId, parseAtomList, parseTargetList, serializePoints } from "./coordinates.js";

describe("coordinate JSON", () => {
  it("accepts tuple arrays and assigns stable identifiers", () => {
    expect(parseAtomList("[[1, 2], [-3, 4]]")).toEqual([
      { atomId: 1, xUm: 1, yUm: 2 },
      { atomId: 2, xUm: -3, yUm: 4 },
    ]);
  });

  it("accepts request wrappers and preserves point metadata", () => {
    expect(parseTargetList(JSON.stringify({
      targetSites: [{ siteId: 9, x: 2, y: 3, required: false }],
    }))).toEqual([{ siteId: 9, x: 2, y: 3, xUm: 2, yUm: 3, required: false }]);
  });

  it("rejects malformed and non-finite coordinates", () => {
    expect(() => parseAtomList("{}")).toThrow("must be a JSON array");
    expect(() => parseAtomList('[{"xUm": 1, "yUm": "2"}]')).toThrow("invalid coordinates");
  });

  it("allocates the next identifier and serializes canonical JSON", () => {
    const atoms = parseAtomList('[{"atomId": 4, "xUm": 0, "yUm": 0}]');
    expect(nextPointId(atoms, "atom")).toBe(5);
    expect(JSON.parse(serializePoints(atoms))).toEqual(atoms);
  });
});
