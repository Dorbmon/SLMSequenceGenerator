import { describe, expect, it } from "vitest";
import {
  opticalTweezersToFrame,
  parseOpticalTweezers,
  serializeOpticalTweezers,
} from "./tweezers.js";

describe("optical tweezer input", () => {
  it("parses object aliases and tuple input", () => {
    expect(parseOpticalTweezers(JSON.stringify({
      traps: [
        { id: 7, x: -2, y: 3, phase: 1.25 },
        [4, -5, -0.5, 0.8],
      ],
    }))).toEqual([
      { trapId: 7, xUm: -2, yUm: 3, phaseRad: 1.25, intensity: 1 },
      { trapId: 2, xUm: 4, yUm: -5, phaseRad: -0.5, intensity: 0.8 },
    ]);
  });

  it("rejects duplicate IDs and invalid phase values", () => {
    expect(() => parseOpticalTweezers('[{"trapId":1,"xUm":0,"yUm":0,"phaseRad":0},{"trapId":1,"xUm":1,"yUm":1,"phaseRad":1}]')).toThrow("duplicated");
    expect(() => parseOpticalTweezers('[[0,0,"pi"]]')).toThrow("invalid phase");
  });

  it("converts phases to target phases without changing them", () => {
    const parsed = parseOpticalTweezers(serializeOpticalTweezers([
      { trapId: 3, xUm: 1, yUm: -2, phaseRad: Math.PI * 1.5, intensity: 0.6 },
    ]));
    const frame = opticalTweezersToFrame(parsed);
    expect(frame.traps[0]).toMatchObject({
      trapId: 3,
      xUm: 1,
      yUm: -2,
      targetPhaseRad: Math.PI * 1.5,
      intensity: 0.6,
    });
  });
});
