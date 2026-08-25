import type { InitialAtom, TargetSite } from "../../../src/types.js";

export const DEFAULT_INITIAL_ATOMS: readonly InitialAtom[] = [
  { atomId: 1, xUm: -6, yUm: -3 }, { atomId: 2, xUm: -3, yUm: -3 },
  { atomId: 3, xUm: 0, yUm: -3 }, { atomId: 4, xUm: 3, yUm: -3 },
  { atomId: 5, xUm: 6, yUm: -3 }, { atomId: 6, xUm: -6, yUm: 1 },
  { atomId: 7, xUm: -3, yUm: 1 }, { atomId: 8, xUm: 0, yUm: 1 },
  { atomId: 9, xUm: 3, yUm: 1 }, { atomId: 10, xUm: 6, yUm: 1 },
  { atomId: 11, xUm: -3, yUm: 5 }, { atomId: 12, xUm: 3, yUm: 5 },
];

export const DEFAULT_TARGET_SITES: readonly TargetSite[] = [
  { siteId: 101, xUm: -4.5, yUm: -1.5 }, { siteId: 102, xUm: -1.5, yUm: -1.5 },
  { siteId: 103, xUm: 1.5, yUm: -1.5 }, { siteId: 104, xUm: 4.5, yUm: -1.5 },
  { siteId: 105, xUm: -4.5, yUm: 1.5 }, { siteId: 106, xUm: -1.5, yUm: 1.5 },
  { siteId: 107, xUm: 1.5, yUm: 1.5 }, { siteId: 108, xUm: 4.5, yUm: 1.5 },
  { siteId: 109, xUm: 0, yUm: 4.5 },
];
