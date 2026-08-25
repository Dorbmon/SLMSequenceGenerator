import type { InitialAtom, TargetSite } from "../../../src/types.js";

export type CoordinateKind = "atom" | "target";
export type CoordinatePoint = InitialAtom | TargetSite;

export function cloneAtoms(points: readonly InitialAtom[]): InitialAtom[] {
  return points.map((point) => ({ ...point }));
}

export function cloneTargets(points: readonly TargetSite[]): TargetSite[] {
  return points.map((point) => ({ ...point }));
}

export function parseAtomList(raw: string): InitialAtom[] {
  return parsePointList(raw, "atom") as InitialAtom[];
}

export function parseTargetList(raw: string): TargetSite[] {
  return parsePointList(raw, "target") as TargetSite[];
}

export function serializePoints(points: readonly CoordinatePoint[]): string {
  return JSON.stringify(points, null, 2);
}

export function nextPointId(points: readonly CoordinatePoint[], kind: CoordinateKind): number {
  const property = kind === "atom" ? "atomId" : "siteId";
  const identifiers = points
    .map((point) => point[property as keyof CoordinatePoint])
    .filter((value): value is number => Number.isSafeInteger(value) && Number(value) >= 0);
  return identifiers.length === 0 ? 1 : Math.max(...identifiers) + 1;
}

export function coordinateLabel(point: CoordinatePoint, kind: CoordinateKind, index: number): string {
  if (kind === "atom") return `Initial atom ${(point as InitialAtom).atomId ?? index + 1}`;
  return `Target site ${(point as TargetSite).siteId ?? index + 1}`;
}

function parsePointList(raw: string, kind: CoordinateKind): CoordinatePoint[] {
  const value: unknown = JSON.parse(raw);
  const property = kind === "atom" ? "initialAtoms" : "targetSites";
  const list = Array.isArray(value)
    ? value
    : isRecord(value) && Array.isArray(value[property])
      ? value[property]
      : isRecord(value) && Array.isArray(value.points)
        ? value.points
        : null;
  if (!list) throw new Error(`${kind === "atom" ? "Initial atoms" : "Target sites"} must be a JSON array`);

  return list.map((entry, index) => {
    const record = isRecord(entry) ? entry : {};
    const xUm: unknown = Array.isArray(entry) ? entry[0] : record.xUm ?? record.x;
    const yUm: unknown = Array.isArray(entry) ? entry[1] : record.yUm ?? record.y;
    if (typeof xUm !== "number" || typeof yUm !== "number" || !Number.isFinite(xUm) || !Number.isFinite(yUm)) {
      throw new Error(`Point ${index + 1} has invalid coordinates`);
    }
    if (kind === "atom") {
      const point = { ...record, xUm, yUm } as InitialAtom;
      point.atomId ??= index + 1;
      return point;
    }
    const point = { ...record, xUm, yUm } as TargetSite;
    point.siteId ??= index + 1;
    return point;
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
