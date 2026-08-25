import type { ForbiddenRegion, Point2D } from "./types.js";
import { clamp } from "./util.js";

export interface Segment2D {
  start: Point2D;
  end: Point2D;
}

export function distanceSquared(a: Point2D, b: Point2D): number {
  const dx = a.xUm - b.xUm;
  const dy = a.yUm - b.yUm;
  return dx * dx + dy * dy;
}

export function distance(a: Point2D, b: Point2D): number {
  return Math.sqrt(distanceSquared(a, b));
}

export function lerpPoint(a: Point2D, b: Point2D, t: number): Point2D {
  return { xUm: a.xUm + (b.xUm - a.xUm) * t, yUm: a.yUm + (b.yUm - a.yUm) * t };
}

export function distancePointToSegment(point: Point2D, segment: Segment2D): number {
  const dx = segment.end.xUm - segment.start.xUm;
  const dy = segment.end.yUm - segment.start.yUm;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return distance(point, segment.start);
  const t = clamp(
    ((point.xUm - segment.start.xUm) * dx + (point.yUm - segment.start.yUm) * dy) / lengthSquared,
    0,
    1,
  );
  return distance(point, lerpPoint(segment.start, segment.end, t));
}

export function orientation(a: Point2D, b: Point2D, c: Point2D): number {
  return (b.xUm - a.xUm) * (c.yUm - a.yUm) - (b.yUm - a.yUm) * (c.xUm - a.xUm);
}

function onSegment(a: Point2D, b: Point2D, point: Point2D, epsilon = 1e-10): boolean {
  return (
    point.xUm >= Math.min(a.xUm, b.xUm) - epsilon &&
    point.xUm <= Math.max(a.xUm, b.xUm) + epsilon &&
    point.yUm >= Math.min(a.yUm, b.yUm) - epsilon &&
    point.yUm <= Math.max(a.yUm, b.yUm) + epsilon
  );
}

export function segmentsIntersect(a: Segment2D, b: Segment2D, epsilon = 1e-10): boolean {
  const o1 = orientation(a.start, a.end, b.start);
  const o2 = orientation(a.start, a.end, b.end);
  const o3 = orientation(b.start, b.end, a.start);
  const o4 = orientation(b.start, b.end, a.end);

  if (((o1 > epsilon && o2 < -epsilon) || (o1 < -epsilon && o2 > epsilon)) &&
      ((o3 > epsilon && o4 < -epsilon) || (o3 < -epsilon && o4 > epsilon))) {
    return true;
  }
  return (
    (Math.abs(o1) <= epsilon && onSegment(a.start, a.end, b.start, epsilon)) ||
    (Math.abs(o2) <= epsilon && onSegment(a.start, a.end, b.end, epsilon)) ||
    (Math.abs(o3) <= epsilon && onSegment(b.start, b.end, a.start, epsilon)) ||
    (Math.abs(o4) <= epsilon && onSegment(b.start, b.end, a.end, epsilon))
  );
}

export function segmentDistance(a: Segment2D, b: Segment2D): number {
  if (segmentsIntersect(a, b)) return 0;
  return Math.min(
    distancePointToSegment(a.start, b),
    distancePointToSegment(a.end, b),
    distancePointToSegment(b.start, a),
    distancePointToSegment(b.end, a),
  );
}

export function pointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const current = polygon[i]!;
    const previous = polygon[j]!;
    if (onSegment(previous, current, point, 1e-10)) return true;
    const crosses =
      (current.yUm > point.yUm) !== (previous.yUm > point.yUm) &&
      point.xUm <
        ((previous.xUm - current.xUm) * (point.yUm - current.yUm)) /
          (previous.yUm - current.yUm) +
          current.xUm;
    if (crosses) inside = !inside;
  }
  return inside;
}

export function regionPolygon(region: ForbiddenRegion): Point2D[] {
  if (region.type === "polygon") {
    const points: Point2D[] = [];
    for (let index = 0; index + 1 < region.coordinates.length; index += 2) {
      points.push({ xUm: region.coordinates[index]!, yUm: region.coordinates[index + 1]! });
    }
    return points;
  }
  if (region.type === "axisAlignedBox") {
    const [xMin, yMin, xMax, yMax] = region.coordinates;
    return [
      { xUm: xMin!, yUm: yMin! },
      { xUm: xMax!, yUm: yMin! },
      { xUm: xMax!, yUm: yMax! },
      { xUm: xMin!, yUm: yMax! },
    ];
  }
  const [x, y, radius] = region.coordinates;
  const count = 32;
  const points: Point2D[] = [];
  for (let index = 0; index < count; index += 1) {
    const angle = (index / count) * Math.PI * 2;
    points.push({ xUm: x! + Math.cos(angle) * radius!, yUm: y! + Math.sin(angle) * radius! });
  }
  return points;
}

export function distancePointToPolygon(point: Point2D, polygon: Point2D[]): number {
  if (polygon.length === 0) return Number.POSITIVE_INFINITY;
  if (pointInPolygon(point, polygon)) return 0;
  let result = Number.POSITIVE_INFINITY;
  for (let index = 0; index < polygon.length; index += 1) {
    const segment = {
      start: polygon[index]!,
      end: polygon[(index + 1) % polygon.length]!,
    };
    result = Math.min(result, distancePointToSegment(point, segment));
  }
  return result;
}

/** Returns true when a point is in an inflated forbidden region. */
export function pointInForbiddenRegion(point: Point2D, region: ForbiddenRegion): boolean {
  const clearance = Math.max(0, region.clearanceUm ?? 0);
  if (region.type === "circle") {
    const [x, y, radius] = region.coordinates;
    return Math.hypot(point.xUm - x!, point.yUm - y!) <= radius! + clearance;
  }
  const polygon = regionPolygon(region);
  return distancePointToPolygon(point, polygon) <= clearance || pointInPolygon(point, polygon);
}

export function segmentIntersectsForbiddenRegion(
  segment: Segment2D,
  region: ForbiddenRegion,
  samples = 32,
): boolean {
  if (region.type === "circle") {
    const [x, y, radius] = region.coordinates;
    return distancePointToSegment(
      { xUm: x!, yUm: y! },
      segment,
    ) <= radius! + Math.max(0, region.clearanceUm ?? 0);
  }
  if (pointInForbiddenRegion(segment.start, region) || pointInForbiddenRegion(segment.end, region)) {
    return true;
  }
  const polygon = regionPolygon(region);
  for (let index = 0; index < polygon.length; index += 1) {
    const edge = { start: polygon[index]!, end: polygon[(index + 1) % polygon.length]! };
    if (segmentsIntersect(segment, edge)) return true;
    const clearance = Math.max(0, region.clearanceUm ?? 0);
    if (clearance > 0 && segmentDistance(segment, edge) <= clearance) return true;
  }
  for (let index = 1; index < samples; index += 1) {
    if (pointInForbiddenRegion(segmentPoint(segment, index / samples), region)) return true;
  }
  return false;
}

export function segmentPoint(segment: Segment2D, fraction: number): Point2D {
  return lerpPoint(segment.start, segment.end, fraction);
}

export function pathIntersectsForbiddenRegion(path: Point2D[], regions: ForbiddenRegion[]): boolean {
  for (const point of path) {
    if (regions.some((region) => pointInForbiddenRegion(point, region))) return true;
  }
  for (let index = 0; index + 1 < path.length; index += 1) {
    const segment = { start: path[index]!, end: path[index + 1]! };
    if (regions.some((region) => segmentIntersectsForbiddenRegion(segment, region))) return true;
  }
  return false;
}

export function pointClearOfStaticAtoms(point: Point2D, staticPoints: Point2D[], clearance: number): boolean {
  return staticPoints.every((staticPoint) => distance(point, staticPoint) >= clearance - 1e-9);
}

export function pathClearOfStaticAtoms(path: Point2D[], staticPoints: Point2D[], clearance: number): boolean {
  for (const point of staticPoints) {
    for (let index = 0; index + 1 < path.length; index += 1) {
      if (distancePointToSegment(point, { start: path[index]!, end: path[index + 1]! }) < clearance - 1e-9) {
        return false;
      }
    }
    if (path.some((pathPoint) => distance(pathPoint, point) < clearance - 1e-9)) return false;
  }
  return true;
}

export function boundingBox(points: Point2D[], padding = 0): {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
} {
  if (points.length === 0) return { xMin: -padding, xMax: padding, yMin: -padding, yMax: padding };
  return {
    xMin: Math.min(...points.map((point) => point.xUm)) - padding,
    xMax: Math.max(...points.map((point) => point.xUm)) + padding,
    yMin: Math.min(...points.map((point) => point.yUm)) - padding,
    yMax: Math.max(...points.map((point) => point.yUm)) + padding,
  };
}
