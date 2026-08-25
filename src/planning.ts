import { SlmError } from "./errors.js";
import {
  boundingBox,
  distance,
  distancePointToSegment,
  pathIntersectsForbiddenRegion,
  pointClearOfStaticAtoms,
  segmentDistance,
} from "./geometry.js";
import type {
  AtomAssignment,
  AtomTrajectory,
  InitialAtom,
  MotionPlan,
  NormalizedInput,
  PlannedPath,
  Point2D,
} from "./types.js";
import { parameterizeTrajectories, validateContinuousTrajectories } from "./trajectory.js";

interface Conflict {
  first: number;
  second: number;
  time: number;
  kind: "vertex" | "edge" | "near-edge";
}

interface GridNode {
  point: Point2D;
  neighbors: number[];
}

interface Constraint {
  agent: number;
  time: number;
  node?: number;
  from?: number;
  to?: number;
}

interface GridPath {
  nodes: number[];
  ticks: number[];
}

interface CbsNode {
  constraints: Constraint[];
  paths: GridPath[];
  cost: number;
}

const EPSILON = 1e-9;

export function atomTrapIds(input: NormalizedInput): Map<number, number> {
  const used = new Set(input.staticTraps.map((trap) => trap.trapId));
  const result = new Map<number, number>();
  let next = Math.max(-1, ...used, ...input.initialAtoms.map((atom) => atom.atomId)) + 1;
  for (const atom of input.initialAtoms) {
    let trapId = atom.atomId;
    if (used.has(trapId)) {
      while (used.has(next)) next += 1;
      if (next > 0xffffffff) throw new SlmError("INVALID_ARGUMENT", "No uint32 trap identifier is available", { stage: "PLANNING" });
      trapId = next;
      next += 1;
    }
    used.add(trapId);
    result.set(atom.atomId, trapId);
  }
  return result;
}

export function buildDirectPaths(input: NormalizedInput, assignments: AtomAssignment[]): PlannedPath[] {
  const trapIds = atomTrapIds(input);
  return assignments.map((assignment) => {
    const atom = input.initialAtoms[assignment.sourceIndex];
    if (!atom) throw new SlmError("INTERNAL_ERROR", `Missing atom at source index ${assignment.sourceIndex}`, { stage: "PLANNING" });
    const destination = destinationForAssignment(input, assignment, atom);
    const moving = distance(atom, destination) > EPSILON;
    return {
      atomId: atom.atomId,
      trapId: trapIds.get(atom.atomId)!,
      goalSiteId: assignment.targetSiteId,
      disposition: assignment.disposition,
      waypointsUm: moving ? [{ xUm: atom.xUm, yUm: atom.yUm }, destination] : [{ xUm: atom.xUm, yUm: atom.yUm }],
      discreteTicks: moving ? [0, 1] : [0],
    };
  });
}

export function planMotion(input: NormalizedInput, assignments: AtomAssignment[]): MotionPlan {
  const directPaths = buildDirectPaths(input, assignments);
  const movingPathIndices = directPaths
    .map((path, index) => ({ path, index }))
    .filter(({ index }) => {
      const atom = input.initialAtoms[assignments[index]?.sourceIndex ?? -1];
      return atom?.movable && (assignments[index]?.disposition === "MOVE_TO_TARGET" || assignments[index]?.disposition === "PARK");
    })
    .map(({ index }) => index);
  const conflicts = findPathConflicts(directPaths, assignments, input);
  const staticConflicts = directPaths
    .map((path, index) => ({ path, index }))
    .filter(({ path }) => pathTouchesStaticGeometry(path, input));
  for (const conflict of staticConflicts) {
    const assignment = assignments[conflict.index];
    const atom = assignment ? input.initialAtoms[assignment.sourceIndex] : undefined;
    if (atom?.movable && (assignment?.disposition === "MOVE_TO_TARGET" || assignment?.disposition === "PARK")) {
      if (!movingPathIndices.includes(conflict.index)) movingPathIndices.push(conflict.index);
    } else {
      throw new SlmError("PATH_NOT_FOUND", "A fixed trap path intersects static geometry", {
        stage: "PLANNING",
        details: { atomId: conflict.path.atomId },
      });
    }
  }
  movingPathIndices.sort((a, b) => a - b);
  let paths = directPaths;
  let conflictComponentCount = connectedConflictComponents(conflicts).length + (staticConflicts.length > 0 ? 1 : 0);
  let directPathCount = directPaths.filter((path) => path.waypointsUm.length <= 2).length;

  if (conflicts.length > 0 || staticConflicts.length > 0) {
    if (movingPathIndices.length === 0) {
      throw new SlmError("PATH_NOT_FOUND", "A fixed trap path conflicts with static geometry", { stage: "PLANNING" });
    }
    const planned = planWithCbs(input, directPaths, movingPathIndices);
    if (!planned) {
      const fallback = serializedDetourPlan(input, directPaths, movingPathIndices);
      if (!fallback) {
        throw new SlmError("PATH_NOT_FOUND", "No collision-free path plan was found", {
          stage: "PLANNING",
          retryable: true,
          details: { conflicts },
        });
      }
      paths = fallback;
    } else {
      paths = planned;
    }
    directPathCount = paths.filter((path) => path.waypointsUm.length <= 2 && path.discreteTicks.length <= 2).length;
  } else {
    directPathCount = directPaths.filter((path) => path.waypointsUm.length <= 2).length;
  }

  const trajectories = parameterizeTrajectories(input, paths);
  const validation = validateContinuousTrajectories(trajectories, input);
  if (!validation.accepted) {
      throw new SlmError("COLLISION_VALIDATION_FAILED", "Continuous trajectory validation failed", {
      stage: "PLANNING",
      retryable: true,
      details: { validation: validation as unknown },
    });
  }
  const waitCount = paths.reduce((count, path) => count + countRepeatedWaypoints(path.waypointsUm), 0);
  const detourCount = paths.filter((path) => path.waypointsUm.length > 2).length;
  const makespanUs = Math.max(0, ...trajectories.map((trajectory) => trajectory.endTimeUs));
  const minimumValidatedSeparationUm = validation.minimumAtomSeparationUm;
  return {
    assignment: assignments,
    plannedPaths: paths,
    trajectories,
    directPathCount,
    conflictComponentCount,
    waitCount,
    detourCount,
    makespanUs,
    minimumValidatedSeparationUm,
  };
}

export function findPathConflicts(
  paths: PlannedPath[],
  assignments: AtomAssignment[],
  input: NormalizedInput,
): Conflict[] {
  const conflicts: Conflict[] = [];
  const maxTicks = Math.max(1, ...paths.map((path) => path.discreteTicks.at(-1) ?? 0));
  for (let first = 0; first < paths.length; first += 1) {
    for (let second = first + 1; second < paths.length; second += 1) {
      const atomA = input.initialAtoms[assignments[first]?.sourceIndex ?? -1];
      const atomB = input.initialAtoms[assignments[second]?.sourceIndex ?? -1];
      if (!atomA || !atomB) continue;
      const safe = pairSafeSeparation(atomA, atomB, input);
      for (let tick = 0; tick < maxTicks; tick += 1) {
        const a0 = plannedPointAt(paths[first]!, tick);
        const a1 = plannedPointAt(paths[first]!, tick + 1);
        const b0 = plannedPointAt(paths[second]!, tick);
        const b1 = plannedPointAt(paths[second]!, tick + 1);
        if (distance(a0, b0) < safe - EPSILON) {
          conflicts.push({ first, second, time: tick, kind: "vertex" });
          break;
        }
        if (distance(a0, b1) < safe - EPSILON && distance(a1, b0) < safe - EPSILON) {
          conflicts.push({ first, second, time: tick, kind: "edge" });
          break;
        }
        if (segmentDistance({ start: a0, end: a1 }, { start: b0, end: b1 }) < safe - EPSILON) {
          conflicts.push({ first, second, time: tick, kind: "near-edge" });
          break;
        }
      }
    }
  }
  return conflicts;
}

export function connectedConflictComponents(conflicts: Conflict[]): number[][] {
  const vertices = new Set<number>();
  conflicts.forEach((conflict) => {
    vertices.add(conflict.first);
    vertices.add(conflict.second);
  });
  const components: number[][] = [];
  const remaining = new Set(vertices);
  while (remaining.size > 0) {
    const start = [...remaining].sort((a, b) => a - b)[0]!;
    const component = [start];
    remaining.delete(start);
    for (let index = 0; index < component.length; index += 1) {
      const current = component[index]!;
      for (const conflict of conflicts) {
        const neighbor = conflict.first === current ? conflict.second : conflict.second === current ? conflict.first : -1;
        if (neighbor >= 0 && remaining.has(neighbor)) {
          remaining.delete(neighbor);
          component.push(neighbor);
        }
      }
    }
    components.push(component.sort((a, b) => a - b));
  }
  return components;
}

function planWithCbs(input: NormalizedInput, directPaths: PlannedPath[], activeIndices: number[]): PlannedPath[] | undefined {
  if (activeIndices.length === 0) return directPaths;
  const graph = buildPlanningGraph(input, directPaths, activeIndices);
  const starts: number[] = [];
  const goals: number[] = [];
  for (const pathIndex of activeIndices) {
    const path = directPaths[pathIndex]!;
    const start = exactGraphNode(graph.nodes, path.waypointsUm[0]!);
    const goal = exactGraphNode(graph.nodes, path.waypointsUm.at(-1)!);
    if (start < 0 || goal < 0) return undefined;
    starts.push(start);
    goals.push(goal);
  }

  const initialPaths: GridPath[] = [];
  for (let agent = 0; agent < activeIndices.length; agent += 1) {
    const path = lowLevelAStar(graph, starts[agent]!, goals[agent]!, [], input.plannerConfig.maxSearchTicks, input.plannerConfig.maxAStarExpansions);
    if (!path) return undefined;
    initialPaths.push(path);
  }
  const priorityPlan = prioritizedSpaceTimePlan(
    graph,
    starts,
    goals,
    activeIndices,
    directPaths,
    input,
  );
  if (priorityPlan && !findGridConflict(priorityPlan, activeIndices, directPaths, input, graph)) {
    return replaceGridPaths(directPaths, activeIndices, priorityPlan, graph);
  }
  const open: CbsNode[] = [{ constraints: [], paths: initialPaths, cost: cbsCost(initialPaths) }];
  let expanded = 0;
  while (open.length > 0 && expanded < input.plannerConfig.maxCbsNodes) {
    open.sort((a, b) => a.cost - b.cost || a.constraints.length - b.constraints.length);
    const current = open.shift()!;
    expanded += 1;
    const conflict = findGridConflict(current.paths, activeIndices, directPaths, input, graph);
    if (!conflict) {
      const result = [...directPaths];
      return replaceGridPaths(result, activeIndices, current.paths, graph);
    }
    const branches = conflict.kind === "vertex"
      ? [
          { agent: conflict.first, constraint: { agent: conflict.first, time: conflict.time, node: conflict.nodeA! } },
          { agent: conflict.second, constraint: { agent: conflict.second, time: conflict.time, node: conflict.nodeB! } },
        ]
      : [
          { agent: conflict.first, constraint: { agent: conflict.first, time: conflict.time, from: conflict.fromA!, to: conflict.toA! } },
          { agent: conflict.second, constraint: { agent: conflict.second, time: conflict.time, from: conflict.fromB!, to: conflict.toB! } },
        ];
    for (const branch of branches) {
      const constraints = [...current.constraints, branch.constraint];
      const paths = current.paths.slice();
      const replanned = lowLevelAStar(
        graph,
        starts[branch.agent]!,
        goals[branch.agent]!,
        constraints.filter((constraint) => constraint.agent === branch.agent),
        input.plannerConfig.maxSearchTicks,
        input.plannerConfig.maxAStarExpansions,
      );
      if (replanned) {
        paths[branch.agent] = replanned;
        open.push({ constraints, paths, cost: cbsCost(paths) });
      }
    }
  }
  return undefined;
}

function replaceGridPaths(
  directPaths: PlannedPath[],
  activeIndices: number[],
  gridPaths: GridPath[],
  graph: { nodes: GridNode[] },
): PlannedPath[] {
  const result = [...directPaths];
  for (let agent = 0; agent < activeIndices.length; agent += 1) {
    const pathIndex = activeIndices[agent]!;
    const gridPath = gridPaths[agent]!;
    result[pathIndex] = {
      ...directPaths[pathIndex]!,
      waypointsUm: gridPath.nodes.map((node) => ({ ...graph.nodes[node]!.point })),
      discreteTicks: [...gridPath.ticks],
    };
  }
  return result;
}

function prioritizedSpaceTimePlan(
  graph: { nodes: GridNode[] },
  starts: number[],
  goals: number[],
  activeIndices: number[],
  directPaths: PlannedPath[],
  input: NormalizedInput,
): GridPath[] | undefined {
  const orders: number[][] = [];
  const baseline = Array.from({ length: starts.length }, (_, index) => index);
  orders.push(baseline);
  orders.push([...baseline].sort((a, b) => {
    const lengthA = directPaths[activeIndices[a]!]!.waypointsUm;
    const lengthB = directPaths[activeIndices[b]!]!.waypointsUm;
    return pathLength(lengthB) - pathLength(lengthA) || a - b;
  }));
  orders.push([...baseline].reverse());
  const uniqueOrders = orders.filter((order, index) => orders.findIndex((candidate) => candidate.every((value, position) => value === order[position])) === index);
  for (const order of uniqueOrders.slice(0, input.plannerConfig.maxPriorityRetries)) {
    const paths: GridPath[] = Array(starts.length);
    let success = true;
    for (const agent of order) {
      const constraints: Constraint[] = [];
      for (let other = 0; other < starts.length; other += 1) {
        const reserved = paths[other];
        if (!reserved) continue;
        for (let time = 0; time <= input.plannerConfig.maxSearchTicks; time += 1) {
          const node = gridNodeAt(reserved, time);
          constraints.push({ agent, time, node });
          const next = gridNodeAt(reserved, time + 1);
          if (next !== node) constraints.push({ agent, time, from: next, to: node });
        }
      }
      const path = lowLevelAStar(graph, starts[agent]!, goals[agent]!, constraints, input.plannerConfig.maxSearchTicks, input.plannerConfig.maxAStarExpansions);
      if (!path) {
        success = false;
        break;
      }
      paths[agent] = path;
    }
    if (success && paths.every((path) => path !== undefined)) return paths as GridPath[];
  }
  return undefined;
}

interface GridConflict {
  first: number;
  second: number;
  time: number;
  kind: "vertex" | "edge";
  nodeA?: number;
  nodeB?: number;
  fromA?: number;
  toA?: number;
  fromB?: number;
  toB?: number;
}

function findGridConflict(
  paths: GridPath[],
  activeIndices: number[],
  directPaths: PlannedPath[],
  input: NormalizedInput,
  graph: { nodes: GridNode[] },
): GridConflict | undefined {
  const maxTicks = Math.max(1, ...paths.map((path) => path.ticks.at(-1) ?? 0));
  for (let first = 0; first < paths.length; first += 1) {
    for (let second = first + 1; second < paths.length; second += 1) {
      const pathAtomA = directPaths[activeIndices[first]!]?.atomId;
      const pathAtomB = directPaths[activeIndices[second]!]?.atomId;
      const atomA = input.initialAtoms.find((atom) => atom.atomId === pathAtomA);
      const atomB = input.initialAtoms.find((atom) => atom.atomId === pathAtomB);
      const safe = atomA && atomB ? pairSafeSeparation(atomA, atomB, input) : input.plannerConfig.minimumSeparationUm;
      for (let time = 0; time < maxTicks; time += 1) {
        const nodeA = gridNodeAt(paths[first]!, time);
        const nodeB = gridNodeAt(paths[second]!, time);
        if (nodeA === nodeB) {
          return { first, second, time, kind: "vertex", nodeA, nodeB };
        }
        const fromA = nodeA;
        const toA = gridNodeAt(paths[first]!, time + 1);
        const fromB = nodeB;
        const toB = gridNodeAt(paths[second]!, time + 1);
        if (toA === fromB && toB === fromA) {
          return { first, second, time, kind: "edge", fromA, toA, fromB, toB };
        }
        if (segmentDistance(
          { start: graph.nodes[fromA]!.point, end: graph.nodes[toA]!.point },
          { start: graph.nodes[fromB]!.point, end: graph.nodes[toB]!.point },
        ) < safe - EPSILON) {
          return { first, second, time, kind: "edge", fromA, toA, fromB, toB };
        }
      }
    }
  }
  return undefined;
}

function lowLevelAStar(
  graph: { nodes: GridNode[] },
  start: number,
  goal: number,
  constraints: Constraint[],
  maxTicks: number,
  maxExpansions: number,
): GridPath | undefined {
  interface SearchState {
    node: number;
    time: number;
    g: number;
    f: number;
    key: string;
  }
  const open: SearchState[] = [{ node: start, time: 0, g: 0, f: 0, key: stateKey(start, 0) }];
  const best = new Map<string, number>([[stateKey(start, 0), 0]]);
  const previous = new Map<string, { key: string; node: number; time: number }>();
  let expansions = 0;
  while (open.length > 0 && expansions < maxExpansions) {
    open.sort((a, b) => a.f - b.f || a.time - b.time || a.node - b.node);
    const current = open.shift()!;
    if (current.node === goal) return reconstructGridPath(current.key, current.node, current.time, previous);
    if (current.time >= maxTicks) continue;
    expansions += 1;
    const neighbors = [...graph.nodes[current.node]!.neighbors, current.node].sort((a, b) => a - b);
    for (const next of neighbors) {
      const nextTime = current.time + 1;
      if (hasVertexConstraint(constraints, next, nextTime) || hasEdgeConstraint(constraints, current.node, next, current.time)) continue;
      const key = stateKey(next, nextTime);
      const g = current.g + 1;
      if (g >= (best.get(key) ?? Number.POSITIVE_INFINITY)) continue;
      best.set(key, g);
      previous.set(key, { key: current.key, node: current.node, time: current.time });
      const goalPoint = graph.nodes[goal]!.point;
      const nextPoint = graph.nodes[next]!.point;
      const heuristic = distance(nextPoint, goalPoint);
      open.push({ node: next, time: nextTime, g, f: g + heuristic, key });
    }
  }
  return undefined;
}

function stateKey(node: number, time: number): string {
  return `${node}@${time}`;
}

function reconstructGridPath(
  key: string,
  node: number,
  time: number,
  previous: Map<string, { key: string; node: number; time: number }>,
): GridPath {
  const nodes = [node];
  const ticks = [time];
  let current = key;
  while (previous.has(current)) {
    const prior = previous.get(current)!;
    nodes.push(prior.node);
    ticks.push(prior.time);
    current = prior.key;
  }
  nodes.reverse();
  ticks.reverse();
  return { nodes, ticks };
}

function buildPlanningGraph(input: NormalizedInput, paths: PlannedPath[], activeIndices: number[]): { nodes: GridNode[] } {
  const anchors = activeIndices.flatMap((index) => {
    const path = paths[index]!;
    return [path.waypointsUm[0]!, path.waypointsUm.at(-1)!];
  });
  const staticPoints: Point2D[] = [
    ...input.initialAtoms.filter((atom) => !atom.movable),
    ...input.staticTraps.filter((trap) => trap.containsAtom),
  ];
  const activeSet = new Set(activeIndices);
  for (let index = 0; index < paths.length; index += 1) {
    if (!activeSet.has(index)) staticPoints.push(paths[index]!.waypointsUm.at(-1)!);
  }
  const allPoints = [...anchors, ...staticPoints, ...input.assignmentConfig.parkingSites];
  const maximumSigma = Math.max(0, ...input.initialAtoms.map((atom) => atom.localizationSigmaUm));
  const safeResolution = input.plannerConfig.minimumSeparationUm +
    2 * input.plannerConfig.kSigma * maximumSigma + input.plannerConfig.geometricMarginUm;
  // Leave a small lattice margin so adjacent nodes do not become merely
  // tangent after smoothing and floating-point validation.
  let resolution = Math.max(input.plannerConfig.gridResolutionUm, safeResolution * 1.25);
  const bounds = boundingBox(allPoints, Math.max(input.plannerConfig.minimumSeparationUm * 3, resolution * 2));
  const width = Math.max(1, Math.ceil((bounds.xMax - bounds.xMin) / resolution));
  const height = Math.max(1, Math.ceil((bounds.yMax - bounds.yMin) / resolution));
  const maxAxis = 128;
  if (width > maxAxis) resolution = (bounds.xMax - bounds.xMin) / maxAxis;
  if (height > maxAxis) resolution = Math.max(resolution, (bounds.yMax - bounds.yMin) / maxAxis);

  const xValues: number[] = [];
  const yValues: number[] = [];
  const xCount = Math.ceil((bounds.xMax - bounds.xMin) / resolution);
  const yCount = Math.ceil((bounds.yMax - bounds.yMin) / resolution);
  for (let index = 0; index <= xCount; index += 1) xValues.push(Math.min(bounds.xMax, bounds.xMin + index * resolution));
  for (let index = 0; index <= yCount; index += 1) yValues.push(Math.min(bounds.yMax, bounds.yMin + index * resolution));
  for (const anchor of anchors) {
    xValues.push(anchor.xUm);
    yValues.push(anchor.yUm);
  }
  const unique = (values: number[]) => [...new Set(values.map((value) => Number(value.toFixed(9))))].sort((a, b) => a - b);
  const xs = unique(xValues);
  const ys = unique(yValues);
  const nodes: GridNode[] = [];
  const nodeKey = new Map<string, number>();
  const safeStatic = input.plannerConfig.minimumSeparationUm +
    2 * input.plannerConfig.kSigma * maximumSigma + input.plannerConfig.geometricMarginUm;
  for (const x of xs) {
    for (const y of ys) {
      const point = { xUm: x, yUm: y };
      if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([point], [region]))) continue;
      if (!pointClearOfStaticAtoms(point, staticPoints, safeStatic)) continue;
      const key = coordinateKey(point);
      nodeKey.set(key, nodes.length);
      nodes.push({ point, neighbors: [] });
    }
  }
  // Add anchors even when rounding made their grid coordinate collide.
  for (const anchor of anchors) {
    const key = coordinateKey(anchor);
    if (nodeKey.has(key)) continue;
    if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([anchor], [region]))) continue;
    if (!pointClearOfStaticAtoms(anchor, staticPoints, safeStatic)) continue;
    nodeKey.set(key, nodes.length);
    nodes.push({ point: { ...anchor }, neighbors: [] });
  }

  const bucketSize = Math.max(resolution * 1.6, 1e-6);
  const buckets = new Map<string, number[]>();
  nodes.forEach((node, index) => {
    const key = bucketKey(node.point, bucketSize);
    const bucket = buckets.get(key) ?? [];
    bucket.push(index);
    buckets.set(key, bucket);
  });
  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes[index]!;
    const bx = Math.floor(node.point.xUm / bucketSize);
    const by = Math.floor(node.point.yUm / bucketSize);
    const candidates = new Set<number>();
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (const candidate of buckets.get(`${bx + dx},${by + dy}`) ?? []) candidates.add(candidate);
      }
    }
    for (const candidate of candidates) {
      if (candidate <= index) continue;
      const other = nodes[candidate]!;
      const edgeLength = distance(node.point, other.point);
      if (edgeLength > resolution * Math.SQRT2 * 1.25 + 1e-7) continue;
      const segment = { start: node.point, end: other.point };
      if (input.forbiddenRegions.some((region) => pathIntersectsForbiddenRegion([segment.start, segment.end], [region]))) continue;
      if (!pointClearOfStaticAtoms(segment.start, staticPoints, safeStatic) || !pointClearOfStaticAtoms(segment.end, staticPoints, safeStatic)) continue;
      if (staticPoints.some((point) => distancePointToSegment(point, segment) < safeStatic - EPSILON)) continue;
      node.neighbors.push(candidate);
      other.neighbors.push(index);
    }
  }
  nodes.forEach((node) => node.neighbors.sort((a, b) => a - b));
  return { nodes };
}

function serializedDetourPlan(input: NormalizedInput, directPaths: PlannedPath[], activeIndices: number[]): PlannedPath[] | undefined {
  const result = directPaths.map((path) => ({ ...path, waypointsUm: path.waypointsUm.map((point) => ({ ...point })), discreteTicks: [...path.discreteTicks] }));
  const staticPoints: Point2D[] = [
    ...input.initialAtoms.filter((atom) => !atom.movable),
    ...input.staticTraps.filter((trap) => trap.containsAtom),
  ];
  const activeSet = new Set(activeIndices);
  for (let index = 0; index < result.length; index += 1) {
    if (!activeSet.has(index)) staticPoints.push(result[index]!.waypointsUm.at(-1)!);
  }
  const active = activeIndices.slice().sort((a, b) => a - b);
  let tick = 0;
  const occupied = [...staticPoints];
  for (const index of active) {
    const path = result[index]!;
    const start = path.waypointsUm[0]!;
    const end = path.waypointsUm.at(-1)!;
    if (path.waypointsUm.length === 1) {
      tick += 1;
      occupied.push(end);
      continue;
    }
    const candidates = detourCandidates(start, end, input, occupied);
    let selected: Point2D[] | undefined;
    for (const candidate of candidates) {
      const candidatePath = [start, candidate, end];
      if (!pathIntersectsForbiddenRegion(candidatePath, input.forbiddenRegions) &&
          pathClearOfPoints(candidatePath, occupied, input.plannerConfig.minimumSeparationUm + input.plannerConfig.geometricMarginUm)) {
        selected = candidatePath;
        break;
      }
    }
    if (!selected) {
      if (!pathIntersectsForbiddenRegion([start, end], input.forbiddenRegions) &&
          pathClearOfPoints([start, end], occupied, input.plannerConfig.minimumSeparationUm + input.plannerConfig.geometricMarginUm)) {
        selected = [start, end];
      } else {
        return undefined;
      }
    }
    const waitTicks = tick;
    const movementTicks = Math.max(1, selected.length - 1);
    path.waypointsUm = selected;
    path.discreteTicks = selected.map((_, pointIndex) => waitTicks + pointIndex);
    if (waitTicks > 0) {
      path.waypointsUm = [...Array.from({ length: waitTicks + 1 }, () => ({ ...start })), ...selected.slice(1)];
      path.discreteTicks = path.waypointsUm.map((_, pointIndex) => pointIndex);
    }
    tick += movementTicks + 1;
    occupied.push(end);
  }
  return result;
}

function detourCandidates(start: Point2D, end: Point2D, input: NormalizedInput, occupied: Point2D[]): Point2D[] {
  const points = [...occupied, start, end, ...input.assignmentConfig.parkingSites];
  const bounds = boundingBox(points, input.plannerConfig.minimumSeparationUm * 4 + input.plannerConfig.geometricMarginUm);
  const offsets = [
    { xUm: bounds.xMin, yUm: bounds.yMin },
    { xUm: bounds.xMin, yUm: bounds.yMax },
    { xUm: bounds.xMax, yUm: bounds.yMin },
    { xUm: bounds.xMax, yUm: bounds.yMax },
    { xUm: (bounds.xMin + bounds.xMax) / 2, yUm: bounds.yMin },
    { xUm: (bounds.xMin + bounds.xMax) / 2, yUm: bounds.yMax },
    { xUm: bounds.xMin, yUm: (bounds.yMin + bounds.yMax) / 2 },
    { xUm: bounds.xMax, yUm: (bounds.yMin + bounds.yMax) / 2 },
    { xUm: (start.xUm + end.xUm) / 2, yUm: (start.yUm + end.yUm) / 2 + input.plannerConfig.minimumSeparationUm * 2 },
    { xUm: (start.xUm + end.xUm) / 2, yUm: (start.yUm + end.yUm) / 2 - input.plannerConfig.minimumSeparationUm * 2 },
  ];
  return offsets.sort((a, b) => distance(start, a) + distance(a, end) - distance(start, b) - distance(b, end));
}

function destinationForAssignment(input: NormalizedInput, assignment: AtomAssignment, atom: InitialAtom): Point2D {
  if (assignment.targetIndex !== null && input.targetSites[assignment.targetIndex]) {
    const target = input.targetSites[assignment.targetIndex]!;
    return { xUm: target.xUm, yUm: target.yUm };
  }
  if (assignment.disposition === "PARK") {
    const parking = input.assignmentConfig.parkingSites[
      assignment.parkingSiteIndex ?? assignment.sourceIndex % Math.max(1, input.assignmentConfig.parkingSites.length)
    ];
    if (parking) return { ...parking };
  }
  return { xUm: atom.xUm, yUm: atom.yUm };
}

function pairSafeSeparation(a: InitialAtom, b: InitialAtom, input: NormalizedInput): number {
  return input.plannerConfig.minimumSeparationUm +
     input.plannerConfig.kSigma * ((a.localizationSigmaUm ?? 0) + (b.localizationSigmaUm ?? 0)) +
    input.plannerConfig.geometricMarginUm;
}

function pathTouchesStaticGeometry(path: PlannedPath, input: NormalizedInput): boolean {
  const atom = input.initialAtoms.find((candidate) => candidate.atomId === path.atomId);
  const sigma = atom?.localizationSigmaUm ?? 0;
  const clearance = input.plannerConfig.minimumSeparationUm + input.plannerConfig.kSigma * sigma + input.plannerConfig.geometricMarginUm;
  const staticPoints: Point2D[] = [
    ...input.initialAtoms.filter((candidate) => !candidate.movable && candidate.atomId !== path.atomId),
    ...input.staticTraps
      .filter((trap) => trap.containsAtom && trap.atomId !== path.atomId),
  ];
  if (pathClearOfPoints(path.waypointsUm, staticPoints, clearance) === false) return true;
  return pathIntersectsForbiddenRegion(path.waypointsUm, input.forbiddenRegions);
}

function plannedPointAt(path: PlannedPath, tick: number): Point2D {
  if (path.waypointsUm.length === 1) return path.waypointsUm[0]!;
  const ticks = path.discreteTicks;
  if (tick <= ticks[0]!) return path.waypointsUm[0]!;
  const last = ticks.length - 1;
  if (tick >= ticks[last]!) return path.waypointsUm[last]!;
  for (let index = 0; index < last; index += 1) {
    const startTick = ticks[index]!;
    const endTick = ticks[index + 1]!;
    if (tick >= startTick && tick <= endTick) {
      const fraction = endTick === startTick ? 1 : (tick - startTick) / (endTick - startTick);
      return {
        xUm: path.waypointsUm[index]!.xUm + (path.waypointsUm[index + 1]!.xUm - path.waypointsUm[index]!.xUm) * fraction,
        yUm: path.waypointsUm[index]!.yUm + (path.waypointsUm[index + 1]!.yUm - path.waypointsUm[index]!.yUm) * fraction,
      };
    }
  }
  return path.waypointsUm[last]!;
}

function gridNodeAt(path: GridPath, time: number): number {
  if (time <= path.ticks[0]!) return path.nodes[0]!;
  const last = path.ticks.length - 1;
  if (time >= path.ticks[last]!) return path.nodes[last]!;
  for (let index = 0; index < last; index += 1) {
    if (time < path.ticks[index + 1]!) return path.nodes[index]!;
  }
  return path.nodes[last]!;
}

function hasVertexConstraint(constraints: Constraint[], node: number, time: number): boolean {
  return constraints.some((constraint) => constraint.time === time && constraint.node === node);
}

function hasEdgeConstraint(constraints: Constraint[], from: number, to: number, time: number): boolean {
  return constraints.some((constraint) =>
    constraint.time === time && constraint.from === from && constraint.to === to,
  );
}

function cbsCost(paths: GridPath[]): number {
  return paths.reduce((sum, path) => sum + (path.ticks.at(-1) ?? 0), 0);
}

function exactGraphNode(nodes: GridNode[], point: Point2D): number {
  return nodes.findIndex((node) => distance(node.point, point) <= 1e-8);
}

function coordinateKey(point: Point2D): string {
  return `${point.xUm.toFixed(9)},${point.yUm.toFixed(9)}`;
}

function bucketKey(point: Point2D, size: number): string {
  return `${Math.floor(point.xUm / size)},${Math.floor(point.yUm / size)}`;
}

function pathClearOfPoints(path: Point2D[], points: Point2D[], clearance: number): boolean {
  return points.every((point) => {
    if (path.some((pathPoint) => distance(pathPoint, point) < clearance - EPSILON)) return false;
    for (let index = 0; index + 1 < path.length; index += 1) {
      if (distancePointToSegment(point, { start: path[index]!, end: path[index + 1]! }) < clearance - EPSILON) return false;
    }
    return true;
  });
}

function countRepeatedWaypoints(points: Point2D[]): number {
  let count = 0;
  for (let index = 1; index < points.length; index += 1) {
    if (distance(points[index - 1]!, points[index]!) <= EPSILON) count += 1;
  }
  return count;
}

function pathLength(points: Point2D[]): number {
  let length = 0;
  for (let index = 1; index < points.length; index += 1) length += distance(points[index - 1]!, points[index]!);
  return length;
}

export const planPaths = planMotion;
