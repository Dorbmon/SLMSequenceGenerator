import { SlmError } from "./errors.js";
import {
  distance,
  pathIntersectsForbiddenRegion,
  pathClearOfStaticAtoms,
} from "./geometry.js";
import type {
  AssignmentConfig,
  AtomAssignment,
  ForbiddenRegion,
  InitialAtom,
  NormalizedInput,
  Point2D,
  StaticTrap,
  TargetSite,
} from "./types.js";

export const ASSIGNMENT_INFINITY = 1e30;

export interface HungarianResult {
  assignment: number[];
  cost: number;
  feasible: boolean;
}

export interface AssignmentProblem {
  atoms: InitialAtom[];
  targets: TargetSite[];
  forbiddenRegions?: ForbiddenRegion[];
  staticPoints?: Point2D[];
  config?: AssignmentConfig;
  conflictPenalties?: Map<string, number>;
}

/**
 * Solve a rectangular minimum-cost assignment. Rows are assigned to distinct
 * columns; when there are more rows than columns, every row still receives a
 * column by the caller's dummy-column construction.
 */
export function hungarianSolve(costMatrix: number[][]): HungarianResult {
  const rows = costMatrix.length;
  if (rows === 0) return { assignment: [], cost: 0, feasible: true };
  const columns = costMatrix[0]?.length ?? 0;
  if (columns === 0 || costMatrix.some((row) => row.length !== columns)) {
    throw new SlmError("INVALID_ARGUMENT", "Hungarian cost matrix must be rectangular and non-empty", {
      stage: "ASSIGNING",
    });
  }

  if (rows > columns) {
    return { assignment: Array(rows).fill(-1), cost: Number.POSITIVE_INFINITY, feasible: false };
  }

  // Potentials-and-augmenting-path implementation of the Hungarian method.
  const u = Array(rows + 1).fill(0) as number[];
  const v = Array(columns + 1).fill(0) as number[];
  const p = Array(columns + 1).fill(0) as number[];
  const way = Array(columns + 1).fill(0) as number[];

  for (let row = 1; row <= rows; row += 1) {
    p[0] = row;
    let columnZero = 0;
    const minValue = Array(columns + 1).fill(ASSIGNMENT_INFINITY) as number[];
    const used = Array(columns + 1).fill(false) as boolean[];
    do {
      used[columnZero] = true;
      const rowZero = p[columnZero]!;
      let delta = ASSIGNMENT_INFINITY;
      let nextColumn = 0;
      for (let column = 1; column <= columns; column += 1) {
        if (used[column]) continue;
        const matrixValue = costMatrix[rowZero - 1]![column - 1]!;
        const reduced = matrixValue - u[rowZero]! - v[column]!;
        if (reduced < minValue[column]!) {
          minValue[column] = reduced;
          way[column] = columnZero;
        }
        if (minValue[column]! < delta ||
            (minValue[column] === delta && column < nextColumn)) {
          delta = minValue[column]!;
          nextColumn = column;
        }
      }
      if (!Number.isFinite(delta) || delta >= ASSIGNMENT_INFINITY / 2) {
        return { assignment: Array(rows).fill(-1), cost: Number.POSITIVE_INFINITY, feasible: false };
      }
      for (let column = 0; column <= columns; column += 1) {
        if (used[column]) {
          const assignedRow = p[column] ?? 0;
          u[assignedRow] = (u[assignedRow] ?? 0) + delta;
          v[column] = (v[column] ?? 0) - delta;
        } else {
          minValue[column] = minValue[column]! - delta;
        }
      }
      columnZero = nextColumn;
    } while (p[columnZero] !== 0);

    do {
      const previousColumn = way[columnZero]!;
      p[columnZero] = p[previousColumn]!;
      columnZero = previousColumn;
    } while (columnZero !== 0);
  }

  const assignment = Array(rows).fill(-1) as number[];
  for (let column = 1; column <= columns; column += 1) {
    const assignedRow = p[column] ?? 0;
    if (assignedRow > 0) assignment[assignedRow - 1] = column - 1;
  }
  let cost = 0;
  for (let row = 0; row < rows; row += 1) {
    const column = assignment[row]!;
    if (column < 0 || costMatrix[row]![column]! >= ASSIGNMENT_INFINITY / 2) {
      return { assignment, cost: Number.POSITIVE_INFINITY, feasible: false };
    }
    cost += costMatrix[row]![column]!;
  }
  return { assignment, cost, feasible: Number.isFinite(cost) };
}

export function buildAssignmentCostMatrix(
  problem: AssignmentProblem,
): { matrix: number[][]; targets: TargetSite[]; dummyCount: number } {
  const config: Required<AssignmentConfig> = {
    distanceWeight: 1,
    obstacleWeight: 1_000,
    staticObstacleWeight: 100,
    groupMismatchPenalty: 1_000,
    groupMismatchPolicy: "forbid",
    extraAtomPolicy: "KEEP",
    parkingSites: [],
    stayToleranceUm: 0.25,
    maxAssignmentRetries: 2,
    conflictPenalty: 1_000,
    ...problem.config,
  };
  const targets = problem.targets.filter((target) => target.required !== false);
  const dummyCount = Math.max(0, problem.atoms.length - targets.length);
  const columns = targets.length + dummyCount;
  if (problem.atoms.length > 0 && columns === 0) {
    throw new SlmError("ASSIGNMENT_INFEASIBLE", "No destination columns are available", { stage: "ASSIGNING" });
  }
  const matrix = problem.atoms.map((atom, atomIndex) => {
    const targetCosts = targets.map((target, targetIndex) =>
      assignmentEdgeCost(atom, target, atomIndex, targetIndex, problem, config),
    );
    const extraCosts = Array.from({ length: dummyCount }, (_, dummyIndex) =>
      dummyDestinationCost(atom, dummyIndex, problem, config),
    );
    return [...targetCosts, ...extraCosts];
  });
  return { matrix, targets, dummyCount };
}

export function assignAtoms(input: NormalizedInput, conflictPenalties: Map<string, number> = new Map()): AtomAssignment[] {
  const movable = input.initialAtoms.filter((atom) => atom.movable);
  const staticAtoms = input.initialAtoms.filter((atom) => !atom.movable);
  const staticPoints = [
    ...staticAtoms,
    ...input.staticTraps.filter((trap) => trap.containsAtom),
  ];
  const { matrix, targets, dummyCount } = buildAssignmentCostMatrix({
    atoms: movable,
    targets: input.targetSites,
    forbiddenRegions: input.forbiddenRegions,
    staticPoints,
    config: input.assignmentConfig,
    conflictPenalties,
  });
  const solved = hungarianSolve(matrix);
  if (!solved.feasible) {
    throw new SlmError("ASSIGNMENT_INFEASIBLE", "No assignment satisfies the identity and group constraints", {
      stage: "ASSIGNING",
      details: { matrix },
    });
  }

  const result: AtomAssignment[] = [];
  for (const atom of staticAtoms) {
    const sourceIndex = input.initialAtoms.indexOf(atom);
    result.push({
      atomId: atom.atomId,
      sourceIndex,
      targetSiteId: null,
      targetIndex: null,
      disposition: "STAY",
      assignmentCost: 0,
    });
  }
  for (let movableIndex = 0; movableIndex < movable.length; movableIndex += 1) {
    const atom = movable[movableIndex]!;
    const column = solved.assignment[movableIndex]!;
    const sourceIndex = input.initialAtoms.indexOf(atom);
    if (column >= 0 && column < targets.length) {
      const target = targets[column]!;
      const targetIndex = input.targetSites.findIndex((candidate) => candidate.siteId === target.siteId);
      const alreadyThere = distance(atom, target) <= input.assignmentConfig.stayToleranceUm;
      result.push({
        atomId: atom.atomId,
        sourceIndex,
        targetSiteId: target.siteId!,
        targetIndex,
        disposition: alreadyThere ? "STAY" : "MOVE_TO_TARGET",
        assignmentCost: matrix[movableIndex]![column]!,
      });
    } else if (column >= targets.length && column < targets.length + dummyCount) {
      const policy = input.assignmentConfig.extraAtomPolicy;
      result.push({
        atomId: atom.atomId,
        sourceIndex,
        targetSiteId: null,
        targetIndex: null,
        disposition: policy === "KEEP" ? "KEEP" : policy === "RELEASE_IN_PLACE" ? "RELEASE" : "PARK",
        assignmentCost: matrix[movableIndex]![column]!,
        ...(policy === "PARK" || policy === "PARK_AND_RELEASE"
          ? { parkingSiteIndex: (column - targets.length) % Math.max(1, input.assignmentConfig.parkingSites.length) }
          : {}),
      });
    } else {
      throw new SlmError("ASSIGNMENT_INFEASIBLE", "Hungarian solver returned an incomplete assignment", {
        stage: "ASSIGNING",
      });
    }
  }
  return result.sort((a, b) => a.sourceIndex - b.sourceIndex);
}

function assignmentEdgeCost(
  atom: InitialAtom,
  target: TargetSite,
  atomIndex: number,
  targetIndex: number,
  problem: AssignmentProblem,
  config: Required<AssignmentConfig>,
): number {
  if (target.requiredAtomId !== undefined && target.requiredAtomId !== -1 && target.requiredAtomId !== atom.atomId) {
    return ASSIGNMENT_INFINITY;
  }
  if (target.requiredGroup !== undefined && target.requiredGroup !== -1 && target.requiredGroup !== atom.group) {
    if (config.groupMismatchPolicy === "forbid") return ASSIGNMENT_INFINITY;
  }
  const path = [
    { xUm: atom.xUm, yUm: atom.yUm },
    { xUm: target.xUm, yUm: target.yUm },
  ];
  const squaredDistance = (atom.xUm - target.xUm) ** 2 + (atom.yUm - target.yUm) ** 2;
  let cost = config.distanceWeight * squaredDistance;
  if (problem.forbiddenRegions && pathIntersectsForbiddenRegion(path, problem.forbiddenRegions)) {
    cost += config.obstacleWeight;
  }
  if (problem.staticPoints && !pathClearOfStaticAtoms(path, problem.staticPoints, config.stayToleranceUm)) {
    cost += config.staticObstacleWeight;
  }
  if (target.requiredGroup !== undefined && target.requiredGroup !== -1 && target.requiredGroup !== atom.group) {
    cost += config.groupMismatchPenalty;
  }
  if (distance(atom, target) <= config.stayToleranceUm) cost *= 0.001;
  cost += problem.conflictPenalties?.get(`${atom.atomId}:${target.siteId}`) ?? 0;
  // Keep the matrix deterministic when two edges have exactly equal geometry.
  return cost + atomIndex * 1e-12 + targetIndex * 1e-15;
}

function dummyDestinationCost(
  atom: InitialAtom,
  dummyIndex: number,
  problem: AssignmentProblem,
  config: Required<AssignmentConfig>,
): number {
  if (config.extraAtomPolicy === "KEEP" || config.extraAtomPolicy === "RELEASE_IN_PLACE") return 0 + dummyIndex * 1e-15;
  const parking = config.parkingSites[dummyIndex % Math.max(1, config.parkingSites.length)];
  if (!parking) return 0.5 + dummyIndex * 1e-15;
  const path = [{ xUm: atom.xUm, yUm: atom.yUm }, parking];
  let cost = config.distanceWeight * distance(atom, parking) ** 2;
  if (problem.forbiddenRegions && pathIntersectsForbiddenRegion(path, problem.forbiddenRegions)) {
    cost += config.obstacleWeight;
  }
  return cost + dummyIndex * 1e-15;
}

export function assignmentCost(assignments: AtomAssignment[]): number {
  return assignments.reduce((sum, assignment) => sum + assignment.assignmentCost, 0);
}

export const solveAssignment = assignAtoms;
