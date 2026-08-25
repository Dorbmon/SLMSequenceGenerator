# WebAssembly-Based Atom Rearrangement and Classical SLM Sequence Compiler

**Implementation specification for:**

> **initial occupied point array + target point array → automatic atom assignment → collision-free atom motion planning → intermediate optical-trap frames → phase-only SLM frames**

| Field | Value |
|---|---|
| Document version | 0.2 |
| Status | Implementation-ready design draft |
| Date | August 25, 2026 |
| Primary implementation | C++17 compiled to WebAssembly with Emscripten, orchestrated by TypeScript |
| Numerical methods | Hungarian assignment, classical multi-agent path planning, minimum-jerk interpolation, phase-stable Weighted Gerchberg-Saxton |
| Default execution mode | Offline compilation of the complete sequence before the experiment |
| Primary dimensionality | Two-dimensional focal-plane optical-tweezer arrays |

## 1. Purpose

This document defines an end-to-end program that accepts two point arrays:

1. the positions of atoms that are currently occupied and trapped; and
2. the positions that must be occupied in the final target array.

The program automatically produces:

1. an atom-to-target assignment;
2. collision-free trajectories for all selected atoms;
3. a time-sampled sequence of intermediate optical-trap configurations; and
4. one calibrated, full-resolution SLM phase frame for every trap configuration.

The output of the system is therefore **not merely a target point array** and not merely one hologram. It is a complete executable rearrangement sequence:

$$
\mathcal{S}_0,\mathcal{G}
\longrightarrow
\mathcal{A}
\longrightarrow
\mathcal{P}
\longrightarrow
\{T_0,T_1,\ldots,T_{F-1}\}
\longrightarrow
\{H_0,H_1,\ldots,H_{F-1}\}.
$$

Here:

- $\mathcal{S}_0$ is the initial occupied atom set;
- $\mathcal{G}$ is the desired target-site set;
- $\mathcal{A}$ is the atom-to-target assignment;
- $\mathcal{P}$ is the set of collision-free, time-parameterized atom paths;
- $T_f$ is the complete optical-trap frame at sequence frame $f$; and
- $H_f$ is the calibrated phase-only SLM image displayed for frame $f$.

The first implementation uses only classical algorithms. No neural network is required for assignment, path planning, trajectory generation, or hologram solving.

## 2. Normative language

- **MUST** means required for correctness or safety.
- **SHOULD** means the recommended default.
- **MAY** means optional.

## 3. Core product behavior

The primary API MUST support this operation:

```ts
const result = await compiler.compileRearrangement({
  initialAtoms,
  targetSites,
  staticTraps,
  forbiddenRegions,
  plannerConfig,
  motionConfig,
  hologramConfig,
  calibration,
});
```

The returned sequence MUST contain enough information to inspect, simulate, save, and execute the rearrangement:

```ts
interface CompiledSequence {
  manifest: SequenceManifest;
  assignment: AtomAssignment[];
  trajectories: AtomTrajectory[];
  trapFrameStore: FrameStore<TrapFrame>;
  slmFrameStore: FrameStore<Uint8Array | Uint16Array>;
  frameMetrics: FrameMetrics[];
  validation: SequenceValidationReport;
}
```

The implementation MUST expose both:

- a high-level one-call sequence compiler; and
- lower-level stage APIs for debugging, benchmarking, and substitution of individual algorithms.

## 4. End-to-end pipeline

```text
Initial occupied points          Target points
          │                           │
          └────────────┬──────────────┘
                       ▼
             Input normalization
                       ▼
              Feasibility checks
                       ▼
          Atom-to-target assignment
          Hungarian / min-cost matching
                       ▼
         Fast direct-path construction
                       ▼
        Continuous collision detection
                 ┌─────┴─────┐
                 │ no conflict│ conflict
                 ▼            ▼
         Accept direct     MAPF fallback
             paths       A* + reservations,
                         then ECBS if needed
                 └─────┬─────┘
                       ▼
          Minimum-jerk time parameterization
                       ▼
          Intermediate trap-frame sampling
                       ▼
       Sequential phase-stable WGS hologram solve
                       ▼
       Calibration, quantization, quality gates
                       ▼
        Complete validated SLM frame sequence
```

The complete compilation process MUST be deterministic when given the same:

- inputs;
- configuration;
- calibration revision;
- solver seed;
- Wasm build identifier; and
- numeric mode.

## 5. Scope

### 5.1 In scope for the MVP

- Two-dimensional atom coordinates.
- An initial list of occupied atom positions.
- An arbitrary two-dimensional target point array.
- More initial atoms than required target sites.
- Automatic selection of which initial atoms to use.
- Optional fixed atom identities or group constraints.
- Static atoms and static traps that must not move.
- Forbidden regions and clearance margins.
- Collision-free atom motion planning.
- Parallel motion when paths permit it.
- Automatic insertion of waits and detours when paths conflict.
- Smooth trajectory generation with bounded velocity, acceleration, and jerk.
- Automatic sampling into intermediate trap frames.
- One classical phase-only SLM hologram per trap frame.
- Sequential warm-started WGS with persistent trap identities and phases.
- Measured SLM phase-response lookup tables and optical aberration correction.
- Offline precomputation of the full sequence.
- Browser simulation and a native hardware-adapter path.

### 5.2 Deferred features

- General three-dimensional and multi-plane motion.
- Online camera localization and closed-loop replanning within the SLM update deadline.
- Quantum-state-aware identity constraints beyond caller-supplied labels.
- Learned assignment, learned planning, or learned hologram generation.
- A browser tab as the sole hard-real-time experiment controller.
- Device-specific vendor SDK bindings before the SLM model is selected.

## 6. Physical and software assumptions

1. Every selected atom remains associated with one optical trap throughout its planned motion.
2. A trap identifier follows the atom, not the target site.
3. The SLM can display a sequence of phase patterns generated before execution.
4. The effective physical update behavior of the SLM is calibration-dependent. The compiler MUST not assume that commanded grayscale changes instantaneously produce the requested optical phase.
5. Motion limits, required atom separation, trap depth, and SLM refresh interval are experiment-specific inputs.
6. The program can prove consistency with its configured geometric and numerical model, but physical success still depends on calibration accuracy and the real optical system.

## 7. Input model

### 7.1 Minimal input

The minimal logical input is:

```ts
interface Point2D {
  xUm: number;
  yUm: number;
}

interface InitialAtom extends Point2D {
  atomId?: number;
}

interface TargetSite extends Point2D {
  siteId?: number;
}

interface RearrangementRequest {
  initialAtoms: InitialAtom[];
  targetSites: TargetSite[];
  calibrationId: string;
}
```

If identifiers are omitted, the runtime MUST assign deterministic identifiers according to input order.

### 7.2 Recommended complete input

```ts
interface InitialAtom {
  atomId: number;
  xUm: number;
  yUm: number;
  group?: number;
  movable?: boolean;
  initialTrapIntensity?: number;
  localizationSigmaUm?: number;
}

interface TargetSite {
  siteId: number;
  xUm: number;
  yUm: number;
  required?: boolean;
  requiredAtomId?: number;
  requiredGroup?: number;
  finalTrapIntensity?: number;
}

interface StaticTrap {
  trapId: number;
  xUm: number;
  yUm: number;
  intensity: number;
  containsAtom: boolean;
}

interface ForbiddenRegion {
  type: "circle" | "axisAlignedBox" | "polygon";
  coordinates: number[];
  clearanceUm?: number;
}
```

### 7.3 Alternative lattice-plus-occupancy input

A camera or experiment controller MAY provide an entire initial trap lattice and a Boolean occupancy mask. The TypeScript normalization layer MUST convert that representation to the canonical list of occupied atoms before invoking the planner.

```ts
interface OccupiedLatticeInput {
  sites: Point2D[];
  occupied: Uint8Array;
}
```

Only occupied sites become `InitialAtom` objects. Empty source sites MAY still be added to the planning graph as candidate waypoints or parking locations.

### 7.4 Required feasibility rules

The compiler MUST reject the request before planning when any of the following holds:

- fewer movable initial atoms exist than required target sites;
- two required target sites violate the configured minimum separation;
- an initial atom or target site lies outside the calibrated field of view;
- a fixed-identity constraint cannot be satisfied;
- a target site lies inside a forbidden region;
- identifiers are duplicated;
- a coordinate, intensity, or configuration value is non-finite; or
- the calibration package does not match the requested grid and wavelength.

## 8. Output model

### 8.1 Atom assignment

```ts
interface AtomAssignment {
  atomId: number;
  sourceIndex: number;
  targetSiteId: number | null;
  targetIndex: number | null;
  disposition: "MOVE_TO_TARGET" | "STAY" | "PARK" | "KEEP" | "RELEASE";
  assignmentCost: number;
}
```

### 8.2 Time-parameterized trajectory

```ts
interface TrajectoryWaypoint {
  xUm: number;
  yUm: number;
  arrivalTimeUs: number;
}

interface AtomTrajectory {
  atomId: number;
  trapId: number;
  targetSiteId: number | null;
  waypoints: TrajectoryWaypoint[];
  startTimeUs: number;
  endTimeUs: number;
  moving: boolean;
}
```

### 8.3 Trap frame

```ts
interface TrapState {
  trapId: number;
  atomId: number | null;
  xUm: number;
  yUm: number;
  intensity: number;
  targetPhaseRad: number;
  flags: number;
}

interface TrapFrame {
  frameIndex: number;
  timeUs: number;
  traps: TrapState[];
}
```

The order of `traps` MUST be deterministic and SHOULD be ascending by `trapId`.

### 8.4 SLM frame

Each accepted trap frame produces exactly one SLM frame:

```ts
interface SlmFrameDescriptor {
  frameIndex: number;
  timeUs: number;
  width: number;
  height: number;
  format: "UINT8" | "UINT16";
  byteOffset: bigint;
  byteLength: number;
  crc32: number;
}
```

The actual pixel data SHOULD be streamed to a frame store instead of retained as thousands of JavaScript arrays.

## 9. Stage A: input normalization and feasibility analysis

### 9.1 Coordinate normalization

All planner coordinates use physical micrometres in an experiment coordinate system:

- $+x$ points right in the atom plane;
- $+y$ points upward in the atom plane;
- image-coordinate inversion is handled by calibration;
- SLM pixel coordinates do not enter the motion planner.

### 9.2 Occupancy and identity normalization

The normalizer MUST:

1. assign missing atom, site, and trap identifiers;
2. merge duplicate points within `duplicatePointToleranceUm` or reject them according to policy;
3. classify atoms as movable or static;
4. classify target sites as required or optional;
5. expand localization uncertainty into the safety margin; and
6. create the initial trap identities used later by WGS.

### 9.3 Safety radius

The planner models each atom as a disc. Let:

- $d_{\min}$ be the required center-to-center atom separation;
- $\sigma_i$ be the localization uncertainty of atom $i$; and
- $m$ be an additional geometric margin.

A conservative pairwise separation requirement is:

$$
d_{ij,\mathrm{safe}}
=
d_{\min}+k_\sigma(\sigma_i+\sigma_j)+m.
$$

The default `kSigma` SHOULD be configurable. No universal value is specified in this document.

### 9.4 Static obstacles

Static occupied traps MUST be treated as dynamic-planning obstacles with infinite reservation duration. Optical or mechanical forbidden regions MUST be inflated by the configured atom clearance before graph construction.

## 10. Stage B: automatic atom-to-target assignment

### 10.1 Assignment problem

For $N$ movable initial atoms and $M$ required target sites, with $N \ge M$, the compiler chooses $M$ atoms and maps each chosen atom to one target site.

The default method is a rectangular minimum-cost bipartite assignment solved by the Hungarian algorithm or a compatible Jonker-Volgenant implementation.

The assignment solver optimizes the supplied cost matrix. It does **not** by itself guarantee globally collision-free continuous trajectories. Collision freedom is enforced by the motion-planning stage.

### 10.2 Cost function

For source atom $i$ and target site $j$, the recommended cost is:

$$
C_{ij}
=
w_d D_{ij}
+w_o O_{ij}
+w_s S_{ij}
+w_g G_{ij}
+w_i I_{ij}.
$$

Where:

- $D_{ij}$ is squared or Euclidean travel distance;
- $O_{ij}$ is a direct-path obstacle penalty;
- $S_{ij}$ is a soft penalty for passing close to static occupied atoms;
- $G_{ij}$ is an optional group mismatch penalty; and
- $I_{ij}$ is zero for allowed identity mappings and infinity for forbidden mappings.

A reasonable distance term is:

$$
D_{ij}=\lVert \mathbf{s}_i-\mathbf{g}_j \rVert_2^2.
$$

The squared distance discourages a small number of very long moves more strongly than a linear distance objective.

### 10.3 Fixed and group-constrained assignments

- When `requiredAtomId` is set, all other source-to-target costs MUST be infinite.
- When `requiredGroup` is set, atoms outside the group MUST be forbidden or assigned a configured penalty.
- Atoms already within `stayToleranceUm` of a compatible target SHOULD receive a zero or strongly preferred cost.

### 10.4 Extra atoms

When $N > M$, the assignment matrix MUST include dummy destinations representing the configured extra-atom policy.

Supported policies:

| Policy | Behavior |
|---|---|
| `KEEP` | Keep the extra atom trapped at its initial position throughout the sequence. |
| `PARK` | Move the extra atom to a caller-supplied or automatically selected parking site. |
| `PARK_AND_RELEASE` | Move it outside the active target region, then ramp its trap to zero. Recommended when the final array must contain only the target atoms. |
| `RELEASE_IN_PLACE` | Ramp down at the source after selected atoms are clear. Use only when experimentally acceptable. |

### 10.5 Assignment repair loop

A distance-optimal assignment can create a difficult or impossible path-planning problem. The compiler SHOULD implement this bounded repair loop:

1. solve the initial assignment;
2. attempt direct and MAPF path planning;
3. identify source-target pairs participating in unresolved conflicts;
4. add conflict penalties to those assignment edges;
5. rerun the assignment; and
6. stop after `maxAssignmentRetries` or when a feasible plan is found.

The final manifest MUST record the number of assignment attempts and the selected cost.

## 11. Stage C: collision-free multi-atom path planning

### 11.1 Planning objectives

The path planner MUST produce one path for every active trap such that:

- every selected atom reaches its assigned target;
- all static atoms remain fixed;
- no pair violates the configured separation;
- no path enters a forbidden region;
- velocity, acceleration, and jerk can be satisfied after time parameterization;
- the plan can be sampled at the SLM update period; and
- all trap identities persist from start to finish.

The default optimization order SHOULD be:

1. feasibility;
2. minimum makespan;
3. minimum maximum travel distance;
4. minimum total travel distance;
5. fewer waits and turns.

### 11.2 Tiered planner

The recommended implementation is deliberately tiered.

#### Tier 1: synchronized direct paths

Each atom initially receives a straight path from its source to its assigned target. Stationary atoms receive a zero-length path. All direct paths are time-parameterized with a common or compatible duration and checked for continuous collision risk.

Most sparse rearrangements should finish at this stage.

#### Tier 2: conflict-component decomposition

If direct paths conflict, construct a conflict graph:

- each vertex is an atom trajectory;
- an edge connects two trajectories that violate clearance at any time.

Connected components can be planned independently when their expanded geometric envelopes do not overlap. This avoids sending the entire array through an expensive global MAPF solver.

#### Tier 3: prioritized space-time A*

For each conflict component, the default scalable fallback is prioritized planning:

1. choose an atom priority order;
2. plan one path at a time with A* in a space-time graph;
3. reserve occupied vertices, swept edges, and safety intervals;
4. permit wait actions; and
5. retry with several deterministic priority orders.

Recommended orders include:

- longest path first;
- most constrained goal first;
- highest conflict degree first; and
- stable atom identifier order as the deterministic baseline.

#### Tier 4: bounded-suboptimal Conflict-Based Search

If prioritized planning fails for a small or medium conflict component, the implementation SHOULD invoke Enhanced Conflict-Based Search (ECBS) or standard Conflict-Based Search (CBS).

CBS separates planning into:

- a high-level conflict tree that adds constraints; and
- low-level single-agent A* searches satisfying those constraints.

ECBS is preferred for production when a bounded-suboptimal solution is acceptable, because exact optimal CBS can expand rapidly on dense instances.

#### Tier 5: batch serialization

If a component remains unsolved, the planner MAY split it into movement batches, keeping non-batch atoms stationary. This sacrifices makespan but often restores feasibility. If no collision-free plan exists with available parking space and constraints, compilation MUST fail explicitly rather than emit an unsafe sequence.

### 11.3 Planning graph

The MVP SHOULD support two graph builders.

#### Lattice graph

Use when source and target points lie on, or close to, a regular lattice.

- vertices are legal lattice sites plus parking sites;
- edges connect four or eight neighboring sites;
- wait actions are allowed;
- static occupied sites are blocked;
- edge swaps are forbidden.

#### Geometric roadmap

Use for arbitrary point arrays.

- generate a uniform or adaptive planning grid;
- add every start, goal, and parking point as a vertex;
- connect visible nearby vertices;
- reject edges whose swept clearance disc intersects an obstacle;
- assign edge cost from geometric length and turn penalty.

The grid or roadmap resolution MUST be fine enough that graph discretization does not invalidate the requested motion clearance.

### 11.4 Discrete conflict definitions

For graph-based planning, the planner MUST reject at least:

- **vertex conflict:** two atoms occupy the same reserved node at the same planning tick;
- **edge-swap conflict:** atom A traverses $u\rightarrow v$ while atom B traverses $v\rightarrow u$ during the same interval;
- **near-edge conflict:** two swept segments approach more closely than the configured separation;
- **static conflict:** an atom intersects a permanently occupied node or forbidden region; and
- **goal blocking:** an atom reaches its goal but blocks another atom's required route without a valid later departure.

### 11.5 Continuous collision validation

Graph conflict freedom is necessary but not sufficient after geometric smoothing. Every final continuous trajectory pair MUST be validated.

The validator SHOULD use adaptive sampling with a velocity-based bound. If two trajectories are sampled at interval $h$, and the maximum relative speed over the interval is $v_{\mathrm{rel,max}}$, then a sample separation of

$$
d_{\mathrm{sample}}
>
d_{\mathrm{safe}}+\frac{v_{\mathrm{rel,max}}h}{2}
$$

is a conservative sufficient condition against an undetected crossing between adjacent samples.

Intervals that do not satisfy the bound MUST be subdivided until they are proven safe or the configured validation depth is reached.

### 11.6 Path-planning result

The planner output before smoothing is:

```cpp
struct PlannedPath {
  uint32_t atom_id;
  uint32_t trap_id;
  uint32_t goal_site_id;
  Span<Vec2f> waypoints_um;
  Span<uint32_t> discrete_ticks;
};
```

## 12. Stage D: smooth trajectory and intermediate trap-frame generation

### 12.1 Trajectory requirements

Every path segment MUST satisfy configured limits:

- maximum speed $v_{\max}$;
- maximum acceleration $a_{\max}$;
- maximum jerk $j_{\max}$;
- maximum position change per displayed SLM frame;
- minimum dwell time before movement;
- minimum settle time after arrival; and
- minimum pairwise atom separation.

### 12.2 Default minimum-jerk interpolation

For a segment from point $\mathbf{p}_0$ to $\mathbf{p}_1$ with duration $T$, define normalized time $s=t/T$ and:

$$
q(s)=10s^3-15s^4+6s^5, \qquad s\in[0,1].
$$

The trajectory is:

$$
\mathbf{p}(t)=\mathbf{p}_0+q(t/T)(\mathbf{p}_1-\mathbf{p}_0).
$$

This interpolation has zero velocity and acceleration at both endpoints. It is a robust MVP default because graph waypoints can be treated as full stops. A later optimization MAY replace full stops with globally smooth splines after preserving collision constraints.

For a segment of length $L$, useful conservative duration lower bounds are:

$$
T \ge \frac{1.875L}{v_{\max}},
$$

$$
T \ge \sqrt{\frac{5.774L}{a_{\max}}},
$$

$$
T \ge \sqrt[3]{\frac{60L}{j_{\max}}}.
$$

The implementation MUST also round the duration upward to an integer number of SLM frame intervals.

### 12.3 Synchronization

The planner MAY use either:

- **global synchronization:** all active atoms start and finish together; or
- **scheduled synchronization:** each atom or batch has explicit start and end times.

Scheduled synchronization is required when waits are introduced by MAPF.

### 12.4 Trap intensity profile

The default selected-atom intensity policy is:

1. hold the initial trap intensity during the pre-move dwell;
2. ramp to `movingTrapIntensity` if a movement boost is configured;
3. hold approximately constant during motion;
4. ramp to the final site intensity during settle; and
5. preserve the final intensity afterward.

Ramps SHOULD use the same quintic smoothstep unless calibration requires another profile.

For `PARK_AND_RELEASE`, an extra atom MUST first reach its parking site and complete a settle dwell before its trap intensity is ramped toward zero.

### 12.5 Trap optical phase policy

Every moving trap has a persistent `trapId`. The default target-plane optical phase is constant over the entire trajectory:

$$
\theta_j(f)=\theta_j(0).
$$

This gives the hologram solver an explicit inter-frame phase-continuity target. Optional policies MAY interpolate unwrapped phases between anchor values, but wrapped phases MUST never be linearly interpolated directly.

### 12.6 Frame sampling

Let the commanded SLM frame interval be $\Delta t_{\mathrm{SLM}}$. The trap-frame generator MUST sample every active trajectory at:

$$
t_f=f\Delta t_{\mathrm{SLM}}, \qquad f=0,1,\ldots,F-1.
$$

It MUST include:

- the exact initial frame;
- all movement frames;
- the exact final frame;
- configured pre-move and post-move dwell frames; and
- intensity-ramp frames.

The generated frame is:

$$
T_f=\{(\mathrm{trapId}_j,x_j(t_f),y_j(t_f),I_j(t_f),\theta_j(t_f))\}_{j=1}^{K_f}.
$$

### 12.7 Per-frame geometric validation

Before hologram solving, every trap frame MUST pass:

- coordinate bounds;
- minimum pairwise separation;
- maximum per-frame position delta;
- maximum per-frame intensity delta;
- stable trap identifier rules; and
- lifecycle rules for births and deaths.

The compiler MUST also validate the continuous interpolated motion between adjacent trap frames, not only the discrete endpoints.

## 13. Stage E: classical SLM hologram generation

### 13.1 Optical field model

Let $A_{\mathrm{in}}(u,v)$ be the measured incident amplitude at the SLM plane and $\phi_f(u,v)$ be the phase optimized for trap frame $f$.

The SLM-plane field is:

$$
U_f(u,v)=A_{\mathrm{in}}(u,v)e^{i\phi_f(u,v)}.
$$

For a focal-plane Fourier configuration:

$$
V_f(x,y)=\mathcal{F}\{U_f(u,v)\}.
$$

At each requested trap coordinate $\mathbf{x}_{f,j}$, the solver seeks:

$$
|V_f(\mathbf{x}_{f,j})|^2 \propto I_{f,j}
$$

while also controlling the complex phase:

$$
\arg V_f(\mathbf{x}_{f,j}) \approx \theta_{f,j}.
$$

### 13.2 Coordinate calibration

The production solver MUST use either measured coordinate calibration or the
calibrated Fraunhofer transform:

$$
C:(x_{\mu\mathrm{m}},y_{\mu\mathrm{m}})\rightarrow(u_{\mathrm{FFT}},v_{\mathrm{FFT}}).
$$

For wavelength $\lambda$, Fourier-lens focal length $f$, SLM pitches
$p_x,p_y$, and computational extents $N_x,N_y$:

$$
u_{\mathrm{FFT}}=\frac{xN_xp_x}{\lambda f},\qquad
v_{\mathrm{FFT}}=-\frac{yN_yp_y}{\lambda f}.
$$

The sign on $v$ converts the physical +y-up convention to row-major image
coordinates. Signed frequencies MUST be retained until an actual FFT array is
indexed; prematurely adding $N_y$ loses fractional precision on float32 GPU
backends. The valid physical field is the Nyquist interval
$|x|,|y|\leq\lambda f/(2p)$.

The MVP MAY use an affine transform. Polynomial distortion correction or a measured lookup map MAY be added later.

### 13.3 Computational grid

- SLM arrays are stored row-major.
- The active physical aperture MAY be embedded in a larger zero-padded FFT grid.
- The active-aperture mask sets incident amplitude to zero outside physical pixels.
- Two-times oversampling SHOULD be supported.
- Arbitrary fractional trap coordinates MUST be evaluated with the exact
  discrete Fourier sum. Bilinear FFT-bin sampling and adjoint scattering MUST
  NOT be used as a physical target-field constraint.

### 13.4 First-frame initialization

For the first trap frame, initialize the hologram from a coherent superposition:

$$
U_{\mathrm{init}}(u,v)
=
\sum_j d_j
\exp\left(i\left[2\pi(f_{x,j}u+f_{y,j}v)+\theta_j\right]\right),
$$

where $d_j=\sqrt{I_j}$.

Then:

$$
\phi_0^{(0)}(u,v)=\arg U_{\mathrm{init}}(u,v).
$$

A deterministic random offset MAY be assigned to the initial target phases to break symmetry. The seed MUST be persisted.

### 13.5 Sequential warm start

Consecutive holograms MUST NOT be solved as unrelated random problems.

For frame $f>0$:

$$
\phi_f^{(0)}=\phi_{f-1}^{\mathrm{accepted}}.
$$

Per-trap WGS weights and target phases MUST be restored by `trapId`:

$$
w_{f,j}^{(0)}=w_{f-1,j}^{\mathrm{accepted}}.
$$

This sequential dependency means frame-level solving is normally serial, although FFT operations inside a frame MAY use SIMD and threads.

### 13.6 Weighted Gerchberg-Saxton iteration

For iteration $k$ of trap frame $f$:

1. Apply the SLM-plane amplitude constraint:

   $$
   U^{(k)}=A_{\mathrm{in}}e^{i\phi^{(k)}}.
   $$

2. Evaluate the exact, unnormalised NUDFT at every target coordinate:

   $$
   v_j^{(k)}=
   \sum_{m=0}^{N_y-1}\sum_{n=0}^{N_x-1}
   U_{m,n}^{(k)}
   \exp\left[-i2\pi\left(\frac{u_jn}{N_x}+\frac{v_jm}{N_y}\right)\right].
   $$

3. Measure every target amplitude:

   $$
   v_j=V^{(k)}(\mathbf{x}_j), \qquad a_j=|v_j|.
   $$

4. Fit a common amplitude scale:

   $$
   s=\frac{\sum_j d_ja_j}{\sum_j d_j^2}.
   $$

5. Update persistent WGS weights:

   $$
   w_j\leftarrow
   w_j\left(\frac{sd_j}{a_j+\epsilon}\right)^\gamma.
   $$

6. Clip and normalize the weights to unit mean.

7. Select the constrained complex target phasors:

   $$
   q_j
   =w_jd_je^{i\theta_j}.
   $$

8. Apply the exact adjoint trap sum at every SLM pixel:

   $$
   B_{m,n}^{(k)}=
   \sum_j q_j
   \exp\left[i2\pi\left(\frac{u_jn}{N_x}+\frac{v_jm}{N_y}\right)\right].
   $$

9. Apply the phase-only constraint:

   $$
   \phi^{(k+1)}=\arg B^{(k)}.
   $$

10. Evaluate convergence and numerical validity with a fresh exact target
    sum. After quantization, decode the actual output codes and evaluate the
    exact target sum again. A full FFT MAY then be run once for full-plane
    power and ghost diagnostics; it is not part of the sparse WGS iteration.

### 13.7 Target-phase modes

| Mode | Behavior |
|---|---|
| `REFERENCE_WGS` | Uses the phase measured at each target in the current iteration. Baseline only; inter-frame phase is not controlled. |
| `PHASE_LOCKED_WGS` | Uses the persistent requested $\theta_j$ for each trap. Recommended default. |
| `SOFT_PHASE_LOCKED_WGS` | Allows bounded relaxation toward the measured phase while penalizing inter-frame phase change. Optional fallback for difficult frames. |
| `PHASE_INTERPOLATED_WGS` | Uses an explicitly unwrapped phase trajectory supplied by the trap-frame generator. |

### 13.8 Trap lifecycle

- A trap that exists in the initial frame SHOULD persist until its atom reaches its final, parking, or release state.
- New empty traps MUST ramp from zero intensity if introduced.
- A trap MUST NOT disappear while it still represents an atom that is meant to remain captured.
- Removed traps MUST complete an intensity ramp before their state is discarded.

### 13.9 Final phase composition

The optimized phase is combined with static corrections:

$$
\phi_{\mathrm{display}}
=
\operatorname{wrap}
(\phi_{\mathrm{hologram}}
+\phi_{\mathrm{aberration}}
+\phi_{\mathrm{grating}}
+\phi_{\mathrm{lens}}).
$$

The calibration manifest defines signs and enabled terms.

### 13.10 Phase quantization

The final wrapped phase MUST be converted through the measured inverse phase-response lookup table:

$$
g(u,v)=\mathrm{LUT}^{-1}(\phi_{\mathrm{display}}(u,v)).
$$

The implementation MUST NOT assume a linear grayscale-to-phase relationship unless the calibration explicitly establishes one.

## 14. Automatic sequence refinement

The sequence compiler MUST treat planning and hologram generation as one coupled compilation process rather than independent tools.

### 14.1 Frame-quality checks

For every solved SLM frame, compute at least:

- target intensity coefficient of variation;
- minimum-to-mean target intensity ratio;
- diffraction efficiency;
- maximum ghost intensity in configured exclusion regions;
- WGS iteration count;
- maximum and RMS target-plane phase change from the previous frame;
- maximum display-code change;
- predicted transition minimum intensity; and
- numerical validity flags.

### 14.2 Adaptive retry ladder

When a frame fails a quality gate, the compiler SHOULD try, in order:

1. continue WGS for additional iterations;
2. reduce WGS damping or use a soft phase lock;
3. reinitialize from a deterministic superposition while preserving target phases;
4. split the preceding motion interval and insert an intermediate trap frame;
5. reduce the number of simultaneously moving atoms in that interval;
6. re-time or replan the local conflict component; and
7. fail compilation with a diagnostic report.

### 14.3 Interval subdivision

Suppose trap frame $T_f$ cannot be connected safely or solved robustly from $T_{f-1}$. The compiler inserts a midpoint trap frame:

$$
T_{f-1/2}=T\left(\frac{t_{f-1}+t_f}{2}\right),
$$

solves it after $T_{f-1}$, then solves $T_f$ from the accepted midpoint state. Subdivision MUST preserve the original continuous trajectory and collision guarantees.

### 14.4 Bounded refinement

The compiler MUST enforce limits including:

- maximum inserted frames;
- maximum WGS iterations per frame;
- maximum local replans;
- maximum total compilation time; and
- maximum output sequence size.

Exceeding a limit returns a structured failure. It MUST never silently skip a frame or emit an unvalidated hologram.

## 15. Sequence compiler state machine

```text
CREATED
  │
  ▼
INPUT_VALIDATED
  │
  ▼
ASSIGNED
  │
  ▼
PATHS_PLANNED
  │
  ▼
TRAJECTORIES_PARAMETERIZED
  │
  ▼
TRAP_FRAMES_READY
  │
  ▼
SOLVING_FRAME_n ──quality failure──► REFINING_INTERVAL_n
  │                                      │
  │ accepted                             └──► SOLVING_FRAME_n
  ▼
STORING_FRAME_n
  │
  ├──more frames──► SOLVING_FRAME_n+1
  │
  ▼
SEQUENCE_VALIDATING
  │
  ├──failure──► FAILED
  ▼
COMPLETE
```

Cancellation is legal from every nonterminal state. After cancellation, the context MUST not be reused without an explicit reset.

## 16. Software architecture

### 16.1 Component responsibilities

| Component | Responsibility |
|---|---|
| TypeScript API | Input validation, progress events, storage selection, UI integration, and hardware-service communication. |
| Wasm geometry core | Coordinate normalization, distance calculations, collision checks, roadmap construction. |
| Wasm assignment module | Cost matrix construction and Hungarian/Jonker-Volgenant assignment. |
| Wasm path planner | Direct planner, reservation-table A*, conflict decomposition, and ECBS fallback. |
| Wasm trajectory module | Minimum-jerk time parameterization and trap-frame sampling. |
| Wasm hologram core | Exact trap NUDFT/adjoint WGS, diagnostic FFT propagation, phase continuity, calibration composition, and metrics. |
| Sequence orchestrator | Couples trap-frame generation to sequential hologram solving and adaptive refinement. |
| Frame store | Persists large trap and SLM frame streams without retaining the entire sequence in Wasm memory. |
| Native hardware adapter | Loads validated frame packets and presents them through a vendor SDK or monitor output. |

### 16.2 Worker model

The browser implementation SHOULD use a dedicated worker so that planning and FFT computation do not block the UI.

Recommended deployment:

```text
Main thread
  ├─ configuration UI
  ├─ progress and preview
  └─ storage / hardware-client coordination

Compiler worker
  ├─ one Wasm instance
  ├─ planner state
  ├─ sequential WGS state
  └─ fixed scratch buffers

Optional Emscripten pthread workers
  └─ FFT rows, columns, transposes, and selected geometry kernels
```

### 16.3 Why one sequential solver context is required

The accepted phase and per-trap weights of frame $f$ initialize frame $f+1$. Therefore, ordinary frame-level parallel solving would break the defined warm-start sequence. Parallelism SHOULD instead be used inside each FFT or for independent planning components.

### 16.4 Native parity

The same C++ core SHOULD compile as:

- a native command-line reference compiler;
- a single-threaded Wasm module;
- a Wasm SIMD module; and
- a Wasm SIMD+pthreads module.

Native and Wasm builds MUST share algorithms, data structures, and test vectors.

## 17. Wasm memory and performance design

### 17.1 Boundary rule

No per-point or per-pixel JavaScript calls are permitted in a hot loop. Inputs and outputs MUST be exchanged through packed arrays in Wasm linear memory.

### 17.2 Packed point input

```cpp
struct InitialAtomPacked {
  uint32_t atom_id;
  float x_um;
  float y_um;
  float localization_sigma_um;
  uint32_t group;
  uint32_t flags;
};

struct TargetSitePacked {
  uint32_t site_id;
  float x_um;
  float y_um;
  float final_intensity;
  uint32_t required_atom_id;
  uint32_t required_group;
  uint32_t flags;
};
```

Use explicit static assertions for size and alignment. The ABI MUST specify little-endian encoding.

### 17.3 Numerical buffers

The hologram core SHOULD allocate and reuse:

- incident amplitude;
- active-aperture mask;
- current phase;
- previous accepted phase;
- complex forward field real/imaginary arrays;
- constrained target field real/imaginary arrays;
- FFT transpose scratch;
- static phase maps;
- output code frame; and
- sparse target and persistent trap-state tables.

No heap allocation is permitted inside an FFT stage, WGS iteration, or collision-validation inner loop.

### 17.4 Numeric types

- Geometry SHOULD use `float64` where accumulated planning error matters.
- FFT and WGS production arrays SHOULD use `float32` for capacity and SIMD performance.
- A `float64` scalar reference solver SHOULD exist for numerical tests.
- Frame times SHOULD use integer microseconds or nanoseconds, not floating-point wall-clock time.

### 17.5 Build variants

Recommended artifacts:

```text
slm_compiler_single.wasm
slm_compiler_simd.wasm
slm_compiler_threads_simd.wasm
slm_compiler.js
slm_compiler.d.ts
```

The runtime MUST select a compatible artifact through feature detection. A threaded browser build depends on the environment required for shared WebAssembly memory and must have a single-threaded fallback.

### 17.6 FFT implementation

The implementation SHOULD begin with:

1. a simple radix-2 scalar FFT used as a correctness reference; and
2. a production FFT selected after benchmarking in both native and Wasm builds.

The 2-D FFT uses row FFTs, transpose, row FFTs, and transpose back. This layout enables contiguous SIMD operations and parallel row scheduling.

## 18. Public C ABI

The exact names MAY change, but the ABI SHOULD expose the following concepts.

```c
uint32_t slm_create_context(const SlmContextConfig* config);
void slm_destroy_context(uint32_t context);

int32_t slm_load_calibration(
    uint32_t context,
    const uint8_t* bytes,
    uint32_t byte_count);

int32_t slm_set_initial_atoms(
    uint32_t context,
    const InitialAtomPacked* atoms,
    uint32_t atom_count);

int32_t slm_set_target_sites(
    uint32_t context,
    const TargetSitePacked* sites,
    uint32_t site_count);

int32_t slm_plan_assignment(uint32_t context);
int32_t slm_plan_paths(uint32_t context);
int32_t slm_generate_trap_frames(uint32_t context);

int32_t slm_begin_sequence_solve(uint32_t context);
int32_t slm_solve_next_frame(uint32_t context);

uint32_t slm_get_frame_count(uint32_t context);
uint32_t slm_get_current_frame_index(uint32_t context);

const TrapStatePacked* slm_get_current_trap_frame_ptr(uint32_t context);
const uint8_t* slm_get_current_slm_frame_ptr(uint32_t context);
const FrameMetricsPacked* slm_get_current_metrics_ptr(uint32_t context);

int32_t slm_accept_current_frame(uint32_t context);
int32_t slm_refine_current_interval(uint32_t context);
int32_t slm_cancel(uint32_t context);

const SlmError* slm_get_last_error(uint32_t context);
```

A convenience call MAY run the full pipeline, but it MUST report progress and permit cancellation.

## 19. TypeScript API

```ts
export interface CompileProgress {
  stage:
    | "VALIDATING"
    | "ASSIGNING"
    | "PLANNING"
    | "PARAMETERIZING"
    | "GENERATING_TRAP_FRAMES"
    | "SOLVING_SLM_FRAMES"
    | "VALIDATING_SEQUENCE"
    | "WRITING_OUTPUT";
  completed: number;
  total: number;
  frameIndex?: number;
  message?: string;
}

export interface CompileOptions {
  signal?: AbortSignal;
  onProgress?: (progress: CompileProgress) => void;
  outputStore?: FrameStore;
}

export class SlmSequenceCompiler {
  static create(config: CompilerConfig): Promise<SlmSequenceCompiler>;

  compileRearrangement(
    request: RearrangementRequest,
    options?: CompileOptions,
  ): Promise<CompiledSequenceHandle>;

  planOnly(request: RearrangementRequest): Promise<MotionPlan>;
  solveTrapFrames(frames: AsyncIterable<TrapFrame>): Promise<CompiledSequenceHandle>;
  dispose(): void;
}
```

## 20. Sequence storage format

A compiled sequence SHOULD be exported as a directory or archive:

```text
sequence/
  manifest.json
  assignment.json
  trajectories.json
  trap-frames.bin
  trap-frames.index.json
  slm-frames.bin
  slm-frames.index.json
  frame-metrics.jsonl
  validation.json
  calibration-manifest.json
```

### 20.1 Manifest fields

The manifest MUST include:

- format version;
- creation timestamp;
- compiler and Wasm build identifiers;
- input hash;
- calibration identifier and hash;
- coordinate convention;
- number of atoms, targets, traps, and frames;
- frame period;
- assignment cost and retry count;
- planner backend and parameters;
- WGS backend and parameters;
- output width, height, and pixel format;
- deterministic seed;
- checksums; and
- validation status.

### 20.2 Browser storage

Large sequences SHOULD be written incrementally to a persistent browser store or downloaded as a streamed artifact. The Wasm heap SHOULD contain only the current working frame, neighboring refinement frames, and reusable numerical buffers.

## 21. Calibration package

The calibration package MUST be versioned and SHOULD contain:

```text
calibration/
  manifest.json
  incident-amplitude.f32
  aperture-mask.u8
  coordinate-transform.json
  aberration-phase.f32
  phase-response-lut.u16
  carrier-grating.f32          # optional
  digital-lens.f32             # optional
  transition-model.json        # optional but recommended
```

The manifest MUST identify:

- SLM model and serial number;
- wavelength;
- active dimensions;
- FFT-grid dimensions;
- pixel pitch;
- optical configuration;
- calibration date;
- checksum of every array; and
- coordinate-transform convention.

A mismatch MUST stop sequence compilation or hardware execution.

## 22. Hardware integration

### 22.1 Default policy: compile before execution

The safest initial implementation is:

1. compile the entire atom-motion and SLM sequence;
2. validate every frame;
3. transfer the finished sequence to the experiment controller;
4. arm the hardware; and
5. execute only the prevalidated frame sequence.

This avoids making browser scheduling part of the experiment's hard-real-time path.

### 22.2 Monitor-mode SLM

For an SLM exposed as a display:

- use a dedicated full-screen surface;
- disable scaling, color correction, interpolation, overlays, and compositing when possible;
- verify one output pixel maps to one SLM pixel;
- use the calibrated grayscale format; and
- synchronize frame advancement according to measured device behavior, not only monitor refresh notifications.

### 22.3 Vendor-SDK SLM

A local native service SHOULD:

- read the validated sequence format;
- verify dimensions, checksums, calibration ID, and protocol version;
- pin or preload frames as supported;
- call the vendor SDK;
- report acknowledgement and hardware errors; and
- hold the last safe frame on underflow or fault.

The numerical compiler MUST remain independent of the vendor SDK.

## 23. Metrics and quality gates

### 23.1 Planning metrics

- selected atom count;
- assignment cost;
- total and maximum path length;
- makespan;
- number of direct paths;
- number and size of conflict components;
- number of waits and detours;
- minimum validated atom separation;
- maximum speed, acceleration, and jerk;
- assignment retries;
- planner node expansions; and
- planning time.

### 23.2 Per-SLM-frame metrics

- WGS iterations;
- convergence status;
- target intensity mean and standard deviation;
- target intensity coefficient of variation;
- minimum-to-mean intensity ratio;
- diffraction efficiency;
- maximum ghost intensity;
- maximum WGS weight;
- maximum target-phase error;
- target-phase change from prior frame;
- display-code change from prior frame;
- estimated transition minimum intensity;
- solve time; and
- refinement count.

### 23.3 Acceptance policy

Thresholds are experiment-defined. The compiler MUST allow separate limits for:

- initial static frame;
- moving frames;
- release frames; and
- final static frame.

No frame with NaN, infinity, invalid calibration, failed checksum, or illegal state transition can be accepted regardless of relaxed optical thresholds.

## 24. Failure model

Recommended error categories:

```cpp
enum class SlmStatus : int32_t {
  OK = 0,
  INVALID_ARGUMENT,
  INSUFFICIENT_ATOMS,
  DUPLICATE_ID,
  OUT_OF_BOUNDS,
  INVALID_TARGET_GEOMETRY,
  CALIBRATION_MISMATCH,
  ASSIGNMENT_INFEASIBLE,
  PATH_NOT_FOUND,
  COLLISION_VALIDATION_FAILED,
  MOTION_LIMIT_VIOLATION,
  FRAME_LIMIT_EXCEEDED,
  NUMERIC_ERROR,
  WGS_NOT_CONVERGED,
  FRAME_QUALITY_REJECTED,
  STORAGE_ERROR,
  CANCELLED,
  INTERNAL_ERROR,
};
```

Every error MUST include:

- pipeline stage;
- offending atom, target, interval, or frame if applicable;
- configured and measured values;
- whether retry is possible; and
- a stable machine-readable code.

## 25. Verification and testing

### 25.1 Assignment tests

- one atom and one target;
- rectangular $N>M$ assignment;
- atoms already on targets;
- fixed identity;
- group constraints;
- dummy destinations for extras;
- deterministic tie breaking;
- infeasible identity requirements; and
- known cost matrices with known optimal assignments.

### 25.2 Path-planning tests

- nonconflicting direct paths;
- two paths crossing at the same time;
- an edge swap;
- narrow passage;
- static occupied obstacle;
- wait-required case;
- detour-required case;
- parking-site use;
- multiple disconnected conflict components;
- prioritized-planner failure with CBS success;
- impossible geometry; and
- deterministic replay.

### 25.3 Continuous trajectory tests

- exact initial and final positions;
- zero endpoint velocity and acceleration;
- velocity, acceleration, and jerk bounds;
- monotonic segment progress;
- no pairwise separation violation;
- no forbidden-region intersection; and
- exact sampling of final frame.

### 25.4 FFT and WGS tests

- impulse FFT;
- constant field FFT;
- forward/inverse round trip;
- comparison to a trusted native or NumPy reference;
- exact fractional-frequency NUDFT comparison to an independent direct sum;
- exported indexed-BMP decode followed by an independent off-grid Fourier
  field oracle;
- uniform target array;
- nonuniform requested intensities;
- moving single trap;
- moving multiple traps;
- persistent phase by trap ID;
- trap release ramp;
- static phase composition; and
- LUT quantization endpoints.

### 25.5 End-to-end tests

At minimum, include:

1. **Identity case:** initial and target arrays are equal. The plan contains only dwell frames and stable holograms.
2. **Translation case:** every atom moves by the same displacement.
3. **Permutation case:** target occupancy is identical but identities are constrained to swap.
4. **Crossing case:** direct paths collide and the planner inserts waits or detours.
5. **Sparse-to-compact case:** a partially occupied reservoir is rearranged into a compact target array.
6. **Excess-atom case:** selected atoms fill targets and extras are parked and released.
7. **Dense case:** conflict-component decomposition and MAPF are exercised.
8. **Adaptive-refinement case:** a deliberately large trap step causes midpoint insertion before acceptance.
9. **Calibration rejection case:** dimensions or wavelength mismatch.
10. **Wasm/native parity case:** assignments, trajectories, and output metrics agree within specified tolerances.

## 26. End-to-end acceptance criteria

A compiled sequence is accepted only when all of the following hold:

1. Every required target has exactly one assigned atom.
2. Every selected atom has one continuous trap trajectory.
3. Final coordinates match target coordinates within `finalPositionToleranceUm`.
4. Continuous pairwise clearance is at least the configured safe separation.
5. No trajectory enters a forbidden region.
6. All motion limits are satisfied.
7. Every trap frame passes lifecycle and geometry validation.
8. Every trap frame has exactly one accepted SLM frame.
9. Every SLM frame passes numerical and configured optical quality gates.
10. The output package passes checksums and manifest consistency checks.
11. Recompilation with identical inputs and build settings reproduces the same logical result.

## 27. Performance and capacity planning

### 27.1 Assignment complexity

The dense Hungarian algorithm is approximately $O(n^3)$ for $n=\max(N,M)$. This is normally small compared with solving many large FFT-based holograms, but it must still be benchmarked for large arrays.

### 27.2 Path-planning complexity

MAPF has difficult worst cases. The tiered design controls practical cost by:

- accepting direct paths when safe;
- solving only connected conflict components;
- using prioritized A* first;
- invoking ECBS only for unresolved components;
- permitting movement batches; and
- bounding all searches.

### 27.3 Hologram cost

For $F$ trap frames, $K$ WGS iterations per frame, $P=N_xN_y$ SLM cells, and
$T$ traps, exact sparse trap-domain WGS costs approximately:

$$
O(FKPT).
$$

The optional final full-plane diagnostic FFT adds $O(FP\log P)$.

Warm starting is therefore essential. The first frame can receive a larger iteration budget, while later frames SHOULD use fewer iterations unless a quality gate fails.

### 27.4 Approximate working memory

For a 1024×1024 float32 computational grid:

- one scalar field is approximately 4 MiB;
- one complex split field is approximately 8 MiB;
- multiple reusable FFT and phase buffers can require tens of MiB;
- one uint8 output frame is approximately 1 MiB.

For 2048×2048, these values are four times larger. The implementation MUST compute an explicit memory budget before allocating buffers.

### 27.5 Benchmark matrix

Benchmark at least:

- 32, 100, 500, 1000, and 2000 atoms where practical;
- sparse and dense target geometries;
- 512², 1024², and 2048² FFT grids;
- 1, 2, 4, 8, and 12 WGS iterations;
- direct-only and MAPF-heavy rearrangements;
- native scalar, native threaded, Wasm scalar, Wasm SIMD, and Wasm threads+SIMD; and
- cold start versus warm sequence frames.

Report medians and tail latencies separately.

## 28. Implementation roadmap

### Milestone 1: schemas and deterministic geometry core

Deliver:

- canonical input/output types;
- coordinate and calibration validation;
- deterministic identifiers;
- distance, segment, and forbidden-region kernels;
- native and Wasm test harnesses.

Exit criterion: identical normalized inputs and geometry results in native and Wasm builds.

### Milestone 2: assignment

Deliver:

- cost-matrix construction;
- Hungarian or Jonker-Volgenant solver;
- fixed identity and group constraints;
- extra-atom dummy destinations;
- assignment diagnostics.

Exit criterion: all assignment unit tests pass and known optima are reproduced.

### Milestone 3: direct trajectories and trap frames

Deliver:

- straight paths;
- quintic minimum-jerk time parameterization;
- synchronized and scheduled starts;
- intensity ramps;
- trap-frame sampling;
- continuous collision validator.

Exit criterion: nonconflicting examples compile from two point arrays to validated trap frames.

### Milestone 4: MAPF fallback

Deliver:

- planning graph builders;
- reservation-table A*;
- conflict graph decomposition;
- deterministic priority retries;
- ECBS/CBS fallback for bounded component sizes;
- parking and batch serialization.

Exit criterion: crossing, swap, and dense test cases produce collision-free trajectories or explicit infeasibility.

### Milestone 5: scalar optical reference

Deliver:

- scalar FFT;
- coordinate mapping;
- WGS;
- phase-locked sequence state;
- calibration phase composition;
- uint8/uint16 output.

Exit criterion: a supplied trap-frame sequence produces numerically valid SLM frames in native mode.

### Milestone 6: Wasm optical implementation

Deliver:

- fixed Wasm memory layout;
- TypeScript bindings;
- worker execution;
- SIMD kernels;
- optional pthread FFT;
- native/Wasm parity tests.

Exit criterion: the browser compiles a full small rearrangement and reproduces reference metrics.

### Milestone 7: coupled sequence refinement

Deliver:

- frame-quality gates;
- retry ladder;
- automatic midpoint insertion;
- local movement rescheduling;
- bounded failure reports.

Exit criterion: deliberately difficult frame transitions are refined automatically or rejected safely.

### Milestone 8: persistent output and hardware bridge

Deliver:

- streamed sequence package;
- checksums and manifest;
- mock hardware adapter;
- monitor-mode adapter or vendor-service protocol;
- acknowledgement and hold-last-frame behavior.

Exit criterion: a precompiled sequence is replayed without illegal state transitions or missing frames.

### Milestone 9: experimental calibration and validation

Deliver:

- real coordinate transform;
- incident-amplitude map;
- aberration map;
- phase LUT;
- transition model;
- camera-based offline verification reports.

Exit criterion: experimentally defined trap intensity, continuity, and rearrangement acceptance gates pass.

## 29. Recommended MVP defaults

These values are software starting points, not universal experimental settings.

| Parameter | Initial default |
|---|---|
| Dimensions | 2-D focal plane |
| Assignment | Hungarian, squared-distance cost plus obstacle penalties |
| Direct planner | Enabled |
| Conflict fallback | Prioritized space-time A* with deterministic retries |
| Secondary fallback | ECBS for bounded conflict components |
| Trajectory interpolation | Quintic minimum jerk with stops at graph waypoints |
| Extra atom policy | `PARK_AND_RELEASE` when parking sites are supplied; otherwise `KEEP` |
| Trap frame period | Configuration/calibration-defined |
| Target phase mode | `PHASE_LOCKED_WGS` |
| First-frame WGS iterations | 12 |
| Subsequent-frame WGS iterations | 4 |
| WGS damping $\gamma$ | 0.7 |
| WGS epsilon | $10^{-8}$ |
| WGS weight limits | 0.1 to 10.0 |
| FFT grid | 1024×1024, configurable |
| Numeric type | float32 optical core; float64 geometry where needed |
| Output format | calibrated uint8 initially |
| Solver execution | dedicated worker |
| Sequence mode | complete offline precomputation |
| Adaptive refinement | enabled |
| Maximum refinement depth | configuration-defined |
| Hardware path | native adapter preferred for experiment execution |

## 30. Reference end-to-end pseudocode

```text
CompileRearrangement(request):
    input = normalizeAndValidate(request)
    calibration = loadAndValidateCalibration(input.calibrationId)

    for assignmentAttempt in 0 .. maxAssignmentRetries:
        costMatrix = buildAssignmentCost(input, conflictPenalties)
        assignment = hungarianSolve(costMatrix, identityConstraints)

        if assignment is infeasible:
            return ASSIGNMENT_INFEASIBLE

        directPaths = buildSynchronizedDirectPaths(assignment)
        directConflicts = findContinuousConflicts(directPaths)

        if directConflicts is empty:
            paths = directPaths
        else:
            components = connectedConflictComponents(directConflicts)
            paths = directPaths

            for component in components:
                componentPlan = prioritizedSpaceTimeAStarWithRetries(component)

                if componentPlan failed and component.size <= ecbsLimit:
                    componentPlan = ECBS(component)

                if componentPlan failed:
                    componentPlan = serializeIntoMovementBatches(component)

                if componentPlan failed:
                    addAssignmentConflictPenalties(component, conflictPenalties)
                    goto retry_assignment

                replaceComponentPaths(paths, componentPlan)

        trajectories = minimumJerkTimeParameterize(paths, motionLimits)

        if not validateContinuousTrajectories(trajectories):
            addTrajectoryConflictPenalties(conflictReport, conflictPenalties)
            goto retry_assignment

        break

    if no feasible trajectories:
        return PATH_NOT_FOUND

    trapFrames = sampleTrapFrames(
        trajectories,
        staticTraps,
        intensityProfiles,
        targetPhasePolicy,
        slmFramePeriod)

    validateAllTrapFrames(trapFrames)

    solver.beginSequence(calibration, hologramConfig)
    frameIndex = 0

    while frameIndex < trapFrames.count:
        trapFrame = trapFrames[frameIndex]
        solveResult = solver.solveSequentialFrame(trapFrame)

        if solveResult passes quality gates:
            storeAcceptedSlmFrame(frameIndex, solveResult)
            solver.commitFrameState()
            frameIndex += 1
            continue

        if solveResult can retry with more iterations:
            solver.retryCurrentFrameWithMoreIterations()
            continue

        if intervalBefore(frameIndex) can be subdivided:
            midpoint = sampleTrajectoryAtIntervalMidpoint(trajectories, frameIndex)
            trapFrames.insert(frameIndex, midpoint)
            solver.rollbackToPreviousAcceptedFrame()
            continue

        if local movement can be rescheduled:
            trajectories = replanLocalInterval(trajectories, frameIndex)
            trapFrames = regenerateAffectedTrapFrames(trajectories)
            solver.rollbackToLastUnaffectedFrame()
            continue

        return FRAME_QUALITY_REJECTED

    validation = validateCompleteSequence(
        assignment,
        trajectories,
        trapFrames,
        slmFrameStore,
        metrics)

    if not validation.accepted:
        return validation.error

    writeManifestAndChecksums()
    return COMPLETE
```

## 31. Reference WGS pseudocode

```text
SolveSequentialSlmFrame(frame):
    validateTrapFrame(frame)

    if first accepted frame:
        phi = initializeByComplexSuperposition(frame.traps, deterministicSeed)
        weights = 1.0
        targetPhases = initializePersistentTargetPhases(frame.traps)
    else:
        phi = previousAcceptedPhi
        weights = restoreWeightsByTrapId(frame.traps)
        targetPhases = restorePhasesByTrapId(frame.traps)

    targets = mapPhysicalCoordinatesToFft(frame.traps, calibration)

    for iteration in 0 .. iterationBudget(frame)-1:
        U = incidentAmplitude * exp(i * phi)
        measured = exactNudftTargets(U, targets)
        scale = fitGlobalAmplitudeScale(measured, frame.traps)
        weights = updateClipAndNormalizeWeights(
            weights,
            measured,
            frame.traps,
            scale,
            gamma,
            epsilon)

        for each trap j:
            targetPhase = chooseTargetPhaseByMode(
                trapId=j.id,
                persistentPhase=targetPhases[j.id],
                measuredPhase=arg(measured[j]))

            targetPhasor[j] = (
                weights[j] * sqrt(frame.traps[j].intensity)
                * exp(i * targetPhase))

        phi = arg(exactAdjointTrapSum(targetPhasor, targets))

        iterationMetrics = evaluateWgsIteration(measured, weights)

        if invalid(iterationMetrics):
            return NUMERIC_ERROR

        if converged(iterationMetrics):
            break

    displayPhase = wrap(
        phi
        + aberrationPhase
        + carrierGrating
        + digitalLens)

    outputCodes = inversePhaseLut(displayPhase)
    decodedOutput = phaseResponseLut(outputCodes)
    finalTargets = exactNudftTargets(
        incidentAmplitude * exp(i * decodedOutput),
        targets)
    finalMetrics = evaluateFrameAndTransition(
        outputCodes,
        finalTargets,
        frame,
        previousAcceptedState)

    if not passesQualityGates(finalMetrics):
        return FRAME_QUALITY_REJECTED with retry information

    return candidate frame, state, and metrics
```

## 32. Suggested repository layout

```text
slm-sequence-compiler/
  core/
    common/
      status.h
      span.h
      deterministic_rng.h
    geometry/
      point.h
      segment.cpp
      polygon.cpp
      clearance.cpp
      continuous_collision.cpp
    assignment/
      cost_matrix.cpp
      hungarian.cpp
      assignment_repair.cpp
    planning/
      planning_graph.cpp
      lattice_graph.cpp
      roadmap.cpp
      direct_planner.cpp
      reservation_table.cpp
      space_time_astar.cpp
      conflict_graph.cpp
      cbs.cpp
      ecbs.cpp
      batch_scheduler.cpp
    trajectory/
      minimum_jerk.cpp
      time_parameterization.cpp
      intensity_profile.cpp
      frame_sampler.cpp
      trajectory_validator.cpp
    optics/
      coordinate_map.cpp
      propagation.cpp
      target_sampling.cpp
      phase_composition.cpp
      phase_lut.cpp
    fft/
      fft_reference.cpp
      fft_production.cpp
      fft2d.cpp
      transpose.cpp
      simd_complex.h
    solver/
      wgs_reference.cpp
      wgs_phase_locked.cpp
      convergence.cpp
      transition_metrics.cpp
    sequence/
      sequence_compiler.cpp
      adaptive_refinement.cpp
      trap_state.cpp
      manifest.cpp
      validation.cpp
    calibration/
      calibration.cpp
      calibration_manifest.cpp
    bindings/
      c_api.cpp
      wasm_api.cpp

  web/
    compiler-worker.ts
    compiler-client.ts
    wasm-loader.ts
    frame-store.ts
    sequence-export.ts
    preview.ts
    monitor-output.ts

  native-host/
    main.cpp
    sequence-reader.cpp
    hardware-protocol.h
    slm-adapter.h
    mock-adapter.cpp
    vendor-adapter.cpp

  tests/
    assignment/
    planning/
    trajectory/
    optics/
    solver/
    sequence/
    wasm-parity/

  benchmarks/
    bench-assignment.cpp
    bench-planning.cpp
    bench-fft.cpp
    bench-sequence.cpp
    browser-benchmark.ts

  examples/
    simple-translation.json
    crossing-paths.json
    sparse-to-compact.json
    excess-atoms.json

  calibration/
    example/

  CMakeLists.txt
  Makefile
  package.json
  README.md
```

## 33. Risks and mitigations

| Risk | Consequence | Mitigation |
|---|---|---|
| Assignment minimizes distance but produces difficult conflicts | Slow or failed planning | Conflict-aware cost terms, repair loop, parking sites, and local MAPF. |
| Dense MAPF component grows rapidly | Long compile time | Direct fast path, conflict decomposition, prioritized retries, ECBS bound, and batch serialization. |
| Geometric smoothing reintroduces collisions | Atom loss | Continuous post-smoothing validation with adaptive subdivision and safety margins. |
| SLM transition causes transient trap weakening | Atom loss during frame change | Persistent target phase, warm-started WGS, measured transition model, frame subdivision, and movement slowdown. |
| Per-frame WGS is too slow | Excessive compilation time | Warm starts, few subsequent iterations, SIMD/threads, lower FFT grid where acceptable, and offline compilation. |
| Browser memory is exhausted by all SLM frames | Crash or failed export | Incremental frame store; retain only working buffers in Wasm. |
| Browser timing jitter affects execution | Incorrect frame timing | Precompile and replay through a native hardware adapter. |
| Calibration is stale or mismatched | Incorrect trap positions or phase | Versioned calibration, checksums, strict compatibility validation, and experimental revalidation. |
| Input localization is uncertain | Clearance violations | Include uncertainty in geometric safety radii. |
| Software model passes but physical experiment fails | Atom loss or nonuniform traps | Camera-based offline verification, conservative gates, and explicit calibration-dependent limits. |

## 34. References

1. R. Di Leonardo, F. Ianni, and G. Ruocco, “Computer generation of optimal holograms for optical trap arrays,” *Optics Express*, 15(4), 1913–1922, 2007.
2. H. Kim, M. Kim, W. Lee, and J. Ahn, “Gerchberg-Saxton algorithm for fast and efficient atom rearrangement in optical tweezer traps,” *Optics Express*, 27(3), 2184–2196, 2019.
3. W. Lee, H. Kim, and J. Ahn, “Defect-free atomic array formation using the Hungarian matching algorithm,” *Physical Review A*, 95, 053424, 2017.
4. G. Sharon, R. Stern, A. Felner, and N. R. Sturtevant, “Conflict-based search for optimal multi-agent pathfinding,” *Artificial Intelligence*, 219, 40–66, 2015.
5. M. Phillips and M. Likhachev, “SIPP: Safe interval path planning for dynamic environments,” *IEEE International Conference on Robotics and Automation*, 2011.
6. M. Endres et al., “Atom-by-atom assembly of defect-free one-dimensional cold atom arrays,” *Science*, 354, 1024–1027, 2016.
7. D. Barredo et al., “An atom-by-atom assembler of defect-free arbitrary two-dimensional atomic arrays,” *Science*, 354, 1021–1023, 2016.
8. Emscripten documentation, “Using SIMD with WebAssembly.”
9. Emscripten documentation, “Pthreads support.”

## 35. Final implementation definition

The MVP is complete only when a caller can provide an initial occupied point array and a target point array and receive, without manually specifying trajectories or holograms:

1. a valid atom assignment;
2. collision-free atom paths;
3. all intermediate trap frames;
4. all calibrated SLM frames;
5. complete quality metrics and validation results; and
6. a replayable sequence package for the experiment controller.

That full transformation is the core feature of the system.
