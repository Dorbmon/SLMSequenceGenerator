# SLM Sequence Compiler

This repository contains a dependency-free TypeScript reference implementation
of the pipeline in `design.md`. It is intentionally scalar and suitable for
native or browser reference tests; an FFT/Wasm backend can be substituted
behind the exported stage interfaces later.

```ts
import { SlmSequenceCompiler } from "slm-sequence-compiler";

const compiler = await SlmSequenceCompiler.create({ calibration });
const sequence = await compiler.compileRearrangement({
  initialAtoms: [{ xUm: 0, yUm: 0 }],
  targetSites: [{ xUm: 10, yUm: 0 }],
  calibrationId: calibration.manifest.calibrationId,
});
```

The compiler keeps frame data in `MemoryFrameStore` by default. A caller can
provide another store implementing the exported `FrameStore` interface.

Measured runs must provide a calibration package with a phase-response or
inverse phase LUT. Set `simulationMode: true` only for the synthetic identity
calibration used by reference simulations. Set `hologram.requireConvergence`
to `true` when a run must reject frames that exhaust their WGS budget.
