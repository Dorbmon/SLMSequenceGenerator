import { describe, expect, it } from "vitest";
import { WEBGPU_WGS_PIPELINE_BINDINGS } from "./webgpu-wgs.js";

describe("WebGPU WGS resource contract", () => {
  it("binds target state while initializing the phase field", () => {
    // initialize_targets writes each persistent/synthesis phase to binding 2;
    // initialize_phase must read the same buffer before its first NUDFT pass.
    expect(WEBGPU_WGS_PIPELINE_BINDINGS.initialize_phase).toContain(2);
  });
});
