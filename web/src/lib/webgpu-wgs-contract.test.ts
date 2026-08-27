import { describe, expect, it, vi } from "vitest";

describe("WebGPU WGS resource contract", () => {
  it("does not initialize WebAssembly when the WebGPU-only modules are loaded", async () => {
    vi.resetModules();
    const wasmModule = vi.spyOn(WebAssembly, "Module");
    try {
      await Promise.all([
        import("./webgpu-wgs.js"),
        import("./optical-calibration.js"),
      ]);
      expect(wasmModule).not.toHaveBeenCalled();
    } finally {
      wasmModule.mockRestore();
    }
  });

  it("binds target state while initializing the phase field", async () => {
    const { WEBGPU_WGS_PIPELINE_BINDINGS } = await import("./webgpu-wgs.js");
    // initialize_targets writes each persistent/synthesis phase to binding 2;
    // initialize_phase must read the same buffer before its first NUDFT pass.
    expect(WEBGPU_WGS_PIPELINE_BINDINGS.initialize_phase).toContain(2);
  });
});
