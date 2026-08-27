import { describe, expect, it } from "vitest";
import { TerminalProgressReporter, formatProgress } from "./progress.js";

class FakeOutput {
  readonly chunks: string[] = [];

  constructor(
    readonly isTTY: boolean,
    readonly columns = 120,
  ) {}

  write(chunk: string): boolean {
    this.chunks.push(chunk);
    return true;
  }

  text(): string {
    return this.chunks.join("");
  }
}

function progress(acceptedSamples: number, requestedSamples = 10_000) {
  return {
    kind: "PROGRESS",
    progress: {
      acceptedSamples,
      requestedSamples,
      trapCount: 42,
      acceptedPerHour: 1_200,
      rejectedTotal: 3,
    },
  };
}

describe("terminal dataset progress", () => {
  it("rewrites one interactive line and terminates it on completion", () => {
    const output = new FakeOutput(true);
    const reporter = new TerminalProgressReporter(output);
    reporter.handle({ kind: "DAWN_READY", options: ["backend=metal"] });
    reporter.handle(progress(2_500));
    reporter.handle(progress(5_000));
    reporter.handle({ ...progress(10_000), kind: "COMPLETE" });

    expect(output.text()).toContain("Dawn/WebGPU ready (backend=metal)\n");
    expect(output.text()).toContain("\r\u001b[2K");
    expect(output.text()).toContain(" 50.0% 5000/10000");
    expect(output.text()).toContain("100.0% 10000/10000");
    expect(output.text().endsWith("\n")).toBe(true);
  });

  it("throttles redirected output to approximately one update per percent", () => {
    const output = new FakeOutput(false);
    const reporter = new TerminalProgressReporter(output);
    for (let accepted = 0; accepted <= 10_000; accepted += 1) {
      reporter.handle(progress(accepted));
    }
    reporter.handle({ ...progress(10_000), kind: "COMPLETE" });

    const lines = output.text().trim().split("\n");
    expect(lines).toHaveLength(101);
    expect(lines[0]).toContain("0.0% 0/10000");
    expect(lines.at(-1)).toContain("100.0% 10000/10000");
  });

  it("includes throughput, rejection count, and ETA", () => {
    expect(formatProgress({
      acceptedSamples: 2_500,
      requestedSamples: 10_000,
      trapCount: 42,
      acceptedPerHour: 1_200,
      rejectedTotal: 3,
    })).toContain("traps 42 | rejected 3 | 1,200/h | ETA 6h 15m");
  });
});
