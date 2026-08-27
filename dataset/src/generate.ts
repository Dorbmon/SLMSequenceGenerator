import { create, globals } from "webgpu";

interface DawnBootstrapConfig {
  backend: string;
  dawn: {
    backend: string | null;
    adapter: string | null;
    options: string[];
  };
}

const collectorUrl = parseCollectorUrl(process.argv.slice(2));
const bootstrap = await fetchBootstrapConfig(collectorUrl);
if (bootstrap.backend !== "DAWN_WEBGPU") {
  throw new Error(`Collector backend must be DAWN_WEBGPU, got ${bootstrap.backend}`);
}

const dawnOptions = [
  ...(bootstrap.dawn.backend ? [`backend=${bootstrap.dawn.backend}`] : []),
  ...(bootstrap.dawn.adapter ? [`adapter=${bootstrap.dawn.adapter}`] : []),
  ...bootstrap.dawn.options,
];

Object.assign(globalThis, globals);
const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, "navigator");
let dawnGpu: GPU | undefined = create(dawnOptions);
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  enumerable: true,
  value: { gpu: dawnGpu },
});

const runtime = await import("./generator-core.js");
let signalExitCode: number | undefined;
const cancelForSignal = (exitCode: number): void => {
  signalExitCode ??= exitCode;
  runtime.requestDatasetCancellation();
};
process.once("SIGINT", () => cancelForSignal(130));
process.once("SIGTERM", () => cancelForSignal(143));

try {
  process.stdout.write(`${JSON.stringify({
    kind: "DAWN_READY",
    options: dawnOptions,
    runtime: "dawn.node",
  })}\n`);
  await runtime.runDatasetGeneration(collectorUrl);
  if (signalExitCode !== undefined) process.exitCode = signalExitCode;
} catch (error) {
  const normalized = error instanceof Error ? error : new Error(String(error));
  process.stderr.write(`${normalized.stack ?? `${normalized.name}: ${normalized.message}`}\n`);
  process.exitCode = signalExitCode ?? 1;
} finally {
  runtime.disposeDatasetRuntime();
  delete (globalThis as { navigator?: unknown }).navigator;
  if (originalNavigator) Object.defineProperty(globalThis, "navigator", originalNavigator);
  dawnGpu = undefined;
}

function parseCollectorUrl(args: string[]): string {
  let value: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    if (args[index] === "--collector") {
      value = args[index + 1];
      index += 1;
      continue;
    }
    throw new Error(`Unknown Dawn runner argument: ${args[index]}`);
  }
  if (!value) throw new Error("Usage: generate.ts --collector http://127.0.0.1:8765");
  const url = new URL(value);
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost"].includes(url.hostname)) {
    throw new Error("Collector URL must be loopback HTTP (127.0.0.1 or localhost)");
  }
  url.pathname = "";
  url.search = "";
  url.hash = "";
  return url.toString().replace(/\/$/, "");
}

async function fetchBootstrapConfig(collector: string): Promise<DawnBootstrapConfig> {
  const response = await fetch(`${collector}/api/config`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Collector bootstrap request failed with HTTP ${response.status}`);
  const value = await response.json() as Partial<DawnBootstrapConfig>;
  if (!value.dawn || !Array.isArray(value.dawn.options)
    || !value.dawn.options.every((option) => typeof option === "string" && option.length > 0)) {
    throw new Error("Collector returned an invalid Dawn bootstrap configuration");
  }
  if (value.dawn.backend !== null && typeof value.dawn.backend !== "string") {
    throw new Error("Collector Dawn backend must be a string or null");
  }
  if (value.dawn.adapter !== null && typeof value.dawn.adapter !== "string") {
    throw new Error("Collector Dawn adapter must be a string or null");
  }
  return value as DawnBootstrapConfig;
}
