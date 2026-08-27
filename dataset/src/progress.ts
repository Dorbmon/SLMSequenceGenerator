interface ProgressOutput {
  readonly isTTY?: boolean;
  readonly columns?: number;
  write(chunk: string): unknown;
}

interface ProgressData {
  acceptedSamples: number;
  requestedSamples: number;
  trapCount: number;
  acceptedPerHour: number;
  rejectedTotal: number;
}

const PLAIN_PROGRESS_INTERVAL_MS = 60_000;

/** Render the runner's JSON messages without flooding the user's terminal. */
export class TerminalProgressReporter {
  private readonly interactive: boolean;
  private liveLine = false;
  private finished = false;
  private latestProgress: ProgressData | null = null;
  private lastPlainAccepted = -1;
  private lastPlainWriteMs = 0;

  constructor(private readonly output: ProgressOutput) {
    this.interactive = output.isTTY === true;
  }

  handle(message: unknown): void {
    if (!isRecord(message) || typeof message.kind !== "string") return;

    switch (message.kind) {
      case "DAWN_READY": {
        const options = Array.isArray(message.options)
          ? message.options.filter((value): value is string => typeof value === "string")
          : [];
        this.writeLine(`Dawn/WebGPU ready (${options.length > 0 ? options.join(", ") : "auto"})`);
        break;
      }
      case "STATUS":
        if (this.interactive && this.latestProgress === null && typeof message.phase === "string") {
          this.writeLive(`[dataset] ${formatPhase(message.phase)}`);
        }
        break;
      case "PROGRESS": {
        const progress = parseProgress(message.progress);
        if (progress) this.renderProgress(progress);
        break;
      }
      case "COMPLETE": {
        const progress = parseProgress(message.progress);
        if (progress) this.renderProgress(progress, true);
        this.finish();
        break;
      }
      case "CANCELLED": {
        const progress = parseProgress(message.progress);
        if (progress) this.renderProgress(progress, true);
        this.finish();
        break;
      }
      case "ERROR":
        this.finish();
        if (typeof message.message === "string") this.writeLine(`Runner error: ${message.message}`);
        break;
      default:
        break;
    }
  }

  finish(): void {
    if (this.finished) return;
    if (this.liveLine) this.output.write("\n");
    this.liveLine = false;
    this.finished = true;
  }

  private renderProgress(progress: ProgressData, force = false): void {
    this.latestProgress = progress;
    const line = formatProgress(progress, this.output.columns);
    if (this.interactive) {
      this.writeLive(line);
      return;
    }

    const now = Date.now();
    const stride = Math.max(1, Math.ceil(progress.requestedSamples / 100));
    const advancedEnough = progress.acceptedSamples - this.lastPlainAccepted >= stride;
    const waitedLongEnough = now - this.lastPlainWriteMs >= PLAIN_PROGRESS_INTERVAL_MS;
    const boundary = progress.acceptedSamples === 0
      || progress.acceptedSamples === progress.requestedSamples;
    if (force || boundary || advancedEnough || waitedLongEnough) {
      if (progress.acceptedSamples !== this.lastPlainAccepted) {
        this.output.write(`${line}\n`);
        this.lastPlainAccepted = progress.acceptedSamples;
        this.lastPlainWriteMs = now;
      }
    }
  }

  private writeLive(line: string): void {
    this.output.write(`\r\u001b[2K${fitToTerminal(line, this.output.columns)}`);
    this.liveLine = true;
  }

  private writeLine(line: string): void {
    if (this.liveLine) this.output.write("\r\u001b[2K");
    this.output.write(`${line}\n`);
    this.liveLine = false;
  }
}

export function formatProgress(progress: ProgressData, columns = 120): string {
  const total = Math.max(1, Math.trunc(progress.requestedSamples));
  const accepted = Math.max(0, Math.min(total, Math.trunc(progress.acceptedSamples)));
  const ratio = accepted / total;
  const barWidth = Math.max(10, Math.min(30, Math.floor((columns || 120) / 4)));
  const filled = Math.round(ratio * barWidth);
  const bar = `${"=".repeat(filled)}${".".repeat(barWidth - filled)}`;
  const percent = `${(ratio * 100).toFixed(1).padStart(5)}%`;
  const rate = progress.acceptedPerHour > 0
    ? `${formatRate(progress.acceptedPerHour)}/h`
    : "--/h";
  const remaining = total - accepted;
  const etaHours = progress.acceptedPerHour > 0 ? remaining / progress.acceptedPerHour : Infinity;
  const eta = Number.isFinite(etaHours) ? formatDurationHours(etaHours) : "--";
  return `[${bar}] ${percent} ${accepted}/${total}`
    + ` | traps ${Math.max(0, Math.trunc(progress.trapCount))}`
    + ` | rejected ${Math.max(0, Math.trunc(progress.rejectedTotal))}`
    + ` | ${rate} | ETA ${eta}`;
}

function parseProgress(value: unknown): ProgressData | null {
  if (!isRecord(value)) return null;
  const acceptedSamples = finiteNumber(value.acceptedSamples);
  const requestedSamples = finiteNumber(value.requestedSamples);
  const trapCount = finiteNumber(value.trapCount);
  const acceptedPerHour = finiteNumber(value.acceptedPerHour);
  const rejectedTotal = finiteNumber(value.rejectedTotal);
  if (acceptedSamples === null || requestedSamples === null || requestedSamples <= 0
    || trapCount === null || acceptedPerHour === null || rejectedTotal === null) {
    return null;
  }
  return { acceptedSamples, requestedSamples, trapCount, acceptedPerHour, rejectedTotal };
}

function formatRate(rate: number): string {
  if (rate >= 100) return Math.round(rate).toLocaleString("en-US");
  if (rate >= 10) return rate.toFixed(1);
  return rate.toFixed(2);
}

function formatDurationHours(hours: number): string {
  const seconds = Math.max(0, Math.round(hours * 3600));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const wholeHours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (wholeHours < 24) return `${wholeHours}h ${remainingMinutes}m`;
  const days = Math.floor(wholeHours / 24);
  return `${days}d ${wholeHours % 24}h`;
}

function formatPhase(phase: string): string {
  return phase.toLowerCase().replaceAll("_", " ");
}

function fitToTerminal(line: string, columns: number | undefined): string {
  if (!columns || columns < 20 || line.length < columns) return line;
  return `${line.slice(0, columns - 2)}…`;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
