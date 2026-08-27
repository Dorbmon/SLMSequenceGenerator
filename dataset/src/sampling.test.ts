import { describe, expect, it } from "vitest";
import {
  Uint32Prng,
  generateSampleLayout,
  sampleTrapCount,
  sampleTrapPositions,
  type PositionSamplingConfig,
} from "./sampling.js";
import {
  SAMPLE_PAYLOAD_HEADER_BYTES,
  SAMPLE_PAYLOAD_MAGIC,
  SAMPLE_PAYLOAD_VERSION,
  crc32,
  encodeSamplePayload,
} from "./protocol.js";

const broadSampling: PositionSamplingConfig = {
  bounds: { minXUm: -500, maxXUm: 500, minYUm: -400, maxYUm: 400 },
  minimumSpacingXUm: 2,
  minimumSpacingYUm: 3,
  zeroOrderExclusion: { radiusXUm: 12, radiusYUm: 18 },
  maxAttemptsPerPoint: 256,
};

describe("deterministic dataset sampling", () => {
  it("uses a stable uint32 PRNG stream", () => {
    const first = new Uint32Prng(0x1234_5678);
    const second = new Uint32Prng(0x1234_5678);
    const a = Array.from({ length: 8 }, () => first.nextUint32());
    const b = Array.from({ length: 8 }, () => second.nextUint32());
    expect(a).toEqual(b);
    expect(a).toEqual([
      455919406,
      4042750857,
      4036713555,
      1004527575,
      3885174651,
      3342903291,
      1200158424,
      1464636653,
    ]);
  });

  it("keeps retry count and positions identical for the same sample ID", () => {
    const config = {
      ...broadSampling,
      totalSamples: 257,
      masterSeed: 0xdecafbad,
      minTrapCount: 1,
      maxTrapCount: 2000,
      localWindow: {
        probability: 0.75,
        minWidthFraction: 0.4,
        minHeightFraction: 0.35,
      },
    };
    const first = generateSampleLayout(83, config);
    const retry = generateSampleLayout(83, config);
    expect(retry).toEqual(first);
  });

  it("keeps count fixed but deterministically changes positions on a retry attempt", () => {
    const config = {
      ...broadSampling,
      totalSamples: 257,
      masterSeed: 0xdecafbad,
    };
    const initial = generateSampleLayout(83, config, 0);
    const retry = generateSampleLayout(83, config, 1);
    expect(retry.trapCount).toBe(initial.trapCount);
    expect(retry.samplingSeed).not.toBe(initial.samplingSeed);
    expect(retry.traps).not.toEqual(initial.traps);
    expect(generateSampleLayout(83, config, 1)).toEqual(retry);
  });

  it("covers the logarithmic endpoints and permutes adjacent sample IDs", () => {
    const config = { totalSamples: 101, masterSeed: 0x1020_3040 };
    const counts = Array.from({ length: config.totalSamples }, (_, sampleId) => sampleTrapCount(sampleId, config));
    expect(Math.min(...counts)).toBe(1);
    expect(Math.max(...counts)).toBe(2000);
    expect(counts).not.toEqual([...counts].sort((a, b) => a - b));
    expect(counts.map((count) => Math.log(count + 0.5))).toHaveLength(101);
  });

  it("supports a linearly stratified count schedule when requested", () => {
    const config = {
      totalSamples: 5,
      masterSeed: 9,
      minTrapCount: 10,
      maxTrapCount: 50,
      distribution: "uniform" as const,
    };
    const counts = Array.from({ length: 5 }, (_, sampleId) => sampleTrapCount(sampleId, config));
    expect([...counts].sort((a, b) => a - b)).toEqual([10, 20, 30, 40, 50]);
  });

  it.each([1, 2000])("generates exactly %i traps with stable IDs", (trapCount) => {
    const result = sampleTrapPositions(trapCount, 12345 + trapCount, broadSampling);
    expect(result.traps).toHaveLength(trapCount);
    expect(result.traps[0]!.trapId).toBe(1);
    expect(result.traps.at(-1)!.trapId).toBe(trapCount);
    for (let index = 1; index < result.traps.length; index += 1) {
      const previous = result.traps[index - 1]!;
      const current = result.traps[index]!;
      expect(current.yUm > previous.yUm || (current.yUm === previous.yUm && current.xUm >= previous.xUm)).toBe(true);
    }
  });

  it("respects safe bounds, local-window bounds, zero-order guard, and anisotropic spacing", () => {
    const result = sampleTrapPositions(350, 777, {
      ...broadSampling,
      localWindow: {
        minWidthFraction: 0.45,
        maxWidthFraction: 0.45,
        minHeightFraction: 0.5,
        maxHeightFraction: 0.5,
      },
    });
    expect(result.bounds.minXUm).toBeGreaterThanOrEqual(broadSampling.bounds.minXUm);
    expect(result.bounds.maxXUm).toBeLessThanOrEqual(broadSampling.bounds.maxXUm);
    expect(result.bounds.minYUm).toBeGreaterThanOrEqual(broadSampling.bounds.minYUm);
    expect(result.bounds.maxYUm).toBeLessThanOrEqual(broadSampling.bounds.maxYUm);

    for (const trap of result.traps) {
      expect(trap.xUm).toBeGreaterThanOrEqual(result.bounds.minXUm);
      expect(trap.xUm).toBeLessThanOrEqual(result.bounds.maxXUm);
      expect(trap.yUm).toBeGreaterThanOrEqual(result.bounds.minYUm);
      expect(trap.yUm).toBeLessThanOrEqual(result.bounds.maxYUm);
      expect((trap.xUm / 12) ** 2 + (trap.yUm / 18) ** 2).toBeGreaterThan(1);
    }
    for (let first = 0; first < result.traps.length; first += 1) {
      for (let second = first + 1; second < result.traps.length; second += 1) {
        const dx = (result.traps[first]!.xUm - result.traps[second]!.xUm) / 2;
        const dy = (result.traps[first]!.yUm - result.traps[second]!.yUm) / 3;
        expect(dx * dx + dy * dy).toBeGreaterThanOrEqual(1 - 1e-12);
      }
    }
  });
});

describe("sample payload protocol", () => {
  it("computes the standard CRC-32 check vector", () => {
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf4_3926);
  });

  it("writes the exact little-endian v1 layout including metrics JSON", () => {
    const positions = new Float32Array([1.5, -2.25, 3.75, 4.5]);
    const phases = new Float32Array([-Math.PI, Math.PI / 2]);
    const trapIds = new Uint32Array([1, 2]);
    const frame = new Uint8Array([0x00, 0x7f, 0x80, 0xff, 0x01, 0x02]);
    const metrics = { converged: true, iterations: 4 };
    const payload = encodeSamplePayload({
      sampleId: 0x0123_4567_89ab_cdefn,
      samplingSeed: 0xfedc_ba98,
      width: 3,
      height: 2,
      positions,
      measuredPhases: phases,
      trapIds,
      frame,
      metrics,
    });
    const view = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    const magic = new TextDecoder().decode(payload.subarray(0, 4));
    expect(magic).toBe(SAMPLE_PAYLOAD_MAGIC);
    expect(view.getUint16(4, true)).toBe(SAMPLE_PAYLOAD_VERSION);
    expect(view.getUint16(6, true)).toBe(SAMPLE_PAYLOAD_HEADER_BYTES);
    expect(view.getBigUint64(8, true)).toBe(0x0123_4567_89ab_cdefn);
    expect(view.getUint32(16, true)).toBe(2);
    expect(view.getUint32(20, true)).toBe(0xfedc_ba98);
    expect(view.getUint32(24, true)).toBe(crc32(frame));
    expect(view.getUint32(28, true)).toBe(3);
    expect(view.getUint32(32, true)).toBe(2);
    expect(view.getUint32(36, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(8);
    expect(view.getUint32(44, true)).toBe(8);
    expect(view.getUint32(48, true)).toBe(6);
    const metricsByteLength = view.getUint32(52, true);
    expect(view.getUint32(56, true)).toBe(0);
    expect(view.getUint32(60, true)).toBe(0);

    let offset = SAMPLE_PAYLOAD_HEADER_BYTES;
    expect(Array.from({ length: 4 }, () => {
      const value = view.getFloat32(offset, true);
      offset += 4;
      return value;
    })).toEqual(Array.from(positions));
    expect(Array.from({ length: 2 }, () => {
      const value = view.getFloat32(offset, true);
      offset += 4;
      return value;
    })).toEqual(Array.from(phases));
    expect([view.getUint32(offset, true), view.getUint32(offset + 4, true)]).toEqual([1, 2]);
    offset += 8;
    expect(payload.slice(offset, offset + frame.length)).toEqual(frame);
    offset += frame.length;
    expect(new TextDecoder().decode(payload.subarray(offset, offset + metricsByteLength))).toBe(JSON.stringify(metrics));
    expect(payload.byteLength).toBe(offset + metricsByteLength);
  });

  it("rejects arrays that do not agree on trap count", () => {
    expect(() => encodeSamplePayload({
      sampleId: 0,
      samplingSeed: 0,
      width: 1,
      height: 1,
      positions: new Float32Array([0, 0]),
      measuredPhases: new Float32Array(2),
      trapIds: new Uint32Array([1, 2]),
      frame: new Uint8Array(1),
      metrics: {},
    })).toThrow("Positions length");
  });
});
