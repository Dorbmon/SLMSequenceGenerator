import type { FrameStore } from "./types.js";

export class MemoryFrameStore<T> implements FrameStore<T> {
  private readonly frames: T[] = [];

  get length(): number {
    return this.frames.length;
  }

  get count(): number {
    return this.frames.length;
  }

  append(frame: T): void {
    this.frames.push(frame);
  }

  get(index: number): T {
    if (!Number.isInteger(index) || index < 0 || index >= this.frames.length) {
      throw new RangeError(`Frame index ${index} is out of range`);
    }
    return this.frames[index]!;
  }

  toArray(): T[] {
    return [...this.frames];
  }

  clear(): void {
    this.frames.length = 0;
  }

  [Symbol.iterator](): Iterator<T> {
    return this.frames[Symbol.iterator]();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    for (const frame of this.frames) yield frame;
  }
}

export class MappedFrameStore<T> implements FrameStore<T> {
  private readonly values: T[];
  private readonly writer: (frame: T, index: number) => void | Promise<void>;

  constructor(writer: (frame: T, index: number) => void | Promise<void>, initial: T[] = []) {
    this.values = [...initial];
    this.writer = writer;
  }

  get length(): number {
    return this.values.length;
  }

  get count(): number {
    return this.values.length;
  }

  async append(frame: T): Promise<void> {
    await this.writer(frame, this.values.length);
    this.values.push(frame);
  }

  get(index: number): T {
    if (index < 0 || index >= this.values.length) throw new RangeError(`Frame index ${index} is out of range`);
    return this.values[index]!;
  }

  toArray(): T[] {
    return [...this.values];
  }

  clear(): void {
    this.values.length = 0;
  }

  [Symbol.iterator](): Iterator<T> {
    return this.values[Symbol.iterator]();
  }

  async *[Symbol.asyncIterator](): AsyncIterator<T> {
    for (const frame of this.values) yield frame;
  }
}
