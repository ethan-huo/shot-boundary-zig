import { frameBytes, modelSpec } from "./model-spec";

export type Scene = {
  start: number;
  end: number;
};

export function windowSourceIndices(frameCount: number): number[][] {
  if (!Number.isInteger(frameCount) || frameCount <= 0) {
    throw new Error("Frame count must be a positive integer.");
  }

  const paddedStart = modelSpec.contextFrames;
  const remainder = frameCount % modelSpec.outputFramesPerWindow;
  const paddedEnd =
    modelSpec.contextFrames +
    modelSpec.outputFramesPerWindow -
    (remainder === 0 ? modelSpec.outputFramesPerWindow : remainder);
  const paddedCount = paddedStart + frameCount + paddedEnd;
  const windows: number[][] = [];

  for (let ptr = 0; ptr + modelSpec.windowFrames <= paddedCount; ptr += modelSpec.outputFramesPerWindow) {
    const indices: number[] = [];
    for (let paddedIndex = ptr; paddedIndex < ptr + modelSpec.windowFrames; paddedIndex += 1) {
      if (paddedIndex < paddedStart) {
        indices.push(0);
      } else if (paddedIndex < paddedStart + frameCount) {
        indices.push(paddedIndex - paddedStart);
      } else {
        indices.push(frameCount - 1);
      }
    }
    windows.push(indices);
  }

  return windows;
}

export function buildWindowBatch(framesRgb24: Uint8Array, frameCount: number, windows: number[][]): Uint8Array {
  const bytesPerFrame = frameBytes();
  if (framesRgb24.byteLength !== frameCount * bytesPerFrame) {
    throw new Error("RGB frame buffer length does not match the frame count.");
  }
  if (windows.length === 0) {
    throw new Error("At least one window is required.");
  }

  const output = new Uint8Array(windows.length * modelSpec.windowFrames * bytesPerFrame);
  let cursor = 0;

  for (const indices of windows) {
    if (indices.length !== modelSpec.windowFrames) {
      throw new Error("Window length does not match the model ABI.");
    }

    for (const index of indices) {
      if (!Number.isInteger(index) || index < 0 || index >= frameCount) {
        throw new Error("Window index is outside the frame buffer.");
      }
      const start = index * bytesPerFrame;
      const end = start + bytesPerFrame;
      output.set(framesRgb24.subarray(start, end), cursor);
      cursor += bytesPerFrame;
    }
  }

  return output;
}

export function predictionsToScenes(predictions: Float32Array | number[], threshold: number): Scene[] {
  if (predictions.length === 0) {
    throw new Error("Predictions cannot be empty.");
  }
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error("Threshold must be in [0, 1].");
  }

  const scenes: Scene[] = [];
  let previousIsTransition = false;
  let currentStart = 0;

  for (let index = 0; index < predictions.length; index += 1) {
    const prediction = predictions[index];
    if (prediction === undefined || !Number.isFinite(prediction)) {
      throw new Error("Predictions must be finite numbers.");
    }

    const isTransition = prediction > threshold;
    if (previousIsTransition && !isTransition) {
      currentStart = index;
    }
    if (!previousIsTransition && isTransition && index !== 0) {
      scenes.push({ start: currentStart, end: index });
    }
    previousIsTransition = isTransition;
  }

  if (!previousIsTransition) {
    scenes.push({ start: currentStart, end: predictions.length - 1 });
  }
  if (scenes.length === 0) {
    scenes.push({ start: 0, end: predictions.length - 1 });
  }

  return scenes;
}
