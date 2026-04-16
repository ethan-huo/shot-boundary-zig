import * as ort from "onnxruntime-web";
import { modelSpec } from "./model-spec";
import { buildWindowBatch, predictionsToScenes, windowSourceIndices, type Scene } from "./segment-core";

export type Backend = "wasm" | "webgpu";

export type ModelSource =
  | { kind: "url"; value: string }
  | { kind: "file"; value: File };

export type LoadedModel = {
  session: ort.InferenceSession;
  backend: Backend;
  loadMs: number;
};

export type SegmentTiming = {
  windowingMs: number;
  inferenceMs: number;
  postprocessMs: number;
  totalMs: number;
};

export type SegmentResult = {
  frameCount: number;
  singleFrame: Float32Array;
  manyHot: Float32Array;
  scenes: Scene[];
  timings: SegmentTiming;
};

export type SegmentOptions = {
  batchSize: number;
  threshold: number;
};

// Vite blocks importing JavaScript modules from public/. Override only the
// binary URL so ONNX Runtime keeps using its bundled JS factory.
ort.env.wasm.wasmPaths = { wasm: "/ort-wasm/ort-wasm-simd-threaded.jsep.wasm" };
ort.env.wasm.numThreads = 1;

export async function loadModel(source: ModelSource, backend: Backend): Promise<LoadedModel> {
  const startedAt = performance.now();
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: [backend],
    freeDimensionOverrides: { batch: 1 },
  };
  const session =
    source.kind === "file"
      ? await ort.InferenceSession.create(new Uint8Array(await source.value.arrayBuffer()), options)
      : await ort.InferenceSession.create(source.value, options);

  return {
    session,
    backend,
    loadMs: performance.now() - startedAt,
  };
}

export async function segmentFrames(
  loadedModel: LoadedModel,
  framesRgb24: Uint8Array,
  frameCount: number,
  options: SegmentOptions,
): Promise<SegmentResult> {
  const startedAt = performance.now();
  if (!Number.isInteger(options.batchSize) || options.batchSize <= 0) {
    throw new Error("Batch size must be a positive integer.");
  }

  const windows = windowSourceIndices(frameCount);
  const expectedOutputLength = windows.length * modelSpec.outputFramesPerWindow;
  const singleFrame = new Float32Array(expectedOutputLength);
  const manyHot = new Float32Array(expectedOutputLength);

  let windowingMs = 0;
  let inferenceMs = 0;
  let outputCursor = 0;

  for (let windowIndex = 0; windowIndex < windows.length; windowIndex += options.batchSize) {
    const batch = windows.slice(windowIndex, windowIndex + options.batchSize);
    const windowStartedAt = performance.now();
    const windowData = buildWindowBatch(framesRgb24, frameCount, batch);
    windowingMs += performance.now() - windowStartedAt;

    const input = new ort.Tensor("uint8", windowData, [
      batch.length,
      modelSpec.windowFrames,
      modelSpec.inputHeight,
      modelSpec.inputWidth,
      modelSpec.inputChannels,
    ]);

    const inferenceStartedAt = performance.now();
    const outputs = await loadedModel.session.run({ frames: input });
    inferenceMs += performance.now() - inferenceStartedAt;

    const expectedBatchLength = batch.length * modelSpec.outputFramesPerWindow;
    singleFrame.set(readFloatOutput(outputs, "single_frame", expectedBatchLength), outputCursor);
    manyHot.set(readFloatOutput(outputs, "many_hot", expectedBatchLength), outputCursor);
    outputCursor += expectedBatchLength;
  }

  const postprocessStartedAt = performance.now();
  const trimmedSingleFrame = singleFrame.slice(0, frameCount);
  const trimmedManyHot = manyHot.slice(0, frameCount);
  const scenes = predictionsToScenes(trimmedSingleFrame, options.threshold);
  const postprocessMs = performance.now() - postprocessStartedAt;

  return {
    frameCount,
    singleFrame: trimmedSingleFrame,
    manyHot: trimmedManyHot,
    scenes,
    timings: {
      windowingMs,
      inferenceMs,
      postprocessMs,
      totalMs: performance.now() - startedAt,
    },
  };
}

function readFloatOutput(outputs: ort.InferenceSession.ReturnType, name: string, expectedLength: number): Float32Array {
  const output = outputs[name];
  if (output === undefined) {
    throw new Error(`Model output '${name}' is missing.`);
  }
  if (!(output.data instanceof Float32Array)) {
    throw new Error(`Model output '${name}' must be float32.`);
  }
  if (output.data.length !== expectedLength) {
    throw new Error(`Model output '${name}' length does not match the requested batch.`);
  }

  return output.data;
}
