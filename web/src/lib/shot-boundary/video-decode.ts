import { ALL_FORMATS, BlobSource, CanvasSink, Input } from "mediabunny";
import { frameBytes, modelSpec } from "./model-spec";

export type DecodeProgress = {
  current: number;
  total: number;
};

export type DecodeOptions = {
  file: File;
  maxFrames: number;
  onProgress?: (progress: DecodeProgress) => void;
};

export type TimelineThumbnail = {
  url: string;
  timestampSeconds: number;
};

export type DecodedFrames = {
  framesRgb24: Uint8Array;
  frameCount: number;
  analyzedDurationSeconds: number;
  durationSeconds: number;
  averageFps: number;
  codec: string | null;
};

export async function decodeVideoToRgb24(options: DecodeOptions): Promise<DecodedFrames> {
  if (!Number.isInteger(options.maxFrames) || options.maxFrames <= 0) {
    throw new Error("Max frames must be a positive integer.");
  }

  const input = new Input({
    source: new BlobSource(options.file),
    formats: ALL_FORMATS,
  });

  try {
    const videoTrack = await input.getPrimaryVideoTrack();
    if (videoTrack === null) {
      throw new Error("The media file has no video track.");
    }
    if (!(await videoTrack.canDecode())) {
      throw new Error("The browser cannot decode this video codec.");
    }

    const durationSeconds = await videoTrack.computeDuration();
    const stats = await videoTrack.computePacketStats(Math.min(options.maxFrames, 100));
    const sink = new CanvasSink(videoTrack, {
      width: modelSpec.inputWidth,
      height: modelSpec.inputHeight,
      fit: "fill",
      poolSize: 4,
    });

    const framesRgb24 = new Uint8Array(options.maxFrames * frameBytes());
    let frameCount = 0;
    let cursor = 0;
    let firstFrameTimestamp: number | null = null;
    let lastFrameTimestamp = 0;

    for await (const wrappedCanvas of sink.canvases()) {
      if (wrappedCanvas.timestamp < 0) {
        continue;
      }

      firstFrameTimestamp ??= wrappedCanvas.timestamp;
      lastFrameTimestamp = wrappedCanvas.timestamp;
      const context = getCanvasContext(wrappedCanvas.canvas);
      const rgba = context.getImageData(0, 0, modelSpec.inputWidth, modelSpec.inputHeight).data;
      for (let pixel = 0; pixel < rgba.length; pixel += 4) {
        framesRgb24[cursor] = rgba[pixel] ?? 0;
        framesRgb24[cursor + 1] = rgba[pixel + 1] ?? 0;
        framesRgb24[cursor + 2] = rgba[pixel + 2] ?? 0;
        cursor += 3;
      }

      frameCount += 1;
      options.onProgress?.({ current: frameCount, total: options.maxFrames });
      if (frameCount >= options.maxFrames) {
        break;
      }
    }

    if (frameCount === 0) {
      throw new Error("No decodable video frames were found.");
    }

    const measuredDurationSeconds =
      firstFrameTimestamp === null || frameCount <= 1 ? 0 : Math.max(0, lastFrameTimestamp - firstFrameTimestamp);
    const averageFps =
      stats.averagePacketRate > 0
        ? stats.averagePacketRate
        : measuredDurationSeconds > 0
          ? (frameCount - 1) / measuredDurationSeconds
          : 30;
    const analyzedDurationSeconds =
      measuredDurationSeconds > 0 ? measuredDurationSeconds + 1 / averageFps : frameCount / averageFps;

    return {
      framesRgb24: framesRgb24.slice(0, frameCount * frameBytes()),
      frameCount,
      analyzedDurationSeconds,
      durationSeconds,
      averageFps,
      codec: videoTrack.codec,
    };
  } finally {
    input.dispose();
  }
}

export async function generateTimelineThumbnails(
  file: File,
  count: number,
  durationLimitSeconds?: number,
): Promise<TimelineThumbnail[]> {
  if (!Number.isInteger(count) || count <= 0) {
    throw new Error("Thumbnail count must be a positive integer.");
  }
  if (
    durationLimitSeconds !== undefined &&
    (!Number.isFinite(durationLimitSeconds) || durationLimitSeconds <= 0)
  ) {
    throw new Error("Thumbnail duration limit must be a positive number.");
  }

  const input = new Input({
    source: new BlobSource(file),
    formats: ALL_FORMATS,
  });

  try {
    const videoTrack = await input.getPrimaryVideoTrack();
    if (videoTrack === null) {
      throw new Error("The media file has no video track.");
    }
    if (!(await videoTrack.canDecode())) {
      throw new Error("The browser cannot decode this video codec.");
    }

    const startTimestamp = Math.max(0, await videoTrack.getFirstTimestamp());
    const durationSeconds = await videoTrack.computeDuration();
    const sourceDuration = Math.max(0, durationSeconds - startTimestamp);
    const usableDuration = Math.min(sourceDuration, durationLimitSeconds ?? sourceDuration);
    const sink = new CanvasSink(videoTrack, {
      width: 160,
      height: 90,
      fit: "cover",
      poolSize: 0,
    });
    const timestamps = evenlySpacedTimestamps(startTimestamp, usableDuration, count);
    const thumbnails: TimelineThumbnail[] = [];

    for await (const wrappedCanvas of sink.canvasesAtTimestamps(timestamps)) {
      const timestampSeconds = timestamps[thumbnails.length] ?? startTimestamp;
      if (wrappedCanvas === null) {
        thumbnails.push({ url: "", timestampSeconds });
        continue;
      }

      thumbnails.push({
        url: await canvasToObjectUrl(wrappedCanvas.canvas),
        timestampSeconds: wrappedCanvas.timestamp,
      });
    }

    return thumbnails;
  } finally {
    input.dispose();
  }
}

function getCanvasContext(canvas: HTMLCanvasElement | OffscreenCanvas): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (context === null) {
    throw new Error("Canvas 2D context is not available.");
  }
  return context;
}

function evenlySpacedTimestamps(startTimestamp: number, durationSeconds: number, count: number): number[] {
  if (count === 1 || durationSeconds === 0) {
    return [startTimestamp];
  }

  const lastOffset = Math.max(0, durationSeconds - 0.001);
  return Array.from({ length: count }, (_, index) => startTimestamp + (lastOffset * index) / (count - 1));
}

async function canvasToObjectUrl(canvas: HTMLCanvasElement | OffscreenCanvas): Promise<string> {
  const blob =
    canvas instanceof HTMLCanvasElement
      ? await htmlCanvasToBlob(canvas)
      : await canvas.convertToBlob({ type: "image/jpeg", quality: 0.78 });

  return URL.createObjectURL(blob);
}

function htmlCanvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => {
        if (blob === null) {
          reject(new Error("Could not encode timeline thumbnail."));
          return;
        }
        resolve(blob);
      },
      "image/jpeg",
      0.78,
    );
  });
}
