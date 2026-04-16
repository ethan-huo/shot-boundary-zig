import { useEffect, useMemo, useRef, useState } from "react"
import { ActivityIcon, FilmIcon, PlayIcon, ScissorsIcon, UploadIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { EditorTimeline, type TimelineMarker } from "@/components/editor-timeline"
import { modelSpec } from "@/lib/shot-boundary/model-spec"
import {
  loadModel,
  segmentFrames,
  type Backend,
  type ModelSource,
  type SegmentResult,
} from "@/lib/shot-boundary/onnx-runtime"
import {
  decodeVideoToRgb24,
  generateTimelineThumbnails,
  type DecodedFrames,
  type TimelineThumbnail,
} from "@/lib/shot-boundary/video-decode"

type StatusKind = "idle" | "working" | "done" | "error"

type RunSummary = {
  loadModelMs: number
  decodeMs: number
  thumbnailMs: number
  codec: string | null
  averageFps: number | null
}

const defaultModelUrl = "/models/transnetv2.onnx"

export function App() {
  const previewVideoRef = useRef<HTMLVideoElement | null>(null)
  const [modelUrl, setModelUrl] = useState(defaultModelUrl)
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [backend, setBackend] = useState<Backend>("wasm")
  const [maxFrames, setMaxFrames] = useState(150)
  const [batchSize, setBatchSize] = useState(1)
  const [threshold, setThreshold] = useState<number>(modelSpec.transnetV2SceneThreshold)
  const [status, setStatus] = useState("Ready.")
  const [statusKind, setStatusKind] = useState<StatusKind>("idle")
  const [result, setResult] = useState<SegmentResult | null>(null)
  const [decoded, setDecoded] = useState<DecodedFrames | null>(null)
  const [summary, setSummary] = useState<RunSummary | null>(null)
  const [thumbnails, setThumbnails] = useState<TimelineThumbnail[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [isBusy, setIsBusy] = useState(false)

  useEffect(() => {
    return () => {
      if (videoUrl !== null) {
        URL.revokeObjectURL(videoUrl)
      }
    }
  }, [videoUrl])

  useEffect(() => {
    return () => {
      for (const thumbnail of thumbnails) {
        if (thumbnail.url.length > 0) {
          URL.revokeObjectURL(thumbnail.url)
        }
      }
    }
  }, [thumbnails])

  useEffect(() => {
    const video = previewVideoRef.current
    if (video === null) {
      return
    }

    const syncCurrentTime = () => setCurrentTime(video.currentTime)
    video.addEventListener("loadedmetadata", syncCurrentTime)
    video.addEventListener("timeupdate", syncCurrentTime)
    video.addEventListener("seeked", syncCurrentTime)

    return () => {
      video.removeEventListener("loadedmetadata", syncCurrentTime)
      video.removeEventListener("timeupdate", syncCurrentTime)
      video.removeEventListener("seeked", syncCurrentTime)
    }
  }, [videoUrl])

  const markers = useMemo(() => {
    if (result === null || decoded === null) {
      return []
    }

    return makeTimelineMarkers(result, decoded)
  }, [result, decoded])

  async function runModelSmoke() {
    setIsBusy(true)
    setStatusKind("working")
    setStatus("Loading model...")

    try {
      const loadedModel = await loadModel(readModelSource(modelFile, modelUrl), backend)
      setStatus("Running deterministic smoke window...")
      const frameCount = modelSpec.windowFrames
      const framesRgb24 = makeDeterministicFrames(frameCount)
      const smokeResult = await segmentFrames(loadedModel, framesRgb24, frameCount, {
        batchSize: 1,
        threshold,
      })

      setDecoded(null)
      setResult(smokeResult)
      replaceThumbnails([])
      setSummary({
        loadModelMs: loadedModel.loadMs,
        decodeMs: 0,
        thumbnailMs: 0,
        codec: null,
        averageFps: null,
      })
      setStatusKind("done")
      setStatus("Smoke run complete.")
    } catch (error) {
      reportError(error)
    } finally {
      setIsBusy(false)
    }
  }

  async function segmentVideo() {
    if (videoFile === null) {
      setStatusKind("error")
      setStatus("Select a video file first.")
      return
    }

    setIsBusy(true)
    setStatusKind("working")
    setStatus("Loading model...")

    try {
      const loadedModel = await loadModel(readModelSource(modelFile, modelUrl), backend)

      setStatus("Decoding video frames with Mediabunny...")
      const decodeStartedAt = performance.now()
      const nextDecoded = await decodeVideoToRgb24({
        file: videoFile,
        maxFrames,
        onProgress: updateDecodeProgress,
      })
      const decodeMs = performance.now() - decodeStartedAt

      setStatus("Running inference...")
      const nextResult = await segmentFrames(loadedModel, nextDecoded.framesRgb24, nextDecoded.frameCount, {
        batchSize,
        threshold,
      })

      setStatus("Building thumbnail timeline...")
      const thumbnailStartedAt = performance.now()
      const nextThumbnails = await generateTimelineThumbnails(
        videoFile,
        chooseThumbnailCount(nextDecoded.frameCount),
        nextDecoded.analyzedDurationSeconds,
      )
      const thumbnailMs = performance.now() - thumbnailStartedAt

      setDecoded(nextDecoded)
      setResult(nextResult)
      replaceThumbnails(nextThumbnails)
      setSummary({
        loadModelMs: loadedModel.loadMs,
        decodeMs,
        thumbnailMs,
        codec: nextDecoded.codec,
        averageFps: nextDecoded.averageFps,
      })
      setStatusKind("done")
      setStatus("Segmentation complete.")
    } catch (error) {
      reportError(error)
    } finally {
      setIsBusy(false)
    }
  }

  function replaceThumbnails(nextThumbnails: TimelineThumbnail[]) {
    setThumbnails((currentThumbnails) => {
      for (const thumbnail of currentThumbnails) {
        if (thumbnail.url.length > 0) {
          URL.revokeObjectURL(thumbnail.url)
        }
      }

      return nextThumbnails
    })
  }

  function reportError(error: unknown) {
    const message = error instanceof Error ? error.message : String(error)
    setStatusKind("error")
    setStatus(message)
    console.error(error)
  }

  function updateDecodeProgress(progress: { current: number; total: number }) {
    setStatus(`Decoded ${progress.current} / ${progress.total} frames...`)
  }

  function handleModelFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    setModelFile(event.currentTarget.files?.item(0) ?? null)
  }

  function handleVideoFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const nextFile = event.currentTarget.files?.item(0) ?? null
    setVideoFile(nextFile)
    setResult(null)
    setDecoded(null)
    setSummary(null)
    setCurrentTime(0)
    replaceThumbnails([])

    setVideoUrl((currentUrl) => {
      if (currentUrl !== null) {
        URL.revokeObjectURL(currentUrl)
      }

      return nextFile === null ? null : URL.createObjectURL(nextFile)
    })
  }

  function seekPreview(timeSeconds: number) {
    if (previewVideoRef.current === null) {
      return
    }

    const nextTime = Math.max(0, timeSeconds)
    previewVideoRef.current.currentTime = nextTime
    setCurrentTime(nextTime)
  }

  return (
    <main className="min-h-svh bg-[#f6f7f5] text-[#202324] dark:bg-background dark:text-foreground">
      <section className="mx-auto grid w-full max-w-[1440px] gap-8 px-5 py-8 md:px-8 lg:px-10">
        <header className="grid gap-4">
          <p className="text-sm font-semibold tracking-[0.18em] text-[#b03a2e] uppercase">
            TransNetV2 Web/WASM probe
          </p>
          <div className="grid gap-3 lg:grid-cols-[minmax(0,0.95fr)_minmax(24rem,0.75fr)] lg:items-end">
            <h1 className="max-w-4xl text-5xl leading-[0.94] font-semibold tracking-normal text-balance md:text-7xl">
              Segment shots on a thumbnail timeline.
            </h1>
            <p className="max-w-2xl text-base leading-7 text-[#5d6460] md:text-lg dark:text-muted-foreground">
              Load the ONNX model, decode frames with Mediabunny, then place cut markers over a timeline that looks like
              an editor track.
            </p>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[minmax(0,0.92fr)_minmax(24rem,0.58fr)]">
          <div className="rounded-lg border border-[#dde2df] bg-white p-4 shadow-sm dark:border-border dark:bg-card">
            <div className="grid gap-4">
              <label className="grid gap-2 text-sm font-medium">
                Model URL
                <input
                  value={modelUrl}
                  disabled={isBusy}
                  onChange={(event) => setModelUrl(event.currentTarget.value)}
                  className="h-10 rounded-md border border-[#cdd4d1] bg-white px-3 text-sm outline-none focus:ring-3 focus:ring-[#116b5f]/20 disabled:opacity-60 dark:border-input dark:bg-background"
                />
              </label>

              <div className="grid gap-3 md:grid-cols-2">
                <label className="grid gap-2 text-sm font-medium">
                  Local ONNX model
                  <span className="flex h-10 items-center gap-2 rounded-md border border-dashed border-[#cdd4d1] px-3 text-sm text-[#5d6460] dark:border-input dark:text-muted-foreground">
                    <UploadIcon className="size-4" />
                    <input
                      type="file"
                      accept=".onnx"
                      disabled={isBusy}
                      onChange={handleModelFileChange}
                      className="min-w-0 flex-1 text-xs file:mr-3 file:rounded-md file:border-0 file:bg-[#edf3f1] file:px-2 file:py-1 file:text-[#116b5f]"
                    />
                  </span>
                </label>

                <label className="grid gap-2 text-sm font-medium">
                  Video file
                  <span className="flex h-10 items-center gap-2 rounded-md border border-dashed border-[#cdd4d1] px-3 text-sm text-[#5d6460] dark:border-input dark:text-muted-foreground">
                    <FilmIcon className="size-4" />
                    <input
                      type="file"
                      accept="video/*"
                      disabled={isBusy}
                      onChange={handleVideoFileChange}
                      className="min-w-0 flex-1 text-xs file:mr-3 file:rounded-md file:border-0 file:bg-[#edf3f1] file:px-2 file:py-1 file:text-[#116b5f]"
                    />
                  </span>
                </label>
              </div>

              <div className="grid gap-3 md:grid-cols-4">
                <label className="grid gap-2 text-sm font-medium">
                  Backend
                  <select
                    value={backend}
                    disabled={isBusy}
                    onChange={(event) => setBackend(readBackend(event.currentTarget.value))}
                    className="h-10 rounded-md border border-[#cdd4d1] bg-white px-3 text-sm dark:border-input dark:bg-background"
                  >
                    <option value="wasm">WASM</option>
                    <option value="webgpu">WebGPU</option>
                  </select>
                </label>

                <NumberField label="Max frames" value={maxFrames} min={1} disabled={isBusy} onChange={setMaxFrames} />
                <NumberField label="Batch size" value={batchSize} min={1} disabled={isBusy} onChange={setBatchSize} />
                <NumberField
                  label="Threshold"
                  value={threshold}
                  min={0}
                  max={1}
                  step={0.001}
                  disabled={isBusy}
                  onChange={setThreshold}
                />
              </div>

              <div className="flex flex-wrap gap-3">
                <Button type="button" disabled={isBusy} onClick={() => void runModelSmoke()}>
                  <PlayIcon className="size-4" />
                  Run smoke
                </Button>
                <Button type="button" disabled={isBusy} variant="secondary" onClick={() => void segmentVideo()}>
                  <ScissorsIcon className="size-4" />
                  Segment video
                </Button>
              </div>
            </div>
          </div>

          <aside className="rounded-lg border border-[#dde2df] bg-white p-4 shadow-sm dark:border-border dark:bg-card">
            <div className="grid gap-3">
              <video
                ref={previewVideoRef}
                id="preview-video"
                src={videoUrl ?? undefined}
                controls
                playsInline
                className="aspect-video w-full rounded-md bg-[#111]"
              />
              <div
                className={statusClassName(statusKind)}
                role={statusKind === "error" ? "alert" : "status"}
              >
                <ActivityIcon className="size-4 shrink-0" />
                <span>{status}</span>
              </div>
            </div>
          </aside>
        </section>

        <EditorTimeline
          thumbnails={thumbnails}
          markers={markers}
          frameCount={result?.frameCount ?? 0}
          durationSeconds={decoded?.analyzedDurationSeconds ?? 0}
          sourceDurationSeconds={decoded?.durationSeconds ?? null}
          currentTimeSeconds={currentTime}
          onSeek={seekPreview}
        />

        <section className="grid gap-4 lg:grid-cols-[minmax(0,0.8fr)_minmax(24rem,0.7fr)]">
          <MetricsPanel result={result} summary={summary} />
          <pre className="max-h-[32rem] overflow-auto rounded-lg border border-[#dde2df] bg-white p-4 text-xs leading-5 text-[#202324] shadow-sm dark:border-border dark:bg-card dark:text-card-foreground">
            {JSON.stringify(makeJsonOutput(result, summary, decoded), null, 2)}
          </pre>
        </section>
      </section>
    </main>
  )
}

type NumberFieldProps = {
  label: string
  value: number
  min: number
  max?: number
  step?: number
  disabled: boolean
  onChange: (value: number) => void
}

function NumberField(props: NumberFieldProps) {
  return (
    <label className="grid gap-2 text-sm font-medium">
      {props.label}
      <input
        type="number"
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        disabled={props.disabled}
        onChange={(event) => props.onChange(Number(event.currentTarget.value))}
        className="h-10 rounded-md border border-[#cdd4d1] bg-white px-3 text-sm outline-none focus:ring-3 focus:ring-[#116b5f]/20 disabled:opacity-60 dark:border-input dark:bg-background"
      />
    </label>
  )
}

function MetricsPanel(props: { result: SegmentResult | null; summary: RunSummary | null }) {
  const metrics = [
    ["Frames", props.result?.frameCount.toString() ?? "n/a"],
    ["Scenes", props.result?.scenes.length.toString() ?? "n/a"],
    ["Codec", props.summary?.codec ?? "n/a"],
    [
      "Average FPS",
      props.summary?.averageFps === null || props.summary === null ? "n/a" : props.summary.averageFps.toFixed(2),
    ],
    ["Load model", formatMs(props.summary?.loadModelMs ?? 0)],
    ["Decode", formatMs(props.summary?.decodeMs ?? 0)],
    ["Thumbnails", formatMs(props.summary?.thumbnailMs ?? 0)],
    ["Inference", formatMs(props.result?.timings.inferenceMs ?? 0)],
  ]

  return (
    <section className="grid grid-cols-2 gap-3 rounded-lg border border-[#dde2df] bg-white p-4 shadow-sm md:grid-cols-4 dark:border-border dark:bg-card">
      {metrics.map((metric) => (
        <div key={metric[0]} className="border-l-4 border-[#d7a927] bg-[#f7f9f8] p-3 dark:bg-muted/30">
          <div className="text-xs text-[#5d6460] dark:text-muted-foreground">{metric[0]}</div>
          <div className="mt-1 text-sm font-semibold">{metric[1]}</div>
        </div>
      ))}
    </section>
  )
}

function readModelSource(modelFile: File | null, modelUrl: string): ModelSource {
  if (modelFile !== null) {
    return { kind: "file", value: modelFile }
  }

  const value = modelUrl.trim()
  if (value.length === 0) {
    throw new Error("Model URL cannot be empty.")
  }

  return { kind: "url", value }
}

function readBackend(value: string): Backend {
  if (value !== "wasm" && value !== "webgpu") {
    throw new Error("Backend must be WASM or WebGPU.")
  }

  return value
}

function makeDeterministicFrames(frameCount: number): Uint8Array {
  const length = frameCount * modelSpec.inputWidth * modelSpec.inputHeight * modelSpec.inputChannels
  const frames = new Uint8Array(length)
  for (let index = 0; index < frames.length; index += 1) {
    frames[index] = index & 255
  }
  return frames
}

function makeTimelineMarkers(result: SegmentResult, decoded: DecodedFrames): TimelineMarker[] {
  const durationSeconds = Math.max(decoded.analyzedDurationSeconds, 1 / Math.max(decoded.averageFps, 1))
  const fps = decoded.averageFps && decoded.averageFps > 0 ? decoded.averageFps : result.frameCount / durationSeconds

  return result.scenes.slice(0, -1).map((scene, index) => {
    const frame = Math.min(result.frameCount - 1, scene.end)
    const timeSeconds = Math.min(durationSeconds, frame / fps)
    return {
      frame,
      label: `Cut ${index + 1}`,
      percent: (timeSeconds / durationSeconds) * 100,
      timeSeconds,
    }
  })
}

function chooseThumbnailCount(frameCount: number): number {
  return Math.max(6, Math.min(18, Math.ceil(frameCount / 25)))
}

function makeJsonOutput(result: SegmentResult | null, summary: RunSummary | null, decoded: DecodedFrames | null) {
  if (result === null) {
    return {}
  }

  return {
    frameCount: result.frameCount,
    analyzedDurationSeconds: decoded?.analyzedDurationSeconds ?? null,
    sourceDurationSeconds: decoded?.durationSeconds ?? null,
    scenes: result.scenes,
    timings: {
      loadModelMs: summary?.loadModelMs ?? 0,
      decodeMs: summary?.decodeMs ?? 0,
      thumbnailMs: summary?.thumbnailMs ?? 0,
      windowingMs: result.timings.windowingMs,
      inferenceMs: result.timings.inferenceMs,
      postprocessMs: result.timings.postprocessMs,
      totalSegmentMs: result.timings.totalMs,
    },
    predictions: {
      singleFrame: Array.from(result.singleFrame),
      manyHot: Array.from(result.manyHot),
    },
  }
}

function formatMs(value: number): string {
  return `${value.toFixed(1)} ms`
}

function statusClassName(kind: StatusKind): string {
  const base = "flex min-h-11 items-start gap-2 rounded-md px-3 py-2 text-sm leading-6"

  if (kind === "error") {
    return `${base} bg-[#fde8e6] text-[#9d271c] dark:bg-destructive/15 dark:text-destructive`
  }
  if (kind === "working") {
    return `${base} bg-[#edf3f1] text-[#116b5f] dark:bg-primary/10 dark:text-primary`
  }
  if (kind === "done") {
    return `${base} bg-[#eef7ea] text-[#276c22] dark:bg-emerald-500/10 dark:text-emerald-400`
  }

  return `${base} bg-[#f3f5f4] text-[#5d6460] dark:bg-muted dark:text-muted-foreground`
}
