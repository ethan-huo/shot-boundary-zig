import { useMemo, useRef, useState, type PointerEvent } from "react"

import type { TimelineThumbnail } from "@/lib/shot-boundary/video-decode"

export type TimelineMarker = {
  frame: number
  label: string
  percent: number
  timeSeconds: number
}

export type EditorTimelineProps = {
  thumbnails: TimelineThumbnail[]
  markers: TimelineMarker[]
  frameCount: number
  durationSeconds: number
  sourceDurationSeconds: number | null
  currentTimeSeconds: number
  onSeek: (timeSeconds: number) => void
}

type TimeTick = {
  timeSeconds: number
  percent: number
  label: string
}

export function EditorTimeline(props: EditorTimelineProps) {
  const contentRef = useRef<HTMLDivElement | null>(null)
  const [activePointerId, setActivePointerId] = useState<number | null>(null)

  const hasTimeline = props.thumbnails.length > 0 && props.durationSeconds > 0
  const durationSeconds = Math.max(0, props.durationSeconds)
  const clampedCurrentTime = clamp(props.currentTimeSeconds, 0, durationSeconds)
  const playheadPercent = durationSeconds === 0 ? 0 : (clampedCurrentTime / durationSeconds) * 100
  const ticks = useMemo(() => makeTimeTicks(durationSeconds), [durationSeconds])
  const contentWidth = Math.max(720, Math.ceil(durationSeconds * choosePixelsPerSecond(durationSeconds)))

  function seekFromClientX(clientX: number) {
    if (!hasTimeline || contentRef.current === null) {
      return
    }

    const rect = contentRef.current.getBoundingClientRect()
    const x = clamp(clientX - rect.left, 0, rect.width)
    props.onSeek((x / rect.width) * durationSeconds)
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>) {
    if (!hasTimeline) {
      return
    }

    event.currentTarget.setPointerCapture(event.pointerId)
    setActivePointerId(event.pointerId)
    seekFromClientX(event.clientX)
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>) {
    if (activePointerId !== event.pointerId) {
      return
    }

    seekFromClientX(event.clientX)
  }

  function handlePointerUp(event: PointerEvent<HTMLDivElement>) {
    if (activePointerId !== event.pointerId) {
      return
    }

    setActivePointerId(null)
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
  }

  return (
    <section className="rounded-lg border border-[#dde2df] bg-white p-4 shadow-sm dark:border-border dark:bg-card">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Timeline</h2>
          <p className="text-sm text-[#5d6460] dark:text-muted-foreground">
            {hasTimeline
              ? `${props.markers.length} markers across ${props.frameCount} decoded frames. Analyzed ${formatTimecode(
                  durationSeconds,
                )}${formatSourceDuration(props.sourceDurationSeconds)}.`
              : "Run video segmentation to render the editable timeline."}
          </p>
        </div>
        {hasTimeline ? (
          <div className="text-sm font-medium text-[#202324] tabular-nums dark:text-card-foreground">
            {formatTimecode(clampedCurrentTime)} / {formatTimecode(durationSeconds)}
          </div>
        ) : null}
      </div>

      <div className="grid overflow-hidden rounded-md border border-[#cfd7d3] bg-[#171b1a] text-white shadow-inner dark:border-border">
        <div className="grid grid-cols-[6.5rem_minmax(0,1fr)]">
          <div className="border-r border-white/10 bg-[#202624]">
            <div className="flex h-9 items-center px-3 text-xs font-semibold text-white/70">Time</div>
            <div className="flex h-28 items-center px-3 text-xs font-semibold text-white/75">V1</div>
          </div>

          <div className="overflow-x-auto">
            <div
              ref={contentRef}
              className="relative min-w-full select-none"
              style={{ width: `${contentWidth}px` }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
            >
              <div className="relative h-9 border-b border-white/10 bg-[#202624]">
                {ticks.map((tick) => (
                  <div
                    key={tick.timeSeconds}
                    className="absolute top-0 bottom-0 border-l border-white/20"
                    style={{ left: `${tick.percent}%` }}
                  >
                    <span className="ml-1 text-[11px] leading-8 text-white/65 tabular-nums">{tick.label}</span>
                  </div>
                ))}
              </div>

              <div className="relative h-28 cursor-col-resize bg-[#101312]">
                {hasTimeline ? (
                  <div
                    className="absolute inset-0 grid"
                    style={{ gridTemplateColumns: `repeat(${props.thumbnails.length}, minmax(104px, 1fr))` }}
                  >
                    {props.thumbnails.map((thumbnail, index) => (
                      <div
                        key={`${thumbnail.timestampSeconds}-${thumbnail.url}-${index}`}
                        className="overflow-hidden border-r border-black/35 bg-[#252b29]"
                      >
                        {thumbnail.url.length > 0 ? (
                          <img src={thumbnail.url} alt="" className="h-full w-full object-cover" draggable={false} />
                        ) : (
                          <span className="block h-full w-full bg-[repeating-linear-gradient(135deg,#252b29_0,#252b29_10px,#303735_10px,#303735_20px)]" />
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex h-full items-center px-4 text-sm text-white/60">
                    Timeline thumbnails will appear here.
                  </div>
                )}

                {props.markers.map((marker) => (
                  <button
                    key={`${marker.frame}-${marker.percent}`}
                    type="button"
                    title={`${marker.label} at ${formatTimecode(marker.timeSeconds)}`}
                    onPointerDown={(event) => event.stopPropagation()}
                    onClick={() => props.onSeek(marker.timeSeconds)}
                    className="absolute top-0 bottom-0 z-20 w-1 -translate-x-1/2 rounded-none border-0 bg-[#f24b39] p-0 shadow-[0_0_0_1px_rgba(255,255,255,0.75)]"
                    style={{ left: `${clamp(marker.percent, 0, 100)}%` }}
                  >
                    <span className="absolute top-1 left-1/2 h-3 w-3 -translate-x-1/2 rotate-45 rounded-[2px] bg-[#f24b39]" />
                    <span className="sr-only">{marker.label}</span>
                  </button>
                ))}

                {hasTimeline ? (
                  <div
                    className="pointer-events-none absolute top-0 bottom-0 z-30 w-0.5 -translate-x-1/2 bg-[#f5d547]"
                    style={{ left: `${playheadPercent}%` }}
                  >
                    <span className="absolute -top-1 left-1/2 h-4 w-4 -translate-x-1/2 rotate-45 rounded-[2px] bg-[#f5d547] shadow-sm" />
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>

      {hasTimeline ? (
        <p className="mt-3 text-xs text-[#5d6460] dark:text-muted-foreground">
          Drag the track to scrub the preview. Red markers are predicted shot boundaries inside the analyzed range.
        </p>
      ) : null}
    </section>
  )
}

function choosePixelsPerSecond(durationSeconds: number): number {
  if (durationSeconds <= 12) {
    return 110
  }
  if (durationSeconds <= 60) {
    return 54
  }
  if (durationSeconds <= 180) {
    return 28
  }
  return 16
}

function makeTimeTicks(durationSeconds: number): TimeTick[] {
  if (durationSeconds <= 0) {
    return []
  }

  const interval = chooseTickInterval(durationSeconds)
  const ticks: TimeTick[] = []
  for (let time = 0; time <= durationSeconds + interval * 0.25; time += interval) {
    const clampedTime = Math.min(time, durationSeconds)
    ticks.push({
      timeSeconds: clampedTime,
      percent: (clampedTime / durationSeconds) * 100,
      label: formatTimecode(clampedTime),
    })
  }

  if (ticks[ticks.length - 1]?.timeSeconds !== durationSeconds) {
    ticks.push({
      timeSeconds: durationSeconds,
      percent: 100,
      label: formatTimecode(durationSeconds),
    })
  }

  return ticks
}

function chooseTickInterval(durationSeconds: number): number {
  const targetTicks = 8
  const roughInterval = durationSeconds / targetTicks
  const intervals = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
  return intervals.find((interval) => interval >= roughInterval) ?? 300
}

function formatSourceDuration(sourceDurationSeconds: number | null): string {
  if (sourceDurationSeconds === null || sourceDurationSeconds <= 0) {
    return ""
  }

  return ` of ${formatTimecode(sourceDurationSeconds)} source`
}

function formatTimecode(seconds: number): string {
  const safeSeconds = Math.max(0, seconds)
  const wholeSeconds = Math.floor(safeSeconds)
  const minutes = Math.floor(wholeSeconds / 60)
  const remainderSeconds = wholeSeconds % 60
  const fraction = safeSeconds < 10 ? `.${Math.floor((safeSeconds % 1) * 10)}` : ""
  return `${minutes}:${remainderSeconds.toString().padStart(2, "0")}${fraction}`
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}
