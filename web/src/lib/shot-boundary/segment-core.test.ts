import { expect, test } from "bun:test";
import { predictionsToScenes, windowSourceIndices } from "./segment-core";

test("windows match native center stride and trim policy", () => {
  const windows = windowSourceIndices(51);

  expect(windows.length).toBe(2);
  expect(windows[0]?.[0]).toBe(0);
  expect(windows[0]?.[25]).toBe(0);
  expect(windows[0]?.[74]).toBe(49);
  expect(windows[1]?.[0]).toBe(25);
  expect(windows[1]?.[25]).toBe(50);
  expect(windows[1]?.[50]).toBe(50);
  expect(windows[1]?.[99]).toBe(50);
});

test("scene postprocess mirrors native transition policy", () => {
  const scenes = predictionsToScenes([0.1, 0.2, 0.8, 0.7, 0.1, 0.2], 0.5);

  expect(scenes).toEqual([
    { start: 0, end: 2 },
    { start: 4, end: 5 },
  ]);
});

test("scene postprocess keeps all-transition fallback", () => {
  const scenes = predictionsToScenes([0.9, 0.8, 0.7], 0.5);

  expect(scenes).toEqual([{ start: 0, end: 2 }]);
});
