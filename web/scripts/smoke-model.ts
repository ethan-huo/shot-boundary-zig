import * as ort from "onnxruntime-web";

const modelPath = Bun.argv[2] ?? "public/models/transnetv2.onnx";

ort.env.wasm.numThreads = 1;

const startedLoad = performance.now();
const session = await ort.InferenceSession.create(modelPath, {
  executionProviders: ["wasm"],
  freeDimensionOverrides: { batch: 1 },
});
const loadMs = performance.now() - startedLoad;

const dims = [1, 100, 27, 48, 3];
const length = dims.reduce((total, value) => total * value, 1);
const data = new Uint8Array(length);
for (let index = 0; index < data.length; index += 1) {
  data[index] = index & 255;
}

const startedRun = performance.now();
const outputs = await session.run({
  frames: new ort.Tensor("uint8", data, dims),
});
const runMs = performance.now() - startedRun;

console.log(
  JSON.stringify(
    {
      modelPath,
      loadMs,
      runMs,
      outputs: Object.fromEntries(
        Object.entries(outputs).map((entry) => {
          const name = entry[0];
          const tensor = entry[1];
          return [
            name,
            {
              type: tensor.type,
              dims: tensor.dims,
              first: firstValues(tensor.data),
            },
          ];
        }),
      ),
    },
    null,
    2,
  ),
);

function firstValues(data: unknown): Array<number | string> {
  if (Array.isArray(data)) {
    return data.slice(0, 5);
  }
  if (!ArrayBuffer.isView(data)) {
    return [];
  }

  if (data instanceof DataView) {
    return [];
  }

  const view = data as unknown as { length: number; [index: number]: number | bigint };
  const values: Array<number | string> = [];
  for (let index = 0; index < Math.min(5, view.length); index += 1) {
    const value = view[index];
    if (value !== undefined) {
      values.push(typeof value === "bigint" ? value.toString() : value);
    }
  }
  return values;
}
