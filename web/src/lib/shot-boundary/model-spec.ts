export const modelSpec = {
  inputWidth: 48,
  inputHeight: 27,
  inputChannels: 3,
  windowFrames: 100,
  contextFrames: 25,
  outputFramesPerWindow: 50,
  transnetV2SceneThreshold: 0.02,
} as const;

export function frameBytes(): number {
  return modelSpec.inputWidth * modelSpec.inputHeight * modelSpec.inputChannels;
}
