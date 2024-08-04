export function convertToRegularArray(typedArray: Float32Array[]): number[][] {
  return typedArray.map(array => Array.from(array));
}


export function convertFromRegularArray(arrays: number[][]): Float32Array[] {
  return arrays.map(array => new Float32Array(array));
}
