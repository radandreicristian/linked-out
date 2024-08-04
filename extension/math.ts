import { InferenceSession, Tensor } from 'onnxruntime-web';

export async function computeFeatureVector(session: InferenceSession, imageSrc: string): Promise<Float32Array> {
  const response = await fetch(imageSrc, { mode: 'cors' });
  if (!response.ok) {
    throw new Error(`Failed to fetch image: ${response.statusText}`);
  }

  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);

  const canvas = new OffscreenCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get canvas context');

  ctx.drawImage(bitmap, 0, 0, 224, 224);
  const imageData = ctx.getImageData(0, 0, 224, 224);
  const { data } = imageData;
  

  const float32Data = new Float32Array(data.length / 4 * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    float32Data[j] = data[i] / 255;
    float32Data[j + 1] = data[i + 1] / 255;
    float32Data[j + 2] = data[i + 2] / 255;
  }

  const inputTensor = new Tensor('float32', float32Data, [1, 3, 224, 224]);
  const output = await session.run({ input: inputTensor });
  const featureVector = output[session.outputNames[0]].data as Float32Array;
  return featureVector;
}

export function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] ** 2;
    normB += vecB[i] ** 2;
  }

  if (normA === 0 || normB === 0) return 0;

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
