import { InferenceSession, Tensor } from 'onnxruntime-web';

function resizeAndPadImage(bitmap: ImageBitmap, targetSize: number): OffscreenCanvas {
    const canvas = new OffscreenCanvas(targetSize, targetSize);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to get canvas context');
  
    // Calculate the scaling factor and new dimensions
    const scale = Math.min(targetSize / bitmap.width, targetSize / bitmap.height);
    const width = Math.round(bitmap.width * scale);
    const height = Math.round(bitmap.height * scale);
  
    // Set canvas size to the target size
    canvas.width = targetSize;
    canvas.height = targetSize;
  
    // Calculate the position to center the image on the canvas
    const xOffset = Math.floor((targetSize - width) / 2);
    const yOffset = Math.floor((targetSize - height) / 2);
  
    // Fill the canvas with a padding color, e.g., black
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, targetSize, targetSize);
  
    // Draw the resized image onto the canvas
    ctx.drawImage(bitmap, xOffset, yOffset, width, height);
  
    return canvas;
  }
  
export async function computeFeatureVector(session: InferenceSession, imageSrc: string): Promise<Float32Array> {
    const response = await fetch(imageSrc, { mode: 'cors' });
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }
  
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);

    const canvas = resizeAndPadImage(bitmap, 224);
  
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