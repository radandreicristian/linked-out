// background.ts
import { InferenceSession, Tensor, env } from 'onnxruntime-web';

import * as ort from 'onnxruntime-web';

//must set wasm path override
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': 'wasm/onnxruntime-web/dist/ort-wasm.wasm',
  'ort-wasm-threaded.wasm': 'wasm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
  'ort-wasm-simd.wasm': 'wasm/onnxruntime-web/dist/ort-wasm-simd.wasm'
};
ort.env.wasm.numThreads = 1;

let session: InferenceSession | null = null;

async function initializeSession(modelPath: string): Promise<InferenceSession | null> {
  try {
    session = await InferenceSession.create(modelPath, {
      executionProviders: ['wasm', 'cpu']
    });
    console.log('ONNX model loaded with WASM or CPU backend');
  } catch (error) {
    console.error('Failed to initialize ONNX model:', error);
    session = null;
  }

  return session;
}

const modelPath = 'model.onnx';

initializeSession(modelPath).then(initializedSession => {
  if (initializedSession) {
    session = initializedSession;
    console.log('ONNX session initialized successfully in the background script with CPU backend.');
  } else {
    console.error('No available backend found for ONNX model.');
  }
});

chrome.runtime.onInstalled.addListener(() => {
  console.log('Extension installed and background script loaded.');
});

function convertFromRegularArray(arrays: number[][]): Float32Array[] {
  return arrays.map(array => new Float32Array(array));
}

// Example usage in background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Received message:', request);

  if (request.imageSrc && session) {
    console.log(`Processing image: ${request.imageSrc}`);
    chrome.storage.local.get(['filterEmbeddings'], async (result) => {
      const storedEmbeddings = result.filterEmbeddings || [];
      const filterEmbeddings = convertFromRegularArray(storedEmbeddings);

      if (filterEmbeddings.length > 0) {
        const imageEmbedding = await computeFeatureVector(request.imageSrc);

        const THRESHOLD = 0.8; // Threshold value for similarity
        let shouldHide = false;

        filterEmbeddings.forEach((embedding, index) => {
          const similarity = cosineSimilarity(embedding, imageEmbedding);
          if (similarity > THRESHOLD) {
            console.log(`Similarity score of image ${request.imgSrc} with stored embedding ${index}: ${similarity} above threshold.`);
            shouldHide = true;
          }
        });

        if (shouldHide && sender.tab?.id) {
          console.log(`Hiding post with image: ${request.imageSrc}`);
          chrome.tabs.sendMessage(sender.tab.id, { hidePost: true, imageSrc: request.imageSrc });
        } else {
          console.log(`Post with image ${request.imageSrc} not hidden; similarity scores calculated.`);
        }
      } else {
        console.log('No filter embeddings found.');
      }
    });
  } else if (!session) {
    console.error('Session not initialized; unable to process image.');
  }
});

async function computeFeatureVector(imageSrc: string): Promise<Float32Array> {
  console.log(`Starting computation for image: ${imageSrc}`);
  
  try {
    console.log('Fetching the image as a blob...');
    // Fetch the image as a blob
    const response = await fetch(imageSrc, { mode: 'cors' });
    if (!response.ok) {
      console.error(`Failed to fetch image: ${response.statusText}`);
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }

    console.log('Image fetched successfully.');
    const blob = await response.blob();
    console.log('Blob size:', blob.size);

    console.log('Creating ImageBitmap...');
    const bitmap = await createImageBitmap(blob);
    console.log('ImageBitmap created successfully.');

    // Use OffscreenCanvas instead of HTMLCanvasElement
    console.log('Creating OffscreenCanvas...');
    const canvas = new OffscreenCanvas(512, 512);
    const ctx = canvas.getContext('2d');

    if (!ctx) {
      console.error('Failed to get canvas context');
      throw new Error('Failed to get canvas context');
    }

    console.log('Drawing image on OffscreenCanvas...');
    ctx.drawImage(bitmap, 0, 0, 512, 512);

    console.log('Extracting image data...');
    const imageData = ctx.getImageData(0, 0, 512, 512);
    const { data } = imageData;
    console.log('Image data extracted. Data length:', data.length);

    console.log('Normalizing pixel data and creating tensor...');
    const float32Data = new Float32Array(data.length / 4 * 3);
    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
      float32Data[j] = data[i] / 255;     // R
      float32Data[j + 1] = data[i + 1] / 255; // G
      float32Data[j + 2] = data[i + 2] / 255; // B
    }

    console.log('Tensor data created.');
    const inputTensor = new Tensor('float32', float32Data, [1, 3, 512, 512]);

    console.log('Running the model...');
    const output = await session!.run({ input: inputTensor });
    const featureVector = output[session!.outputNames[0]].data as Float32Array;
    console.log(`Computed feature vector with shape: ${featureVector.length}`);

    return featureVector;
  } catch (error) {
    console.error('Error processing image:', error);
    throw error;
  }
}


// Function to calculate cosine similarity
function cosineSimilarity(vecA: Float32Array, vecB: Float32Array): number {

  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] ** 2;
    normB += vecB[i] ** 2;
  }

  normA = Math.sqrt(normA);
  normB = Math.sqrt(normB);

  if (normA === 0 || normB === 0) {
    console.warn('One of the vectors has zero length.');
    return NaN; // Return NaN or some other default value indicating no similarity
  }

  // Calculate cosine similarity
  const similarity = dotProduct / (normA * normB);
  console.log(`Cosine similarity: ${similarity}`);

  if (isNaN(similarity) || !isFinite(similarity)) {
    console.error('Cosine similarity calculation resulted in NaN or Infinity.');
  }

  return similarity;
}