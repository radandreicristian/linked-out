// popup.ts
import { InferenceSession, Tensor } from 'onnxruntime-web';
import * as ort from 'onnxruntime-web';
import { convertToRegularArray, convertFromRegularArray} from './utils';
import { getChromeStorage, setChromeStorage } from './storage';
import {APP_NAME} from './constants';
import { computeFeatureVector } from './image';

// Set WASM paths and number of threads
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': 'wasm/onnxruntime-web/dist/ort-wasm.wasm',
  'ort-wasm-threaded.wasm': 'wasm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
  'ort-wasm-simd.wasm': 'wasm/onnxruntime-web/dist/ort-wasm-simd.wasm'
};
ort.env.wasm.numThreads = 1;

let session: InferenceSession | null = null;
let imagesToSave: string[] = [];
let embeddingsToSave: Float32Array[] = [];

async function loadModel() {
  if (!session) {
    try {
      session = await InferenceSession.create('./model.onnx', {
        executionProviders: ['wasm', 'cpu']
      });
      console.log('ONNX model loaded in background');
    } catch (error) {
      console.error('Failed to load ONNX model:', error);
    }
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  const uploadElement = document.getElementById('upload') as HTMLInputElement;
  const saveButton = document.getElementById('save-images') as HTMLButtonElement;
  const clearButton = document.getElementById('clear-all') as HTMLButtonElement;

  // Load the model
  loadModel();

  try {
    const savedItems = await getChromeStorage();
    console.log(`Loaded ${savedItems.length} stored items`);
    savedItems.forEach((item, index) => {
      appendImage(item.imgSrc);
      console.log(`Restored embedding ${index}:`, new Float32Array(item.embedding));
    });
  } catch (error) {
    console.error('Error loading from chrome storage:', error);
  }

  // Handle file uploads
  uploadElement.addEventListener('change', (event) => {
    const files = (event.target as HTMLInputElement).files;
    if (files) {
      imagesToSave = [];
      embeddingsToSave = [];

      Array.from(files).forEach((file, i) => {
        const reader = new FileReader();
        reader.onload = async function (e) {
          const imgSrc = e.target?.result as string;
          appendImage(imgSrc);

          // Compute and store feature vector
          const embedding = await computeFeatureVector(session!, imgSrc);
          console.log(`Computed embedding for src ${imgSrc}:`);
          console.log(`Computed embedding for image ${i}:`, embedding);
          imagesToSave.push(imgSrc);
          embeddingsToSave.push(embedding);
          console.log(`Embeddings to save size ${embeddingsToSave.length}, first element ${embeddingsToSave[0]}, type ${typeof embeddingsToSave[0]}`);
        };
        reader.readAsDataURL(file);
      });
    }
  });

  // Save images and embeddings
  saveButton.addEventListener('click', async () => {
    try {
      const existingData = await getChromeStorage();
      const newData = imagesToSave.map((imgSrc, index) => ({
        imgSrc,
        embedding: Array.from(embeddingsToSave[index])
      }));

      const combinedData = existingData.concat(newData);
      console.log(`In saveButton listener: ${existingData.length}, ${newData.length}, ${combinedData.length}`);

      await setChromeStorage(combinedData);
      console.log('Filter images and embeddings saved.');
      console.log(`Total stored items: ${combinedData.length}`);
      combinedData.forEach((item, index) => {
        console.log(`Stored item ${index}: imgSrc=${item.imgSrc}, embedding length=${item.embedding.length}`);
      });

      // Clear the arrays after saving
      imagesToSave = [];
      embeddingsToSave = [];

      // Check the saved data
      const updatedResult = await getChromeStorage();
      console.log(`After persisting: ${updatedResult.length} items`);
    } catch (error) {
      console.error('Error saving to chrome storage:', error);
    }
  });

  // Clear all uploaded images and embeddings
  clearButton.addEventListener('click', () => {
    document.getElementById('image-list')!.innerHTML = '';
    chrome.storage.local.remove([APP_NAME], () => {
      console.log('All filter images and embeddings cleared.');
    });
  });
});

// Append an image to the image list
function appendImage(src: string) {
  const imageListElement = document.getElementById('image-list') as HTMLDivElement;
  const imgElement = document.createElement('img');
  imgElement.src = src;
  imgElement.classList.add('filter-image');
  imageListElement.appendChild(imgElement);
}
