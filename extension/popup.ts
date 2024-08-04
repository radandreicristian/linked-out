// popup.ts
import { InferenceSession, Tensor } from 'onnxruntime-web';
import * as ort from 'onnxruntime-web';

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

document.addEventListener('DOMContentLoaded', () => {
  const uploadElement = document.getElementById('upload') as HTMLInputElement;
  const imageListElement = document.getElementById('image-list') as HTMLDivElement;
  const saveButton = document.getElementById('save-images') as HTMLButtonElement;
  const clearButton = document.getElementById('clear-all') as HTMLButtonElement;

  // Load the model
  loadModel();

  // Load saved images and embeddings on startup
  chrome.storage.local.get(['filterImages', 'filterEmbeddings'], (result) => {
    console.log('Loading filter images and embeddings from storage');
    const filterImages: string[] = result.filterImages || [];
    const filterEmbeddings: Float32Array[] = (result.filterEmbeddings || []);
    console.log(`Loaded ${filterImages.length} stored images and ${filterEmbeddings.length} embeddings`);
    console.log(`Embedding type ${typeof filterEmbeddings}, ${typeof filterEmbeddings[0]}`);

    // Restore images and embeddings
    filterImages.forEach((src, index) => {
      appendImage(src);
      const embedding = new Float32Array(filterEmbeddings[index]);
      console.log(`Restored embedding ${index}:`, embedding);
    });
  });

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
          const embedding = await computeFeatureVector(imgSrc);
          console.log(`Computed embedding for image ${i}:`, embedding);
          imagesToSave.push(imgSrc);
          embeddingsToSave.push(embedding);
          console.log(`Embeddings to save size ${embeddingsToSave.length}, first element ${embeddingsToSave[0]}, type ${typeof embeddingsToSave[0]}`); 
        };
        reader.readAsDataURL(file);
      });
    }
  });
function setChromeStorage(items: object): Promise<void> {
  return new Promise((resolve, reject) => {
    chrome.storage.local.set(items, () => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve();
      }
    });
  });
}

function getChromeStorage(keys: string[]): Promise<{ [key: string]: any }> {
  return new Promise((resolve, reject) => {
    chrome.storage.local.get(keys, (result) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(result);
      }
    });
  });
}
function convertToRegularArray(typedArray: Float32Array[]): number[][] {
  return typedArray.map(array => Array.from(array));
}

saveButton.addEventListener('click', async () => {
  try {
    const result = await getChromeStorage(['filterImages', 'filterEmbeddings']);
    const existingImages: string[] = result.filterImages || [];
    const existingEmbeddings: number[][] = result.filterEmbeddings || [];

    const filterImages = existingImages.concat(imagesToSave);
    const filterEmbeddings = existingEmbeddings.concat(convertToRegularArray(embeddingsToSave));

    console.log(`In saveButton listener: ${existingEmbeddings.length}, ${embeddingsToSave.length}, ${filterEmbeddings.length}`);

    await setChromeStorage({ filterImages, filterEmbeddings });
    console.log('Filter images and embeddings saved.');
    console.log(`Total stored images: ${filterImages.length}`);
    console.log(`Total stored embeddings: ${filterEmbeddings.length}`);
    filterEmbeddings.forEach((emb, index) => {
      console.log(`Stored embedding ${index} shape: ${emb.length}`);
    });

    // Clear the arrays after saving
    imagesToSave = [];
    embeddingsToSave = [];

    // Check the saved data
    const updatedResult = await getChromeStorage(['filterImages', 'filterEmbeddings']);
    console.log(`After persisting: ${updatedResult.filterImages.length}, ${updatedResult.filterEmbeddings.length}`);
  } catch (error) {
    console.error('Error saving to chrome storage:', error);
  }
   });

  // Clear all uploaded images and embeddings
  clearButton.addEventListener('click', () => {
    imageListElement.innerHTML = '';
    chrome.storage.local.remove(['filterImages', 'filterEmbeddings'], () => {
      console.log('All filter images and embeddings cleared.');
    });
  });

  // Function to append an image to the image list
  function appendImage(src: string) {
    const imgElement = document.createElement('img');
    imgElement.src = src;
    imgElement.classList.add('filter-image');
    imageListElement.appendChild(imgElement);
  }

  // Function to compute the feature vector for an image
  async function computeFeatureVector(imageSrc: string): Promise<Float32Array> {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = 'anonymous';
      image.src = imageSrc;

      image.onload = () => {
        // Create a canvas to draw the resized image
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');

        if (ctx) {
          // Draw the image on the canvas with 512x512 size
          ctx.drawImage(image, 0, 0, 512, 512);

          // Extract the image data from the canvas
          const imageData = ctx.getImageData(0, 0, 512, 512);
          const { data } = imageData;

          // Normalize the pixel data and create a tensor
          const float32Data = new Float32Array(data.length / 4 * 3);
          for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
            float32Data[j] = data[i] / 255;     // R
            float32Data[j + 1] = data[i + 1] / 255; // G
            float32Data[j + 2] = data[i + 2] / 255; // B
          }

          const inputTensor = new Tensor('float32', float32Data, [1, 3, 512, 512]);

          // Run the model to get the feature vector
          session!.run({ input: inputTensor }).then((output) => {
            const featureVector = output[session!.outputNames[0]].data as Float32Array;
            console.log(`Computed feature vector with shape: ${featureVector.length}, type ${typeof featureVector}`);
            resolve(featureVector);
          }).catch(reject);
        } else {
          reject(new Error('Failed to get canvas context'));
        }
      };

      image.onerror = () => {
        reject(new Error('Failed to load image'));
      };
    });
  }
});
