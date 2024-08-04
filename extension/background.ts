// background.ts
import { InferenceSession, Tensor, env } from 'onnxruntime-web';

import { computeFeatureVector, cosineSimilarity } from './math';

import { convertFromRegularArray } from './utils';

import * as ort from 'onnxruntime-web';

const APP_NAME = 'linked-out';

//must set wasm path override
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': 'wasm/onnxruntime-web/dist/ort-wasm.wasm',
  'ort-wasm-threaded.wasm': 'wasm/onnxruntime-web/dist/ort-wasm-threaded.wasm',
  'ort-wasm-simd.wasm': 'wasm/onnxruntime-web/dist/ort-wasm-simd.wasm'
};
ort.env.wasm.numThreads = 1;
const modelPath = 'model.onnx';

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


initializeSession(modelPath).then(initializedSession => {
  if (initializedSession) {
    session = initializedSession;
    console.log('ONNX session initialized successfully in the background script with CPU backend.');
  } else {
    console.error('No available backend found for ONNX model.');
  }
});

class FilteredImagesEmbeddingsStorage {
  async getEmbeddings(): Promise<{ imgSrc: string, embedding: Float32Array }[]> {
    const result = await chrome.storage.local.get([APP_NAME]);
    const data = result[APP_NAME];

    if (data) {
      const parsedData = JSON.parse(data);
      return parsedData.map((item: { imgSrc: string; embedding: number[] }) => ({
        imgSrc: item.imgSrc,
        embedding: new Float32Array(item.embedding),
      }));
    }

    return [];
  }
}

const filteredImagesEmbeddingsStorage = new FilteredImagesEmbeddingsStorage();

async function handleGetPostsToDelete(posts: { imageSrc: string }[]) {
  let postsToRemove: { imageSrc: string, reason: string }[] = [];
  const threshold = 0.8;
  
  if (!session) {
    console.error('ONNX session is not initialized.');
    return postsToRemove;
  }

  const referenceEmbeddings = await filteredImagesEmbeddingsStorage.getEmbeddings();

  for (let post of posts) {
    if (post.imageSrc) {
      console.log(`Processing post with image: ${post.imageSrc}`);
      const postEmbedding = await computeFeatureVector(session, post.imageSrc);

      for (let referenceEmbedding of referenceEmbeddings) {
        const similarity = cosineSimilarity(referenceEmbedding.embedding, postEmbedding);
        console.log(`Similarity: ${similarity} for {${post.imageSrc}`);
        if (similarity > threshold) {
          postsToRemove.push({ ...post, reason: "Image similarity" });
          break;
        }
      }
    }
  }
  return postsToRemove;
}

chrome.runtime.onConnect.addListener(function (port) {
  console.log("Listening")
  console.assert(port.name === APP_NAME);

  port.onMessage.addListener(async function (msg) {
    if (msg.data){
      console.table(msg.data);
    }
    switch (msg.type) {
      case "CONNECTION":
        console.log("Connection established. Getting posts");  
        port.postMessage({ type: "GET_POSTS" });
        break;
      case "GET_POSTS_TO_DELETE":
        if (session) {
          const postsToRemove = await handleGetPostsToDelete(msg.data);
          port.postMessage({ type: "DELETE_POSTS", data: postsToRemove });
        } else {
          console.error('ONNX session not ready.');
        }
        break;
      default:
        console.error("Unknown message type");
    }
  });
});