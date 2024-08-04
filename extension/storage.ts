import {APP_NAME} from './constants';

export async function setChromeStorage(data: { imgSrc: string, embedding: number[] }[]): Promise<void> {
    return new Promise((resolve, reject) => {
      const storageData = { [APP_NAME]: JSON.stringify(data) };
      chrome.storage.local.set(storageData, () => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve();
        }
      });
    });
  }


export async function getChromeStorage(): Promise<{ imgSrc: string, embedding: number[] }[]> {
    return new Promise((resolve, reject) => {
      chrome.storage.local.get([APP_NAME], (result) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          const data = result[APP_NAME] ? JSON.parse(result[APP_NAME]) : [];
          if (Array.isArray(data)) {
            resolve(data);
          } else {
            resolve([]); // Return an empty array if the parsed data is not an array
          }
        }
      });
    });
  }