// content.ts
function sendImageToBackground(imageSrc: string) {
  chrome.runtime.sendMessage({ imageSrc }, (response) => {
    if (response.hidePost) {
      // Logic to hide the post, for example by setting display: none
      const postElement = document.querySelector(`img[src="${imageSrc}"]`)?.closest('div');
      if (postElement) {
        postElement.style.display = 'none';
        console.log(`Post with image source ${imageSrc} has been hidden.`);
      } else {
        console.log(`Post element for image ${imageSrc} not found.`);
      }
    }
  });
}

function processImages() {
  console.log("[linked-out] Processing images");
  const images = document.querySelectorAll<HTMLImageElement>('img');
  images.forEach((img) => {
    // Ensure the image has a valid src and has not been processed yet
    if (img.src && !img.dataset.processed) {
      img.dataset.processed = 'true'; // Mark as processed
      console.log(['[linked-out] Processing image', img.src]);
      sendImageToBackground(img.src);
    }
  });
}

// Run the process on initial load
processImages();

// Also run the process when new content is loaded (e.g., when scrolling)
const observer = new MutationObserver(processImages);
observer.observe(document.body, { childList: true, subtree: true });
