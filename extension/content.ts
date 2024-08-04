interface Post {
  id: number;
  imageSrc: string;
}

let postsIdToNodes: { [key: number]: HTMLElement } = {};
let newPostId = 0;

function handleGetPosts(): Post[] {
  const postsNodes = document.querySelectorAll("div.scaffold-finite-scroll__content > div");
  console.log(`Found ${postsNodes.length} posts`);
  let posts: Post[] = [];

  for (let node of postsNodes) {
    try {
      const postElement = node as HTMLElement;

      if (postElement.id) {
        console.log(`Post with ${postElement.id} already visited, skipping`);
        continue;
      }

      // Using the identified selector for images in LinkedIn posts
      const imgElement = postElement.querySelector("img.update-components-image__image") as HTMLImageElement;
      if (imgElement && imgElement.src) {
        console.log(`Found valid feed image ${imgElement.src}.`);
        posts.push({ imageSrc: imgElement.src, id: newPostId });
        postElement.id = newPostId.toString();
        postsIdToNodes[newPostId] = postElement;
        newPostId++;
      }
    } catch (TypeError) {
      console.log("Bad node!");
    }
  }
  return posts;
}

function handleDeletePosts(posts: Post[]) {
  for (let post of posts) {
    const { id } = post;
    const node = postsIdToNodes[id];
    if (node) {
      node.parentNode?.removeChild(node);
      delete postsIdToNodes[id];
    }
  }
}

const port = chrome.runtime.connect({ name: "linked-out" });
port.postMessage({ type: "CONNECTION" });

port.onMessage.addListener((msg) => {
  console.table(msg.data);
  switch (msg.type) {
    case "GET_POSTS":
      const posts = handleGetPosts();
      port.postMessage({ type: "GET_POSTS_TO_DELETE", data: posts });
      break;
    case "DELETE_POSTS":
      handleDeletePosts(msg.data);
      break;
    default:
      console.error("Unknown message type");
  }
});

// Re-check posts on scroll
window.addEventListener("scroll", () => port.postMessage({ type: "CONNECTION" }));

// Periodically re-check posts
setInterval(() => port.postMessage({ type: "CONNECTION" }), 1000);
