# LinkedOut - Purge Memes from Your Feed

This extension helps professionals keep their LinkedIn feeds professional.

=I've seen one too many memes with [Yusuf Dikec](https://www.google.com/search?q=yusuf+dikec&sa=X&sca_esv=f7839580003801af&sca_upv=1&sxsrf=ADLYWIKqjy8VveYuINKIuDn8h4r3fS6zHw:1722785722145&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWfbQph1uib-VfD_izZO2Y5sC3UdQE5x8XNnxUO1qJLaQR2rwhLa89xym2ORbEZb-gP1zIcrSJSb5m5VLWXlSjIaIG1x3OUX72o1bhPlysQIw2wexIfjr9hIq56rRLI7yjvrm-eU2rldmMeoWofpb8CbZ_Suo&ved=2ahUKEwjr9vf31NuHAxXNhP0HHVmCB8EQtKgLegQIAhAG&biw=1920&bih=934&dpr=1) for my own good in the past week. From attention grabbing to sales and product marketing, it's been adding a lot of noise.

So I've created LinkedOut.

# Project Description

The project has two parts: 
- A Python script that exports a pre-trained neural network to the ONNX format
- A Chrome extension that filters LinkedIn feed posts

The Python script requires Python 3.10 and Poetry to be installed. Oh, and the Torch versions are for MacOS only. You're better off just using `models/model.onnx`, which is an exported version of [EfficientNetB0](https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html).

The Chrome extension loads the model in ONNX format via `onnxruntime-web` and computes embeddings for:
- User-uploaded images, in the extension popup.
- Posts on the LinkedIn feed, while you are scrolling.

If the similarity between the image of a LinkedIn post and any of the user-uploaded images is greater than a threshold, it removes the post from the feed.


# Setup

For the extension, you can build it locally by doing
```
npm run build
```
To use it in Chrome:
- Navigate to `chrome://extensions`
- Enabling `Developer Mode`
- Click `Load unpacked`
- Point to the `dist/` folder in the extension repo, which was generated by the `npm run build` command.

If you want to use the Python script:
```
> poetry env use python3.10
> poetry install
> poetry shell
> (image-embedding-py3.10) python model.py
```

And then your model should be exported under `models/model.onnx`.

# Disclaimers

Q: Does it work?
A: Not really. The similarity scores are weird sometimes, I've had both false positives and false negatives.

Q: Does it impact my Chrome experience?
A: You will probably have small delay/lag when loading your LinkedIn feed. This also depends on your resoruces.

Q: Is this extension production-ready?
A: No, I've just scaffolded it in a few hours with ChatGPT. I can conut on the fingers from my right hand how many times I have written JS/TS before. Apologies if the code makes your eyes bleed.

Q: Can I fork this and do whatever I want?
A: Yes, as long as you're not making money from it.

Q: Why no Docker for the Python thing?
A: :shrug: 