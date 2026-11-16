---
title: PawVision AI - Cat vs Dog Classifier
emoji: 🐾
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# PawVision AI — Cat vs Dog Image Classification

A Streamlit application that uses a trained MobileNetV2 model to classify uploaded images as Cat or Dog.

## Required model file

Place the trained model at:

`model/mobilenetv2_cat_dog.keras`

The app expects the model trained by the supplied notebook's MobileNetV2 workflow:
- Input image: RGB
- Image size: 128 × 128
- Training labels: Cat = 0, Dog = 1
- Output: sigmoid probability for Dog
- Preprocessing: `tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)`

The notebook's evaluation pipeline first scales images to [0, 1], then calls `preprocess_input(images * 255.0)`. This app uses the equivalent operation on 0–255 RGB pixels.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hugging Face Spaces deployment

1. Create a new Space and select **Docker**.
2. Upload all files from this repository.
3. Add the trained `.keras` file to `model/mobilenetv2_cat_dog.keras`.
4. Commit the files. The Space builds from the Dockerfile.
5. Open the Space URL and test with cat and dog images.

Do not commit private access tokens or credentials.
