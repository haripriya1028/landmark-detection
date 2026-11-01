# Landmark Detection

A computer vision project that recognizes landmark categories in photographs using transfer learning with VGG19 and a filtered subset of Google Landmarks Dataset v2.

Developed as part of the **Artificial Intelligence with Python internship at Coincent**.

## Overview

This project uses an ImageNet-pretrained VGG19 network as a frozen feature extractor and trains a new classification head to recognize landmark categories.

Although the project is titled **Landmark Detection**, the implementation performs image classification. It predicts a landmark category without generating bounding boxes or geographic coordinates.

## Features

- Image acquisition from packaged GLDv2 archives
- Image-to-label matching using dataset metadata
- Class-frequency filtering and per-class image limits
- Stratified training and validation split
- Transfer learning with a frozen VGG19 model
- Batch-wise training
- Analysis of correct and incorrect predictions

## Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- pandas
- scikit-learn
- urllib
- tarfile

## Dataset

The project uses a filtered subset of **Google Landmarks Dataset v2 (GLDv2)**.

### Dataset Preparation

- Download and extract `images_000.tar` through `images_004.tar`.
- Match extracted image filenames with image IDs in `train.csv`.
- Retrieve the corresponding `landmark_id` for each image.
- Exclude classes containing fewer than 15 available images.
- Limit each retained class to a maximum of 30 images.
- Convert landmark IDs into sequential labels using `LabelEncoder`.

The project report describes approximately 40,000 raw images before filtering and an experiment involving 15 landmark classes. The raw image count is not the final training-set size.

The number of classes produced by the code depends on the downloaded images and filtering results.

## Methodology

### 1. Data Acquisition

Initial downloads from individual image URLs encountered rate limits and incompatible image formats.

The workflow switched to packaged GLDv2 archives containing JPEG images.

### 2. Preprocessing

- Load images using OpenCV.
- Resize images to 224 × 224 pixels.
- Convert images from BGR to RGB.
- Apply the VGG19 `preprocess_input` function.
- Encode landmark labels as integers.

### 3. Training and Validation Split

The filtered dataset is split into:

- **80% training data**
- **20% validation data**

Stratified sampling preserves class proportions, using `random_state=42`.

### 4. Transfer Learning

- Load VGG19 with pretrained ImageNet weights.
- Remove the original classification layers.
- Freeze the VGG19 base.
- Apply global average pooling.
- Add a Dense softmax layer for the retained landmark categories.

### 5. Evaluation

- Generate class probabilities for validation images.
- Select the highest-scoring class using `argmax`.
- Compare predictions with reference labels.
- Collect correct and incorrect prediction records.

## Model Architecture

| Component | Configuration |
|-----------|---------------|
| Input | 224 × 224 × 3 |
| Base network | VGG19 |
| Pretrained weights | ImageNet |
| Original classification layers | Removed using `include_top=False` |
| Base layers | Frozen |
| Pooling | Global average pooling |
| Classification head | Dense with softmax activation |
| Output size | Number of retained landmark classes |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Batch size | 16 |
| Epochs in the supplied implementation | 10 |
| Training method | `train_on_batch` |
| Data split | 80% training, 20% validation |

Training samples are shuffled at the beginning of each epoch.

## Results

| Metric | Result |
|--------|--------|
| Validation classification accuracy | Approximately 40% |
| Classes in the reported experiment | 15 |

Uniform random selection among 15 classes has an expected accuracy of approximately 6.67%. This is a reference point rather than a measured baseline model.

The reported result applies to the selected validation subset and does not represent performance across the complete GLDv2 dataset.

## Challenges and Solutions

### Image Download Failures

Individual image URLs triggered rate limits, and some downloaded files used incompatible formats.

**Solution:** Use packaged GLDv2 image archives.

### Dominant-Class Predictions

An early model predicted one category for many inputs.

**Solution:** Freeze the pretrained VGG19 base and apply the expected VGG19 preprocessing.

### Limited Data and Computing Resources

Working with the complete dataset was impractical.

**Solution:** Filter the dataset and train only a new classification head on top of frozen pretrained features.

## How to Run

1. Install the required libraries:

       pip install tensorflow opencv-python numpy pandas scikit-learn jupyter

2. Open the project notebook in Google Colab or Jupyter.
3. Run the image archive download and extraction cells.
4. Ensure extracted images are available under `train_images/`.
5. Load the metadata and match image IDs to landmark labels.
6. Run class filtering, label encoding and the stratified split.
7. Check the resulting class counts and dataset sizes.
8. Run preprocessing, model construction and training.
9. Evaluate the model and inspect prediction records.

The supplied implementation uses:

- Image archives: `https://s3.amazonaws.com/google-landmark/train/`
- Metadata: `https://s3.amazonaws.com/google-landmark/metadata/train.csv`

Downloading and extracting the archives requires substantial disk space and time. A GPU can accelerate training.

## Limitations

- Uses a small filtered subset of GLDv2.
- Performs landmark classification without localization.
- Similar architectural features can cause confusion.
- Lighting, viewpoint and occlusion can affect predictions.
- Validation accuracy can be sensitive to the limited sample size.
- No separate held-out test-set result is reported.

## Future Improvements

- Increase the number of balanced examples per class.
- Apply image augmentation.
- Fine-tune selected upper VGG19 layers.
- Compare alternative pretrained architectures.
- Add a confusion matrix and per-class evaluation metrics.
- Evaluate on a separate test set.
