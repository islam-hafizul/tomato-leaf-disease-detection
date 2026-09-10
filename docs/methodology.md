# Project Methodology

## 1. Overview

This project develops an AI-based tomato leaf disease classification
system using deep learning and computer vision.

The overall methodology consists of the following stages:

```text
Dataset
   ↓
Data Quality Audit
   ↓
Duplicate Detection
   ↓
Custom Stratified 5-Fold Cross-Validation
   ↓
Image Preprocessing & Augmentation
   ↓
Transfer Learning with ResNet18
   ↓
Model Training
   ↓
Multi-Fold Evaluation
   ↓
External Evaluation on Taiwan Dataset
   ↓
Error Analysis & Explainability
```

The objective is not only to achieve high classification accuracy, but
also to establish a reproducible evaluation procedure and investigate
how well the model generalizes to images from a different source.

---

## 2. Data Preparation

The project uses the **Dataset of Tomato Leaves**, obtained from
Mendeley Data.

The dataset contains two image sources:

1. PlantVillage-derived tomato leaf images
2. Taiwan tomato leaf images

The PlantVillage-derived dataset is used as the primary dataset for model development and cross-validation. The dataset provides five predefined cross-validation splits. Before model training, the dataset is inspected for:

* Invalid or corrupted images
* Unexpected image dimensions
* Incorrect file formats
* Duplicate images
* Potential train/test leakage

SHA-256 hashes are used to identify exact duplicate image files.

The PlantVillage-derived dataset contains 14,531 files, including 10 exact duplicate pairs. Therefore, 14,521 unique image contents are used when generating the project's custom cross-validation manifest.

Because of the duplicate images, a custom train/validation split was created from the preprocessed PlantVillage dataset to maintain control over the experimental split.

More details are available in:

* [Dataset](/docs/dataset.md)

---

## 3. Train/Validation Split

The 14,531 images were divided into:

- **Training:** 11,616 images
- **Validation:** 2,905 images
- **Split:** 80/20

The split is performed using the dataset classes to maintain class representation across the two subsets.

---

## 4. Image Preprocessing

The original PlantVillage-derived images are 227 × 227 pixels. Images are resized to **224 × 224 pixels** to match the expected input size of ResNet18.

The following preprocessing is applied:

1. Resize image to 224 × 224
2. Convert image to tensor
3. Normalize using ImageNet mean and standard deviation

The ImageNet normalization values are:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

The same normalization is used for both training and validation
images.

---

## 5. Model Architecture

The initial model is **ResNet18** with ImageNet-pretrained weights.

ResNet18 was selected as the initial baseline because it provides a
good balance between:

* Model capacity
* Computational cost
* Training speed
* Transfer-learning performance
* Suitability for limited GPU memory

The final fully connected layer of the pretrained network is replaced
with a new classification layer containing 10 output neurons.

```text
Image
  │
  ▼
ResNet18 Backbone
(ImageNet pretrained)
  │
  ▼
512-dimensional feature representation
  │
  ▼
Fully Connected Layer
512 → 10
  │
  ▼
Class Prediction
```

The ten output classes correspond to the ten PlantVillage tomato leaf
categories.

### Transfer Learning

The model starts from weights pretrained on ImageNet rather than
training the entire network from random initialization.

The pretrained ResNet18 backbone provides visual features learned
from a large and diverse image dataset.

The final classification layer is replaced to match the ten tomato
leaf categories.

The initial experiment fine-tunes the complete network rather than
freezing the pretrained backbone.

This approach allows the model to adapt the pretrained visual
features to the specific characteristics of tomato leaf diseases.

---

## 6. Training Configuration

The initial baseline configuration is:

| Parameter             | Value              |
| --------------------- | ------------------ |
| Model                 | ResNet18           |
| Pretrained weights    | ImageNet           |
| Input size            | 224 × 224          |
| Batch size            | 16                 |
| Optimizer             | Adam               |
| Learning rate         | 0.0001             |
| Loss function         | Cross-Entropy Loss |
| Initial epochs        | 10                 |
| Cross-validation      | 5-fold             |
| Random seed for split | 42                 |

The current training configuration is intentionally conservative
because the project is being developed on a GPU with limited VRAM.

Training is performed using CUDA when a compatible NVIDIA GPU is
available.

---

## 7. Evaluation Metrics

The best model checkpoint is evaluated on the held-out validation set.

The evaluation includes:

- Overall accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class performance

The confusion matrix and individual predictions are saved under the `results/` directory.

After cross-validation on the PlantVillage-derived dataset, the model will be evaluated on the original Taiwan dataset.

---

## 8. Experimental Result

The initial ResNet18 experiment achieved:

**Validation Accuracy: 99.41%**

This result should be considered an initial benchmark rather than a definitive estimate of real-world performance. Further experiments using the dataset's cross-validation structure and evaluation on independent field images would provide a stronger assessment of generalization.

---

## 9. Planned Pipeline

The complete project pipeline is:

```text
                    Dataset of Tomato Leaves
                              │
                 ┌────────────┴────────────┐
                 │                         │
          PlantVillage                  Taiwan
                 │                         │
           Data Audit                 Original Data
                 │                         │
          Deduplication                    │
                 │                         │
       Custom 5-Fold CV                    │
                 │                         │
          ResNet18 Training                │
                 │                         │
          Cross-Validation                 │
                 │                         │
                 └────────────┬────────────┘
                              │
                     External Evaluation
                              │
                       Error Analysis
                              │
                           Grad-CAM
                              │
                    FastAPI Model Serving
                              │
                       React Frontend
```

The initial objective is to establish a reliable and reproducible
classification baseline before adding explainability and deployment
components.
