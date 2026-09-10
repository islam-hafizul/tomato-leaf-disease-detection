# Dataset Documentation

## 1. Dataset Overview

**Dataset Name:** Dataset of Tomato Leaves (Version 1)
**Source:** [Mendeley Data](https://data.mendeley.com/datasets/ngdgg79rzb/1)  
**DOI:** 10.17632/ngdgg79rzb.1

The **Dataset of Tomato Leaves** contains tomato leaf images from
two different sources: a PlantVillage-derived dataset and a Taiwan
tomato leaf dataset. The dataset was published on 27 May 2020 and is
available under the CC BY 4.0 license.

1. PlantVillage-derived tomato leaf images
2. Taiwan tomato leaf images

The PlantVillage-derived dataset is used as the primary training
dataset, while the original Taiwan images are reserved for
cross-domain external evaluation.

---

## 2. Dataset Structure

The dataset is organized into two main sources:

```text
Dataset of Tomato Leaves/
├── plantvillage/
│   ├── 5 cross-validation/
│   └── Preprocessed data/
│
└── taiwan/
    ├── data augmentation/
    └── Preprocessed data/
````

The dataset provides preprocessed images with a resolution of
227 × 227 pixels.

---

## 3. PlantVillage Dataset

The PlantVillage-derived portion contains:

* 14,531 images
* 10 tomato leaf categories
* 227 × 227 pixel images
* JPEG image format

The classes are:

1. Bacterial spot
2. Early blight
3. Healthy
4. Late blight
5. Leaf Mold
6. Septoria leaf spot
7. Target Spot
8. Tomato mosaic virus
9. Tomato yellow leaf curl virus
10. Two-spotted spider mite

### Class Distribution

The distribution of the 14,531 images is:

| Class                         |     Images |
| ----------------------------- | ---------: |
| Bacterial spot                |      1,702 |
| Early blight                  |        800 |
| Healthy                       |      1,272 |
| Late blight                   |      1,528 |
| Leaf Mold                     |        762 |
| Septoria leaf spot            |      1,417 |
| Target Spot                   |      1,124 |
| Tomato yellow leaf curl virus |      4,286 |
| Tomato mosaic virus           |        299 |
| Two-spotted spider mite       |      1,341 |
| **Total**                     | **14,531** |

The dataset is therefore imbalanced, with Tomato yellow leaf curl
virus representing the largest class and Tomato mosaic virus the
smallest class.

---

## 4. Dataset Quality Audit

Before training the model, the dataset was audited for:

* Image validity
* Image dimensions
* File format
* Corrupted images
* Duplicate images
* Train/test leakage

All 14,531 PlantVillage preprocessed files were valid JPEG images
with dimensions of 227 × 227 pixels.

No corrupted images were found.

---

## 5. Duplicate Image Analysis

The audit identified 10 exact duplicate image pairs in the
PlantVillage preprocessed dataset.

These duplicates resulted in:

* 14,531 total files
* 14,521 unique image contents

The duplicate files were not physically deleted. Instead, duplicate
content was identified using SHA-256 hashes and only one copy of
each exact duplicate was included when creating the project's
cross-validation split.

---

## 6. Data Leakage in the Provided Cross-Validation Folds

An important issue was discovered during the dataset audit.

The five cross-validation partitions supplied with the dataset
contain exact duplicate images across their training and testing
sets.

The number of exact overlaps detected within each supplied fold was:

| Fold               | Train/Test Duplicate Overlaps |
| ------------------ | ----------------------------: |
| Cross-validation 1 |                             4 |
| Cross-validation 2 |                             6 |
| Cross-validation 3 |                             4 |
| Cross-validation 4 |                             2 |
| Cross-validation 5 |                             4 |

This creates a potential source of data leakage because the model can
encounter identical image content during training and evaluation.

For this reason, the provided cross-validation partitions are **not
used directly** in this project.

---

## 7. Custom Cross-Validation Strategy

Instead of using the supplied folds, the project generates its own
five-fold stratified cross-validation split.

The procedure is:

```text
Original PlantVillage images
        ↓
SHA-256 duplicate detection
        ↓
Remove duplicate observations from the split
        ↓
14,521 unique images
        ↓
Stratified 5-fold cross-validation
        ↓
Training / validation folds
```

The split is generated using:

* 5 folds
* Stratification by class
* Shuffling enabled
* Random seed: 42

This makes the split reproducible while ensuring that the class
distribution remains approximately consistent across folds.

The generated split manifest is stored at:

```text
data/splits/plantvillage_5fold.csv
```

---

## 8. Taiwan Dataset

The Taiwan portion contains:

* 622 original images
* 6 categories
* 227 × 227 pixel preprocessed images

The classes are:

1. Bacterial spot
2. Black leaf mold
3. Gray leaf spot
4. Healthy
5. Late blight
6. Powdery mildew

The dataset also provides an augmented version containing:

* 4,976 images
* 8 images per original image

The augmentation includes transformations such as rotations,
mirroring, and brightness changes.

---

## 9. Taiwan Data Leakage Consideration

The augmented Taiwan dataset is derived from the 622 original
images. The original images are also present among the augmented
data.

Therefore, the augmented images must not be randomly distributed
between training and testing sets, because visually or exactly
identical versions of the same underlying image could appear in
both sets.

For this reason, the project does not treat the 4,976 augmented
images as independent observations for a random train/test split.

Instead, the original Taiwan images are reserved for external
evaluation.

---

## 10. Cross-Domain Evaluation

The Taiwan dataset is particularly useful for testing whether the
model generalizes beyond the PlantVillage image distribution.

The primary model is trained using the PlantVillage-derived
dataset and evaluated separately on the original Taiwan images.

Only the following three classes have direct label correspondence
between the two datasets:

* Bacterial spot
* Healthy
* Late blight

Therefore, the Taiwan external evaluation will focus on these
shared categories.

```text
PlantVillage
14,521 unique images
        │
        ↓
  5-fold CV training
        │
        ↓
    ResNet18
        │
        ↓
Taiwan original images
    External test
        │
        ↓
Cross-domain robustness
```

---

## 11. Data Preparation Summary

The final data strategy is:

| Dataset                           | Purpose                                     |
| --------------------------------- | ------------------------------------------- |
| PlantVillage preprocessed images  | Primary training and cross-validation       |
| PlantVillage duplicate-free split | Prevent exact-image leakage                 |
| Taiwan original images            | External/cross-domain evaluation            |
| Taiwan augmented images           | Not used as independent random observations |
| Provided PlantVillage CV folds    | Not used because of duplicate leakage       |

This strategy prioritizes reproducibility and evaluation integrity
over simply maximizing the amount of training data.
