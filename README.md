# AI Crop Disease Detection

A deep learning project for classifying tomato leaf diseases using computer vision and PyTorch.

The project uses a pretrained ResNet18 model to classify tomato leaf images into 10 disease/health categories.

## Dataset

**Dataset:** Dataset of Tomato Leaves  
**Source:** Mendeley Data  
**DOI:** 10.17632/ngdgg79rzb.1

The dataset contains tomato leaf images from the PlantVillage dataset.
The PlantVillage portion used in this project contains:

- 14,531 images
- 10 classes
- JPEG images
- 227×227 resolution


> Dataset source: https://data.mendeley.com/datasets/ngdgg79rzb/1

## Classes

The model classifies tomato leaves into the following categories:

1. Bacterial Spot
2. Early Blight
3. Healthy
4. Late Blight
5. Leaf Mold
6. Septoria Leaf Spot
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Two-Spotted Spider Mite

## Methodology

The overall workflow is:

```text
Dataset
   ↓
Dataset Inspection
   ↓
Custom Train/Validation Split
   ↓
Image Preprocessing
   ↓
ResNet18 Transfer Learning
   ↓
Model Training
   ↓
Validation
   ↓
Evaluation
````

A custom **80/20 train-validation split** was created from 'Preprocessed data' folder instead of directly using the cross-validation folders supplied with the dataset.

During dataset inspection, exact duplicate images were found across some of the supplied folds. Therefore, the supplied folds were not used for the main experiment.

More details are available in:

* [Methodology](docs/methodology.md)
* [Experiments](docs/experiments.md)

## Model

The project uses **ResNet18** with pretrained ImageNet weights.

The final classification layer was replaced to support the 10 tomato leaf classes.

### Training Configuration

| Parameter         | Value           |
| ----------------- | --------------- |
| Model             | ResNet18        |
| Pretrained        | Yes             |
| Input Size        | 224 × 224       |
| Batch Size        | 16              |
| Learning Rate     | 0.0001          |
| Epochs            | 10              |
| Device            | NVIDIA CUDA GPU |
| Train Images      | 11,616          |
| Validation Images | 2,905           |

## Results

The best validation performance was achieved at **epoch 8**.

| Metric              |      Score |
| ------------------- | ---------: |
| Validation Accuracy | **99.41%** |
| Macro Precision     |     98.84% |
| Macro Recall        |     99.35% |
| Macro F1            |     99.09% |
| Weighted F1         |     99.42% |

### Class-wise Performance

| Class                         | Precision |  Recall |     F1 |
| ----------------------------- | --------: | ------: | -----: |
| Bacterial Spot                |    99.71% |  99.71% | 99.71% |
| Early Blight                  |    96.93% |  98.75% | 97.83% |
| Healthy                       |    99.22% | 100.00% | 99.61% |
| Late Blight                   |    99.67% |  98.68% | 99.17% |
| Leaf Mold                     |    98.70% |  99.35% | 99.02% |
| Septoria Leaf Spot            |    99.29% |  98.94% | 99.12% |
| Target Spot                   |   100.00% |  98.67% | 99.33% |
| Tomato Yellow Leaf Curl Virus |   100.00% |  99.77% | 99.88% |
| Tomato Mosaic Virus           |    95.24% | 100.00% | 97.56% |
| Two-Spotted Spider Mite       |    99.63% |  99.63% | 99.63% |

## Training Progress

| Epoch | Train Accuracy | Validation Accuracy |
| ----: | -------------: | ------------------: |
|     1 |         91.10% |              96.18% |
|     2 |         96.82% |              98.11% |
|     3 |         97.99% |              97.11% |
|     4 |         98.28% |              98.73% |
|     5 |         98.70% |              98.21% |
|     6 |         98.71% |              99.31% |
|     7 |         98.91% |              98.80% |
| **8** |     **99.12%** |          **99.41%** |
|     9 |         98.82% |              98.93% |
|    10 |         99.13% |              99.04% |

## Project Structure

```text
ai-crop-disease-detection/
│
├── data/
│   └── ...
│
├── models/
│   └── ...
│
├── results/
│   ├── fold0_confusion_matrix.csv
│   └── fold0_predictions.csv
│
├── src/
│   ├── create_splits.py
│   ├── dataset.py
│   ├── train.py
│   └── evaluate.py
│
├── docs/
│   ├── methodology.md
│   └── experiments.md
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/islam-hafizul/ai-crop-disease-detection.git
cd ai-crop-disease-detection
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Check the dataset

```bash
python src/check_duplicates.py
```

### 2. Create the train/validation split

```bash
python src/create_splits.py
```

### 3. Train the model

```bash
python src/train.py
```

### 4. Evaluate the model

```bash
python src/evaluate.py
```

The evaluation generates prediction and confusion-matrix data inside the `results/` directory.

## Hardware

Training was performed using an NVIDIA GeForce GTX 960M with 4 GB VRAM.

CUDA acceleration was used through PyTorch.

GPU utilization reached approximately **98–100% during training**, which indicates that the GPU was being effectively utilized.

## Limitations

The reported results should not be interpreted as proof of real-world field performance.

The current experiment uses images derived from the PlantVillage dataset, which are generally more controlled than images captured in real agricultural environments.

The project currently evaluates performance using a single custom train-validation split. Further testing on an independent external dataset would provide a stronger measure of generalization.

## Future Work

Possible improvements include:

* Evaluate the model on an independent external dataset
* Experiment with additional data augmentation
* Compare ResNet18 with other CNN architectures
* Perform hyperparameter tuning
* Analyze misclassified images
* Add Grad-CAM or other explainability techniques
* Build a simple web interface for disease prediction
* Deploy the trained model as an API

## License

This project is intended for educational and research purposes.

The dataset is provided by the original dataset authors under their respective terms.

