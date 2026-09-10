# Experiments

This document records the experiments performed for the tomato leaf disease
classification project.

The goal of the experiments is to evaluate a pretrained ResNet-18 model on
the PlantVillage portion of the **Dataset of Tomato Leaves**.

## Experiment 1 — ResNet18 Baseline

- **Status:** Completed
- **Fold:** 0
- **Model:** ResNet18 (ImageNet pretrained)
- **Dataset:** Dataset of Tomato Leaves — PlantVillage subset
- **Training images:** 11,616
- **Validation images:** 2,905
- **Batch size:** 16
- **Epochs:** 10
- **Learning rate:** 0.0001
- **Device:** NVIDIA GTX 960M (CUDA)

### Results

| Metric                   |     Result |
| ------------------------ | ---------: |
| Best validation accuracy | **99.41%** |
| Macro precision          |     98.84% |
| Macro recall             |     99.35% |
| Macro F1                 | **99.09%** |
| Weighted F1              |     99.42% |
| Best epoch               |          8 |

The best checkpoint was obtained at epoch 8. Training accuracy reached
99.13% by epoch 10, while validation accuracy remained around 99%.

### Observations

* ResNet18 converged quickly with the pretrained weights.
* The model achieved strong performance on the PlantVillage validation
  split.
* The `Tomato_mosaic_virus` class had the lowest F1 score (97.56%),
  although its validation support was relatively small.
* The results are promising but represent **only one fold** and should
  not be considered the final model performance.

### Next Steps

1. Calculate mean and standard deviation across folds.
2. Compare per-class performance across folds.
3. Evaluate the trained model on the Taiwan dataset.
4. Perform error analysis and Grad-CAM visualization.
