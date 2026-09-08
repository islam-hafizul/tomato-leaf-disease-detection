# Evaluation and metrics
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from dataset import (
    CLASS_NAMES,
    create_dataloaders,
)
from model import create_model


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FOLD = 0

BATCH_SIZE = 16
NUM_WORKERS = 2

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / f"resnet18_fold{FOLD}_best.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = (
    RESULTS_DIR
    / f"fold{FOLD}_predictions.csv"
)


# =============================================================================
# LOAD MODEL
# =============================================================================

def load_trained_model():
    """Load the trained ResNet18 checkpoint."""

    model = create_model(
        num_classes=len(CLASS_NAMES)
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model, checkpoint


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, validation_loader):
    """
    Run the model on the validation set and collect predictions.
    """

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            # Convert logits into probabilities
            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    return (
        all_labels,
        all_predictions,
        all_probabilities,
    )


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    validation_loader,
    labels,
    predictions,
    probabilities,
):
    """
    Save predictions and confidence scores to CSV.
    """

    dataframe = validation_loader.dataset.dataframe.copy()

    dataframe["true_label"] = labels
    dataframe["predicted_label"] = predictions

    dataframe["true_class"] = [
        CLASS_NAMES[label]
        for label in labels
    ]

    dataframe["predicted_class"] = [
        CLASS_NAMES[prediction]
        for prediction in predictions
    ]

    dataframe["correct"] = (
        dataframe["true_label"]
        == dataframe["predicted_label"]
    )

    # Highest probability assigned by the model
    dataframe["confidence"] = [
        max(probability)
        for probability in probabilities
    ]

    dataframe.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"\nPredictions saved to:\n{PREDICTIONS_FILE}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")
    print(f"Fold: {FOLD}")
    print(f"Model: {MODEL_FILE}")

    # -------------------------------------------------------------------------
    # Check model
    # -------------------------------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_FILE}"
        )

    # -------------------------------------------------------------------------
    # Load validation data
    # -------------------------------------------------------------------------

    print("\nLoading validation dataset...")

    _, validation_loader = create_dataloaders(
        fold=FOLD,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(
        f"Validation images: "
        f"{len(validation_loader.dataset)}"
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    print("\nLoading trained model...")

    model, checkpoint = load_trained_model()

    print(
        f"Best checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation accuracy: "
        f"{checkpoint['validation_accuracy']:.4f}"
    )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    print("\nRunning evaluation...")

    labels, predictions, probabilities = evaluate(
        model,
        validation_loader,
    )

    # -------------------------------------------------------------------------
    # Accuracy
    # -------------------------------------------------------------------------

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    print("\n")
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print(
        f"\nAccuracy: {accuracy:.4f}"
        f" ({accuracy * 100:.2f}%)"
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    print(report)

    # -------------------------------------------------------------------------
    # Confusion matrix
    # -------------------------------------------------------------------------

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print("\nRows = actual class")
    print("Columns = predicted class\n")

    print(matrix)

    # -------------------------------------------------------------------------
    # Save confusion matrix
    # -------------------------------------------------------------------------

    confusion_matrix_file = (
        RESULTS_DIR
        / f"fold{FOLD}_confusion_matrix.csv"
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    confusion_dataframe.to_csv(
        confusion_matrix_file
    )

    print(
        f"\nConfusion matrix saved to:"
        f"\n{confusion_matrix_file}"
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    save_predictions(
        validation_loader,
        labels,
        predictions,
        probabilities,
    )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()