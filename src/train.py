# Model training
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import create_model


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 10

BATCH_SIZE = 16
NUM_EPOCHS = 10

LEARNING_RATE = 0.0001

NUM_WORKERS = 2

FOLD = 0

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TRAINING
# =============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one complete epoch.
    """

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Move data to GPU / CPU
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # ---------------------------------------------------------
        # Forward pass
        # ---------------------------------------------------------

        outputs = model(images)

        loss = criterion(outputs, labels)

        # ---------------------------------------------------------
        # Backward pass
        # ---------------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ---------------------------------------------------------
        # Statistics
        # ---------------------------------------------------------

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# =============================================================================
# VALIDATION
# =============================================================================

def validate(model, validation_loader, criterion):
    """
    Evaluate the model on the validation fold.
    """

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    validation_loss = running_loss / total
    validation_accuracy = correct / total

    return validation_loss, validation_accuracy


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("TOMATO LEAF DISEASE DETECTION - TRAINING")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")
    print(f"Fold: {FOLD}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, validation_loader = create_dataloaders(
        fold=FOLD,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print("\nCreating model...")

    model = create_model(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # -------------------------------------------------------------------------
    # Loss function
    # -------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    best_validation_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):

        print("\n" + "-" * 80)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )
        print("-" * 80)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        validation_loss, validation_accuracy = validate(
            model,
            validation_loader,
            criterion,
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

        # ---------------------------------------------------------------------
        # Save best model
        # ---------------------------------------------------------------------

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            model_path = MODEL_DIR / f"resnet18_fold{FOLD}_best.pth"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "fold": FOLD,
                    "epoch": epoch + 1,
                    "validation_accuracy": validation_accuracy,
                },
                model_path,
            )

            print(
                f"✓ Best model saved: {model_path}"
            )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )


if __name__ == "__main__":
    main() 
