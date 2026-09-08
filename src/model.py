# ResNet18 model
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_CLASSES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# MODEL
# =============================================================================

def create_model(num_classes=NUM_CLASSES):
    """
    Create a ResNet18 model with ImageNet pretrained weights.

    The final classification layer is replaced so that the model
    predicts our 10 tomato leaf classes.
    """

    # Load ResNet18 with pretrained ImageNet weights
    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )

    # Get number of inputs to the original final layer
    num_features = model.fc.in_features

    # Replace the original ImageNet classifier
    model.fc = nn.Linear(
        num_features,
        num_classes
    )

    return model


# =============================================================================
# MODEL TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("TESTING RESNET18 MODEL")
    print("=" * 80)

    print(f"\nDevice: {DEVICE}")

    # Create model
    model = create_model()

    # Move model to GPU / CPU
    model = model.to(DEVICE)

    print("\nModel created successfully.")
    print(f"Number of classes: {NUM_CLASSES}")

    print("\nFinal classification layer:")
    print(model.fc)

    # -------------------------------------------------------------------------
    # Test forward pass
    # -------------------------------------------------------------------------

    # Create a fake batch of 16 RGB images of size 224x224
    dummy_input = torch.randn(
        16,
        3,
        224,
        224
    ).to(DEVICE)

    # Disable gradient calculation because this is only a test
    with torch.no_grad():
        output = model(dummy_input)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nExpected output shape:")
    print("(16, 10)")

    print("\nModel device:")
    print(next(model.parameters()).device)

    print("\n" + "=" * 80)
    print("MODEL TEST COMPLETE")
    print("=" * 80)