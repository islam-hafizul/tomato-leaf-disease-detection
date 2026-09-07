from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_FILE = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "plantvillage_5fold.csv"
)


# =============================================================================
# CLASS MAPPING
# =============================================================================

CLASS_NAMES = [
    "Bacterial_spot227",
    "Early_blight227",
    "healthy227",
    "Late_blight227",
    "Leaf_Mold227",
    "Septoria_leaf_spot227",
    "Target_Spot227",
    "Tomato_Yellow_Leaf_Curl_Virus227",
    "Tomato_mosaic_virus227",
    "Two-spotted_spider_mite227",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# =============================================================================
# DATASET
# =============================================================================

class TomatoLeafDataset(Dataset):
    """
    PyTorch Dataset for the PlantVillage tomato leaf dataset.
    """

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        """Return the number of images in the dataset."""

        return len(self.dataframe)

    def __getitem__(self, index):
        """
        Load one image and its corresponding label.
        """

        row = self.dataframe.iloc[index]

        image_path = PROJECT_ROOT / row["path"]
        class_name = row["class_name"]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Convert class name to integer label
        label = CLASS_TO_INDEX[class_name]

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        return image, label


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_train_transforms():
    """
    Transformations used during training.

    Data augmentation is applied here so that the model
    sees slightly different versions of training images.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(10),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_validation_transforms():
    """
    Transformations used for validation.

    No random augmentation is applied here.
    """

    return transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# =============================================================================
# DATAFRAME / FOLD LOADING
# =============================================================================

def load_split_dataframe():
    """Load the cross-validation manifest."""

    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{SPLIT_FILE}"
        )

    dataframe = pd.read_csv(SPLIT_FILE)

    return dataframe


def create_dataloaders(fold, batch_size=16, num_workers=2):
    """
    Create training and validation DataLoaders for one fold.

    Example:
        fold=0

        Training:
            folds 1, 2, 3, 4

        Validation:
            fold 0
    """

    dataframe = load_split_dataframe()

    train_dataframe = dataframe[dataframe["fold"] != fold]
    validation_dataframe = dataframe[dataframe["fold"] == fold]

    train_dataset = TomatoLeafDataset(
        train_dataframe,
        transform=get_train_transforms(),
    )

    validation_dataset = TomatoLeafDataset(
        validation_dataframe,
        transform=get_validation_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, validation_loader


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("TESTING PYTORCH DATASET")
    print("=" * 80)

    # Use fold 0 for this test
    train_loader, validation_loader = create_dataloaders(
        fold=0,
        batch_size=16,
    )

    print(f"\nTraining images:   {len(train_loader.dataset)}")
    print(f"Validation images: {len(validation_loader.dataset)}")

    # Get one batch
    images, labels = next(iter(train_loader))

    print("\nFirst training batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    print("\nLabels:")
    print(labels)

    print("\nImage value range:")
    print(f"Minimum: {images.min().item():.4f}")
    print(f"Maximum: {images.max().item():.4f}")

    print("\nNumber of classes:")
    print(len(CLASS_NAMES))

    print("\nClass mapping:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {index}: {class_name}")

    print("\n" + "=" * 80)
    print("DATASET TEST COMPLETE")
    print("=" * 80)
