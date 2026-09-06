from pathlib import Path
import hashlib
import csv

from sklearn.model_selection import StratifiedKFold


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Mendeley"
    / "plantvillage"
    / "Preprocessed data"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_FILE = OUTPUT_DIR / "plantvillage_5fold.csv"

N_SPLITS = 5
RANDOM_STATE = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def collect_unique_images():
    """
    Collect PlantVillage images and remove exact duplicate files.

    We do NOT delete anything from the dataset.
    We simply keep one path for each unique SHA-256 hash.
    """

    print("=" * 80)
    print("COLLECTING PLANTVILLAGE IMAGES")
    print("=" * 80)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATASET_ROOT}"
        )

    image_files = sorted(
        path
        for path in DATASET_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Total image files found: {len(image_files)}")

    unique_images = {}
    duplicate_count = 0

    print("\nChecking for exact duplicates...")

    for index, image_path in enumerate(image_files, start=1):

        file_hash = calculate_sha256(image_path)

        if file_hash in unique_images:
            duplicate_count += 1
        else:
            unique_images[file_hash] = image_path

        if index % 1000 == 0 or index == len(image_files):
            print(f"Processed: {index}/{len(image_files)}")

    print("\nDuplicate files found:", duplicate_count)
    print("Unique images:", len(unique_images))

    return list(unique_images.values())


def create_folds(image_paths):
    """Create stratified 5-fold assignments."""

    labels = [path.parent.name for path in image_paths]

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_assignments = {}

    for fold_number, (_, validation_indices) in enumerate(
        splitter.split(image_paths, labels)
    ):
        for index in validation_indices:
            fold_assignments[index] = fold_number

    return fold_assignments


def save_split_manifest(image_paths, fold_assignments):
    """Save image paths, labels and fold assignments to CSV."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "path",
            "class_name",
            "fold",
        ])

        for index, image_path in enumerate(image_paths):

            relative_path = image_path.relative_to(PROJECT_ROOT)

            writer.writerow([
                relative_path.as_posix(),
                image_path.parent.name,
                fold_assignments[index],
            ])

    print(f"\nSplit manifest saved to:")
    print(OUTPUT_FILE)


def print_fold_statistics(image_paths, fold_assignments):
    """Print class distribution for every fold."""

    print("\n")
    print("=" * 80)
    print("FOLD STATISTICS")
    print("=" * 80)

    for fold in range(N_SPLITS):

        fold_images = [
            image_paths[index]
            for index, assigned_fold in fold_assignments.items()
            if assigned_fold == fold
        ]

        class_counts = {}

        for image_path in fold_images:
            class_name = image_path.parent.name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"\nFold {fold}: {len(fold_images)} images")

        for class_name in sorted(class_counts):
            print(f"  {class_name}: {class_counts[class_name]}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("PLANTVILLAGE 5-FOLD CROSS-VALIDATION SPLIT")
    print("=" * 80)

    print(f"\nDataset:")
    print(DATASET_ROOT)

    print(f"\nConfiguration:")
    print(f"  Number of folds: {N_SPLITS}")
    print(f"  Random seed:     {RANDOM_STATE}")

    # -------------------------------------------------------------------------
    # Step 1: Collect unique images
    # -------------------------------------------------------------------------

    image_paths = collect_unique_images()

    # -------------------------------------------------------------------------
    # Step 2: Create stratified folds
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("CREATING STRATIFIED FOLDS")
    print("=" * 80)

    fold_assignments = create_folds(image_paths)

    # -------------------------------------------------------------------------
    # Step 3: Print statistics
    # -------------------------------------------------------------------------

    print_fold_statistics(
        image_paths,
        fold_assignments,
    )

    # -------------------------------------------------------------------------
    # Step 4: Save CSV manifest
    # -------------------------------------------------------------------------

    save_split_manifest(
        image_paths,
        fold_assignments,
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"\nUnique images used: {len(image_paths)}")
    print(f"Number of folds:    {N_SPLITS}")
    print(f"Random seed:        {RANDOM_STATE}")

    print("\nHow the folds work:")
    print("  Fold 0 → validation, Folds 1-4 → training")
    print("  Fold 1 → validation, Folds 0,2-4 → training")
    print("  Fold 2 → validation, Folds 0,1,3,4 → training")
    print("  Fold 3 → validation, Folds 0-2,4 → training")
    print("  Fold 4 → validation, Folds 0-3 → training")

    print("\nNo images were copied or deleted.")


if __name__ == "__main__":
    main()