from pathlib import Path
from collections import Counter
from PIL import Image


# ============================================================
# Configuration
# ============================================================


DATASET_ROOT = Path("data/raw/Dataset of Tomato Leaves")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Helper functions
# ============================================================

def get_image_files(directory):
    """Return all image files recursively inside a directory."""
    return [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def count_images_by_class(directory):
    """Count images directly inside each class directory."""
    results = {}

    for class_dir in sorted(directory.iterdir()):
        if not class_dir.is_dir():
            continue

        images = get_image_files(class_dir)
        results[class_dir.name] = len(images)

    return results


def print_class_counts(title, directory):
    print(f"\n{title}")
    print("-" * 60)

    counts = count_images_by_class(directory)

    total = 0

    for class_name, count in counts.items():
        print(f"{class_name:<45} {count:>6}")
        total += count

    print("-" * 60)
    print(f"{'TOTAL':<45} {total:>6}")

    return counts


def inspect_images(directory):
    """Inspect image formats, sizes, and corrupted images."""

    image_files = get_image_files(directory)

    extensions = Counter(
        path.suffix.lower()
        for path in image_files
    )

    formats = Counter()
    sizes = Counter()
    corrupted = []

    for path in image_files:
        try:
            with Image.open(path) as image:
                formats[image.format] += 1
                sizes[image.size] += 1

                # Force Pillow to actually read the image.
                image.verify()

        except Exception:
            corrupted.append(path)

    return image_files, extensions, formats, sizes, corrupted


def print_image_audit(title, directory):
    print(f"\n{title}")
    print("=" * 60)

    (
        image_files,
        extensions,
        formats,
        sizes,
        corrupted,
    ) = inspect_images(directory)

    print(f"Total images: {len(image_files)}")

    print("\nFilename extensions:")
    for extension, count in sorted(extensions.items()):
        print(f"  {extension:<10} {count}")

    print("\nActual image formats:")
    for image_format, count in formats.most_common():
        print(f"  {image_format:<10} {count}")

    print("\nMost common image sizes:")
    for size, count in sizes.most_common(10):
        print(f"  {str(size):<15} {count}")

    print(f"\nCorrupted images: {len(corrupted)}")

    if corrupted:
        print("\nCorrupted files:")
        for path in corrupted[:20]:
            print(f"  {path}")


# ============================================================
# Main audit
# ============================================================

def main():

    print("=" * 60)
    print("DATASET OF TOMATO LEAVES AUDIT")
    print("=" * 60)

    # --------------------------------------------------------
    # PlantVillage
    # --------------------------------------------------------

    plantvillage = DATASET_ROOT / "plantvillage"

    print("\n\nPLANTVILLAGE")
    print("=" * 60)

    # Preprocessed data
    pv_preprocessed = plantvillage / "Preprocessed data"

    print_class_counts(
        "Preprocessed data",
        pv_preprocessed
    )

    print_image_audit(
        "Preprocessed data image audit",
        pv_preprocessed
    )

    # --------------------------------------------------------
    # PlantVillage 5-fold cross-validation
    # --------------------------------------------------------

    cv_root = plantvillage / "5 cross-validation"

    print("\n\n5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    for fold in sorted(cv_root.iterdir()):

        if not fold.is_dir():
            continue

        train_dir = fold / "Train"
        test_dir = fold / "Test"

        train_images = get_image_files(train_dir)
        test_images = get_image_files(test_dir)

        print(f"\n{fold.name}")
        print("-" * 60)
        print(f"Train images: {len(train_images)}")
        print(f"Test images:  {len(test_images)}")
        print(f"Total:        {len(train_images) + len(test_images)}")

    # --------------------------------------------------------
    # Taiwan
    # --------------------------------------------------------

    taiwan = DATASET_ROOT / "taiwan"

    print("\n\nTAIWAN")
    print("=" * 60)

    # Taiwan preprocessed
    taiwan_preprocessed = taiwan / "Preprocessed data"

    print_class_counts(
        "Preprocessed data - Train",
        taiwan_preprocessed / "Train"
    )

    print_class_counts(
        "Preprocessed data - Test",
        taiwan_preprocessed / "Test"
    )

    print_image_audit(
        "Taiwan preprocessed image audit",
        taiwan_preprocessed
    )

    # --------------------------------------------------------
    # Taiwan augmented data
    # --------------------------------------------------------

    taiwan_augmented = taiwan / "data augmentation"

    print_class_counts(
        "Augmented data - Train",
        taiwan_augmented / "Train"
    )

    print_class_counts(
        "Augmented data - Test",
        taiwan_augmented / "Test"
    )

    print_image_audit(
        "Taiwan augmented image audit",
        taiwan_augmented
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()