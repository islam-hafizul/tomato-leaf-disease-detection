# Dataset leakage/duplicate checking
from pathlib import Path
from collections import defaultdict
import hashlib


# ============================================================
# Configuration
# ============================================================


DATASET_ROOT = Path("data/raw/Dataset of Tomato Leaves")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================================================
# Helpers
# ============================================================

def get_images(directory):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path, chunk_size=1024 * 1024):
    """Calculate SHA-256 hash of a file."""

    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def build_hash_map(directory):
    """Map SHA-256 hashes to files."""

    hash_map = defaultdict(list)

    images = get_images(directory)

    print(f"Scanning: {directory}")
    print(f"Images: {len(images)}")

    for index, path in enumerate(images, start=1):

        file_hash = sha256_file(path)
        hash_map[file_hash].append(path)

        if index % 1000 == 0:
            print(f"  Processed {index}/{len(images)}")

    return hash_map


def find_duplicates(hash_map):
    """Return only hashes associated with multiple files."""

    return {
        file_hash: paths
        for file_hash, paths in hash_map.items()
        if len(paths) > 1
    }


def print_duplicates(title, duplicates, max_groups=20):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if not duplicates:
        print("No exact duplicates found.")
        return

    print(f"Duplicate groups: {len(duplicates)}")

    for index, (file_hash, paths) in enumerate(
        duplicates.items(), start=1
    ):

        if index > max_groups:
            print(
                f"\n... showing first {max_groups} groups only"
            )
            break

        print(f"\nGroup {index}")
        print(f"SHA-256: {file_hash}")

        for path in paths:
            print(f"  {path}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DATASET OF TOMATO LEAVES DUPLICATE / LEAKAGE AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. PlantVillage preprocessed data
    # --------------------------------------------------------

    pv_preprocessed = (
        DATASET_ROOT
        / "plantvillage"
        / "Preprocessed data"
    )

    pv_hashes = build_hash_map(pv_preprocessed)

    pv_duplicates = find_duplicates(pv_hashes)

    print_duplicates(
        "1. PlantVillage - Internal duplicates",
        pv_duplicates
    )

    # --------------------------------------------------------
    # 2. Check each CV fold for Train/Test leakage
    # --------------------------------------------------------

    cv_root = (
        DATASET_ROOT
        / "plantvillage"
        / "5 cross-validation"
    )

    print("\n" + "=" * 70)
    print("2. PLANTVILLAGE TRAIN / TEST LEAKAGE")
    print("=" * 70)

    for fold in sorted(cv_root.iterdir()):

        if not fold.is_dir():
            continue

        train_dir = fold / "Train"
        test_dir = fold / "Test"

        train_hashes = build_hash_map(train_dir)
        test_hashes = build_hash_map(test_dir)

        overlapping_hashes = (
            set(train_hashes.keys())
            & set(test_hashes.keys())
        )

        print(f"\n{fold.name}")
        print("-" * 70)
        print(f"Train unique hashes: {len(train_hashes)}")
        print(f"Test unique hashes:  {len(test_hashes)}")
        print(
            f"Train/Test overlapping hashes: "
            f"{len(overlapping_hashes)}"
        )

        if overlapping_hashes:

            print("\nWARNING: Potential train/test leakage!")

            for file_hash in list(overlapping_hashes)[:10]:

                print(f"\nSHA-256: {file_hash}")

                print("Train:")
                for path in train_hashes[file_hash]:
                    print(f"  {path}")

                print("Test:")
                for path in test_hashes[file_hash]:
                    print(f"  {path}")

        else:
            print("✓ No exact Train/Test overlap.")

    # --------------------------------------------------------
    # 3. PlantVillage vs Taiwan original data
    # --------------------------------------------------------

    taiwan_preprocessed = (
        DATASET_ROOT
        / "taiwan"
        / "Preprocessed data"
    )

    taiwan_hashes = build_hash_map(taiwan_preprocessed)

    overlapping_hashes = (
        set(pv_hashes.keys())
        & set(taiwan_hashes.keys())
    )

    print("\n" + "=" * 70)
    print("3. PLANTVILLAGE vs TAIWAN")
    print("=" * 70)

    print(
        f"PlantVillage unique hashes: {len(pv_hashes)}"
    )

    print(
        f"Taiwan unique hashes: {len(taiwan_hashes)}"
    )

    print(
        f"Exact overlapping images: "
        f"{len(overlapping_hashes)}"
    )

    if overlapping_hashes:

        print("\nOverlapping images:")

        for file_hash in list(overlapping_hashes)[:20]:

            print(f"\nSHA-256: {file_hash}")

            print("PlantVillage:")
            for path in pv_hashes[file_hash]:
                print(f"  {path}")

            print("Taiwan:")
            for path in taiwan_hashes[file_hash]:
                print(f"  {path}")

    # --------------------------------------------------------
    # 4. Taiwan original vs augmented
    # --------------------------------------------------------

    taiwan_augmented = (
        DATASET_ROOT
        / "taiwan"
        / "data augmentation"
    )

    augmented_hashes = build_hash_map(taiwan_augmented)

    original_augmented_overlap = (
        set(taiwan_hashes.keys())
        & set(augmented_hashes.keys())
    )

    print("\n" + "=" * 70)
    print("4. TAIWAN ORIGINAL vs AUGMENTED")
    print("=" * 70)

    print(
        f"Original images: {len(taiwan_hashes)}"
    )

    print(
        f"Augmented images: {len(augmented_hashes)}"
    )

    print(
        f"Exact identical files: "
        f"{len(original_augmented_overlap)}"
    )

    print("\nNote:")
    print(
        "Augmented images may intentionally differ from "
        "their originals, so a zero exact-hash overlap "
        "does NOT mean there is no relationship."
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATE / LEAKAGE AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()