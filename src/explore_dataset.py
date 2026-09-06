from pathlib import Path

DATASET_DIR = Path("data/raw/mendeley")
if not DATASET_DIR.exists():
    DATASET_DIR = Path("data/raw/Mendeley")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

class_dirs = sorted(
    directory
    for directory in DATASET_DIR.rglob("*")
    if directory.is_dir()
    and any(
        file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        for file in directory.iterdir()
    )
)

print(f"Number of classes: {len(class_dirs)}")
print()

for class_dir in class_dirs:
    images = [
        file
        for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]
    print(f"{class_dir.relative_to(DATASET_DIR)}: {len(images)} images")