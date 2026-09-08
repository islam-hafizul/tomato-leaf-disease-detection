from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
# import random

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

DATASET_DIR = Path("data/raw/Dataset of Tomato Leaves")
plantvillage_dir = DATASET_DIR / "plantvillage" / "Preprocessed data"
taiwan_dir = DATASET_DIR / "taiwan" / "Preprocessed data" / "Test"

class_dirs = [
    (directory, "PlantVillage")
    for directory in sorted(plantvillage_dir.iterdir())
    if directory.is_dir()
]
class_dirs.extend(
    (directory, "Taiwan")
    for directory in sorted(taiwan_dir.iterdir())
    if directory.is_dir()
)

fig, axes = plt.subplots(4, 4, figsize=(14, 14))

for ax, (class_dir, dataset_name) in zip(axes.flat, class_dirs):

    image_files = [
        file
        for file in class_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        ax.set_title(f"{class_dir.name}\nNO IMAGE", fontsize=8)
        ax.axis("off")
        continue

    image_path = image_files[0]
    # image_path = random.choice(image_files)

    with Image.open(image_path) as opened_image:
        image = opened_image.copy()

    ax.imshow(image)
    ax.set_title(f"{dataset_name}: {class_dir.name}", fontsize=8)
    ax.axis("off")

for ax in axes.flat[len(class_dirs):]:
    ax.axis("off")

plt.tight_layout()
plt.show()