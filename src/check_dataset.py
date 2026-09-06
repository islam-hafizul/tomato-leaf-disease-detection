from pathlib import Path
from collections import Counter
from PIL import Image

DATASET_DIR = Path("data/raw/mendeley")

# Common image extensions we expect
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
}

total_files = 0
image_files = 0
non_image_files = []

extension_counts = Counter()
format_counts = Counter()
size_counts = Counter()

corrupted_images = []


for file_path in sorted(file for file in DATASET_DIR.rglob("*") if file.is_file()):

    total_files += 1

    # Count the actual filename extension
    extension = file_path.suffix.lower()

    extension_counts[extension] += 1

    # Skip files that don't look like images
    if extension not in IMAGE_EXTENSIONS:
        non_image_files.append(file_path)
        continue

    image_files += 1

    try:
        # First check whether the image is valid
        with Image.open(file_path) as image:
            image.verify()

        # Reopen after verify()
        with Image.open(file_path) as image:
            format_counts[image.format] += 1
            size_counts[image.size] += 1

    except Exception as error:
        corrupted_images.append(
            (file_path, str(error))
        )


print("=" * 60)
print("DATASET HEALTH CHECK")
print("=" * 60)

print(f"\nTotal files: {total_files}")
print(f"Image files: {image_files}")
print(f"Non-image files: {len(non_image_files)}")

print("\nFilename extensions:")
for extension, count in extension_counts.most_common():
    extension_name = extension if extension else "[no extension]"
    print(f"  {extension_name}: {count}")

print("\nActual image formats detected by Pillow:")
for image_format, count in format_counts.most_common():
    print(f"  {image_format}: {count}")

print("\nMost common image sizes:")
for size, count in size_counts.most_common(10):
    print(f"  {size}: {count}")

print("\nCorrupted images:")
print(f"  {len(corrupted_images)}")

if corrupted_images:
    for path, error in corrupted_images:
        print(f"  {path}")
        print(f"    {error}")

print("\nNon-image files:")

if non_image_files:
    for path in non_image_files:
        print(f"  {path}")
else:
    print("  None")