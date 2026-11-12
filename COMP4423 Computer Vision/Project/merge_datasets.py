"""
Merge two Roboflow YOLOv8 datasets:
1) pet.yolov8.zip
2) household_dataset.zip

What it does:
- Keeps pet classes as:
  0 cat
  1 dog
- Remaps household classes by +2:
  2 backpack
  3 book
  ...
- Creates a new YOLOv8 dataset:
  merged_pet_household_yolov8/
    train/images, train/labels
    valid/images, valid/labels
    test/images, test/labels
    data.yaml

How to run:
  python merge_datasets.py

Put this script in the same folder as:
  pet.yolov8.zip
  household_dataset.zip
"""

from pathlib import Path
import zipfile
import shutil
import random
import json
from collections import Counter

PET_ZIP = Path("pet.yolov8.zip")
HOUSE_ZIP = Path("household_dataset.zip")

OUT_DIR = Path("merged_pet_household_yolov8")
OUT_ZIP = Path("merged_pet_household_yolov8.zip")

# Final shared class list.
# Household IDs will be shifted by +2.
CLASSES = [
    "cat", "dog",
    "backpack", "book", "box", "chair", "fan", "fork",
    "glass", "plate", "spoon", "table", "vase"
]

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_names(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return z.namelist()


def find_root_prefix(names):
    """
    Roboflow zips sometimes have:
      train/images/...
    or:
      household_dataset/train/images/...
    This detects the prefix before train/.
    """
    for n in names:
        if "train/images/" in n:
            return n.split("train/images/")[0]
    return ""


def collect_image_label_pairs(zip_path, source_name, class_offset):
    names = get_names(zip_path)
    name_set = set(names)
    root = find_root_prefix(names)

    pairs = []
    for n in names:
        if not n.startswith(root + "train/images/"):
            continue
        if Path(n).suffix.lower() not in IMAGE_EXTS:
            continue

        stem = Path(n).stem
        label_path = root + f"train/labels/{stem}.txt"

        # Skip images without useful label files
        if label_path not in name_set:
            continue

        with zipfile.ZipFile(zip_path) as z:
            try:
                label_text = z.read(label_path).decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

        if not label_text:
            continue

        pairs.append({
            "zip_path": zip_path,
            "image_path": n,
            "label_path": label_path,
            "source": source_name,
            "offset": class_offset,
        })

    return pairs


def remap_label_text(label_text, offset):
    new_lines = []
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        try:
            old_id = int(float(parts[0]))
        except ValueError:
            continue

        new_id = old_id + offset
        if new_id < 0 or new_id >= len(CLASSES):
            continue

        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    return "\n".join(new_lines).strip()


def main():
    if not PET_ZIP.exists():
        raise FileNotFoundError(f"Cannot find {PET_ZIP}")
    if not HOUSE_ZIP.exists():
        raise FileNotFoundError(f"Cannot find {HOUSE_ZIP}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    for split in ["train", "valid", "test"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    print("Collecting pet dataset pairs...")
    pet_pairs = collect_image_label_pairs(PET_ZIP, "pet", 0)

    print("Collecting household dataset pairs...")
    household_pairs = collect_image_label_pairs(HOUSE_ZIP, "household", 2)

    all_pairs = pet_pairs + household_pairs
    print(f"Pet pairs: {len(pet_pairs)}")
    print(f"Household pairs: {len(household_pairs)}")
    print(f"Total pairs before split: {len(all_pairs)}")

    random.Random(RANDOM_SEED).shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    split_pairs = {
        "train": all_pairs[:n_train],
        "valid": all_pairs[n_train:n_train+n_valid],
        "test": all_pairs[n_train+n_valid:],
    }

    image_counts = Counter()
    instance_counts = Counter()
    source_counts = Counter()

    zip_cache = {
        str(PET_ZIP): zipfile.ZipFile(PET_ZIP),
        str(HOUSE_ZIP): zipfile.ZipFile(HOUSE_ZIP),
    }

    try:
        for split, pairs in split_pairs.items():
            for i, p in enumerate(pairs):
                z = zip_cache[str(p["zip_path"])]

                image_bytes = z.read(p["image_path"])
                label_text = z.read(p["label_path"]).decode("utf-8", errors="ignore")
                remapped = remap_label_text(label_text, p["offset"])

                if not remapped:
                    continue

                prefix = "pet_" if p["source"] == "pet" else "house_"
                image_name = prefix + Path(p["image_path"]).name
                label_name = prefix + Path(p["image_path"]).with_suffix(".txt").name

                (OUT_DIR / split / "images" / image_name).write_bytes(image_bytes)
                (OUT_DIR / split / "labels" / label_name).write_text(remapped + "\n", encoding="utf-8")

                image_counts[split] += 1
                source_counts[f"{split}_{p['source']}"] += 1

                for line in remapped.splitlines():
                    cls_id = int(line.split()[0])
                    instance_counts[CLASSES[cls_id]] += 1
    finally:
        for z in zip_cache.values():
            z.close()

    # Write data.yaml manually to avoid requiring PyYAML
    yaml_text = "path: .\n"
    yaml_text += "train: train/images\n"
    yaml_text += "val: valid/images\n"
    yaml_text += "test: test/images\n"
    yaml_text += f"nc: {len(CLASSES)}\n"
    yaml_text += "names:\n"
    for i, name in enumerate(CLASSES):
        yaml_text += f"  {i}: {name}\n"

    (OUT_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")

    summary = {
        "random_seed": RANDOM_SEED,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "valid": VALID_RATIO,
            "test": TEST_RATIO,
        },
        "classes": {i: c for i, c in enumerate(CLASSES)},
        "image_counts": dict(image_counts),
        "source_counts": dict(source_counts),
        "instance_counts": dict(instance_counts),
        "notes": [
            "Pet class IDs were kept as 0=cat and 1=dog.",
            "Household class IDs were remapped by +2.",
            "Images without useful labels were skipped.",
            "File names were prefixed with pet_ or house_ to avoid name collisions.",
            "A new 80/10/10 split was created using random seed 42."
        ],
    }
    (OUT_DIR / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = "Merged Pet + Household YOLOv8 Dataset\n\n"
    readme += "Training command:\n"
    readme += "  yolo detect train model=yolov8n.pt data=merged_pet_household_yolov8/data.yaml epochs=50 imgsz=640 batch=16\n\n"
    readme += "Classes:\n"
    for i, name in enumerate(CLASSES):
        readme += f"  {i}: {name}\n"
    readme += "\nImage counts:\n"
    for split in ["train", "valid", "test"]:
        readme += f"  {split}: {image_counts[split]}\n"
    (OUT_DIR / "README_MERGED_DATASET.txt").write_text(readme, encoding="utf-8")

    print("Creating zip file...")
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for file in OUT_DIR.rglob("*"):
            if file.is_file():
                zout.write(file, file.relative_to(OUT_DIR.parent))

    print("Done.")
    print(f"Output folder: {OUT_DIR}")
    print(f"Output zip: {OUT_ZIP}")
    print(f"Image counts: {dict(image_counts)}")
    print("Final classes:")
    for i, name in enumerate(CLASSES):
        print(f"  {i}: {name}")


if __name__ == "__main__":
    main()