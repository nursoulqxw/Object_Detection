from pathlib import Path
import csv
import random
import shutil
import yaml

from collections import Counter
import imagehash
from PIL import Image, ImageOps
import kagglehub


CLASS_NAMES = ["crop", "weed"]
HASH_THRESHOLD = 10  # max different bits to count two images as copies
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15  # test gets the rest (0.15)
OUTPUT_DIR = Path("data")

def get_raw_data_dir():
    path = kagglehub.dataset_download(
        "ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes"
    )
    return Path(path) / "agri_data" / "data"


def read_labels(label_path):
    boxes = []
    problems = []
    lines = label_path.read_text().splitlines()

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            continue  # empty line, YOLO ignores it

        if len(parts) != 5:
            problems.append(f"line {line_number}: expected 5 values, got {len(parts)}")
            continue

        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        if class_id not in range(len(CLASS_NAMES)):
            problems.append(f"line {line_number}: unknown class {class_id}")
            continue
        if not all(0 <= value <= 1 for value in (x, y, w, h)):
            problems.append(f"line {line_number}: value outside 0..1")
        if w <= 0 or h <= 0:
            problems.append(f"line {line_number}: box has zero size")

        boxes.append((class_id, x, y, w, h))

    return boxes, problems


def check_labels(data_dir):
    image_paths = sorted(data_dir.glob("*.jpeg"))
    print("Images found:", len(image_paths))

    object_counts = [0, 0]
    images_without_boxes = []
    problem_count = 0

    for image_path in image_paths:
        label_path = image_path.with_suffix(".txt")
        if not label_path.exists():
            print("Missing label file:", image_path.name)
            problem_count += 1
            continue

        boxes, problems = read_labels(label_path)
        for problem in problems:
            print(label_path.name, "-", problem)
        problem_count += len(problems)

        if not boxes:
            images_without_boxes.append(image_path.name)
        for box in boxes:
            object_counts[box[0]] += 1

    print("Problems found:", problem_count)
    print("Images with no boxes:", len(images_without_boxes))
    for name, count in zip(CLASS_NAMES, object_counts):
        print(f"Objects of class '{name}':", count)
    print("Total objects:", sum(object_counts))

def image_hashes(image_path):
    image = Image.open(image_path).convert("RGB")
    hashes = []
    for version in (image, ImageOps.mirror(image)):
        for angle in (0, 90, 180, 270):
            phash = imagehash.phash(version.rotate(angle))
            hashes.append(int(str(phash), 16))
    return hashes


def find_duplicate_groups(image_paths):
    print("Computing hashes...")
    hashes = [image_hashes(path) for path in image_paths]

    count = len(image_paths)
    parent = list(range(count))
    nearest_distance = [64] * count

    def find_root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    print("Comparing all pairs...")
    for i in range(count):
        for j in range(i + 1, count):
            distance = min((hashes[i][0] ^ h).bit_count() for h in hashes[j])
            nearest_distance[i] = min(nearest_distance[i], distance)
            nearest_distance[j] = min(nearest_distance[j], distance)
            if distance <= HASH_THRESHOLD:
                parent[find_root(i)] = find_root(j)

    print("\nDistance to the most similar other image:")
    distance_counts = Counter(nearest_distance)
    for distance in range(0, 21):
        print(f"  {distance:2d} bits: {distance_counts[distance]} images")
    print(f"  >20 bits: {sum(c for d, c in distance_counts.items() if d > 20)} images")

    groups = {}
    for i, path in enumerate(image_paths):
        groups.setdefault(find_root(i), []).append(path)
    groups = list(groups.values())

    print("\nNumber of groups:", len(groups))
    size_counts = Counter(len(group) for group in groups)
    for size in sorted(size_counts):
        print(f"  groups with {size} image(s): {size_counts[size]}")

    return groups

def split_groups(groups):
    random.Random(SEED).shuffle(groups)

    total_images = sum(len(group) for group in groups)
    train_limit = TRAIN_RATIO * total_images
    val_limit = (TRAIN_RATIO + VAL_RATIO) * total_images

    splits = {"train": [], "val": [], "test": []}
    placed = 0
    for group in groups:
        if placed < train_limit:
            split_name = "train"
        elif placed < val_limit:
            split_name = "val"
        else:
            split_name = "test"
        splits[split_name].append(group)
        placed += len(group)

    return splits


def save_split(splits):
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    rows = []
    for split_name, groups_in_split in splits.items():
        image_dir = OUTPUT_DIR / "images" / split_name
        label_dir = OUTPUT_DIR / "labels" / split_name
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)

        for group in groups_in_split:
            group_id = group[0].stem
            for image_path in group:
                label_path = image_path.with_suffix(".txt")
                shutil.copy(image_path, image_dir / image_path.name)
                shutil.copy(label_path, label_dir / label_path.name)
                rows.append([image_path.name, group_id, split_name])

    with open("res/split.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["image", "group", "split"])
        writer.writerows(rows)

    config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    with open(OUTPUT_DIR / "data.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def print_split_stats(splits):
    print("\n| Split | Groups | Images | crop | weed |")
    print("|---|---|---|---|---|")
    for split_name, groups_in_split in splits.items():
        images = [path for group in groups_in_split for path in group]
        counts = [0, 0]
        for image_path in images:
            boxes, _ = read_labels(image_path.with_suffix(".txt"))
            for box in boxes:
                counts[box[0]] += 1
        print(f"| {split_name} | {len(groups_in_split)} | {len(images)} | {counts[0]} | {counts[1]} |")


if __name__ == "__main__":
    data_dir = get_raw_data_dir()
    check_labels(data_dir)
    image_paths = sorted(data_dir.glob("*.jpeg"))
    groups = find_duplicate_groups(image_paths)
    splits = split_groups(groups)
    save_split(splits)
    print_split_stats(splits)