import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

CLASS_NAMES = ["crop", "weed"]
COLORS = ["lime", "red"]
SEED = 15
NUM_IMAGES = 6


def read_boxes(label_path):
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if parts:
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            boxes.append((class_id, x, y, w, h))
    return boxes


def draw_image(ax, image_path, label_path):
    image = Image.open(image_path)
    width, height = image.size
    ax.imshow(image)

    for class_id, x, y, w, h in read_boxes(label_path):
        left = (x - w / 2) * width
        top = (y - h / 2) * height
        box_width = w * width
        box_height = h * height

        rectangle = Rectangle((left, top), box_width, box_height,
                              fill=False, edgecolor=COLORS[class_id], linewidth=2)
        ax.add_patch(rectangle)
        ax.text(left, top - 5, CLASS_NAMES[class_id], color="black",
                backgroundcolor=COLORS[class_id], fontsize=8)

    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")


if __name__ == "__main__":
    with open("res/split.csv") as file:
        rows = list(csv.DictReader(file))

    candidates = [
        row["image"] for row in rows
        if row["split"] == "train" and row["image"] == row["group"] + ".jpeg"
    ]
    chosen = random.Random(SEED).sample(candidates, NUM_IMAGES)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, name in zip(axes.flat, chosen):
        image_path = Path("data/images/train") / name
        label_path = Path("data/labels/train") / name.replace(".jpeg", ".txt")
        draw_image(ax, image_path, label_path)

    plt.tight_layout()
    plt.savefig("res/gt_examples.png", dpi=120)
    print("Saved res/gt_examples.png")