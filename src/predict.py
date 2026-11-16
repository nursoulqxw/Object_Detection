import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

from show_gt import draw_image

CONF = 0.25  # only show predictions with confidence >= 0.25
OUTPUT_DIR = Path("res/preds")


def main(run_name, imgsz):
    model = YOLO(f"runs/detect/{run_name}/weights/best.pt")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open("res/split.csv") as file:
        rows = list(csv.DictReader(file))
    names = [
        row["image"] for row in rows
        if row["split"] == "test" and row["image"] == row["group"] + ".jpeg"
    ]

    for name in names:
        image_path = Path("data/images/test") / name
        label_path = Path("data/labels/test") / name.replace(".jpeg", ".txt")
        result = model.predict(image_path, imgsz=imgsz, conf=CONF, verbose=False)[0]

        fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(10, 5))
        draw_image(ax_gt, image_path, label_path)
        ax_gt.set_title(f"Ground truth: {name}", fontsize=9)

        ax_pred.imshow(result.plot()[:, :, ::-1])
        ax_pred.set_title(f"Prediction (conf >= {CONF}): {len(result.boxes)} boxes", fontsize=9)
        ax_pred.axis("off")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / name.replace(".jpeg", ".png"), dpi=100)
        plt.close(fig)

    print(f"Saved {len(names)} images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))