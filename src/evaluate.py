import json
import sys

from ultralytics import YOLO

CONF = 0.001  # minimum confidence to keep a prediction
IOU = 0.7     # IoU threshold used when removing duplicate boxes


def main(run_name, imgsz):
    model = YOLO(f"runs/detect/{run_name}/weights/best.pt")
    metrics = model.val(
        data="data/data.yaml",
        split="test",
        imgsz=imgsz,
        conf=CONF,
        iou=IOU,
        name=f"{run_name}_test",
        exist_ok=True,
    )

    results = {
        "run": run_name,
        "split": "test",
        "imgsz": imgsz,
        "conf": CONF,
        "nms_iou": IOU,
        "precision": round(float(metrics.box.mp), 3),
        "recall": round(float(metrics.box.mr), 3),
        "mAP50": round(float(metrics.box.map50), 3),
    }
    print(results)

    with open("res/test_metrics.json", "w") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))