import sys
import time

import yaml
from ultralytics import YOLO


def main(config_path, quick):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    if quick:
        config["epochs"] = 1
        config["fraction"] = 0.1
        config["name"] = config["name"] + "_quick"

    model = YOLO(config.pop("model"))

    start = time.time()
    model.train(**config)
    minutes = (time.time() - start) / 60

    print(f"Training time: {minutes:.1f} min")
    save_dir = model.trainer.save_dir
    with open(save_dir / "train_time.txt", "w") as file:
        file.write(f"{minutes:.1f} min\n")


if __name__ == "__main__":
    config_path = sys.argv[1]
    quick = "--quick" in sys.argv
    main(config_path, quick)