from pathlib import Path

import yaml
from ultralytics import YOLO


def _count_images(path: Path) -> int:
    return sum(len(list(path.glob(ext))) for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"))


def validate_dataset(data_yaml_path: Path) -> None:
    config = yaml.safe_load(data_yaml_path.read_text())
    dataset_root_raw = Path(config.get("path", "."))
    if dataset_root_raw.is_absolute():
        dataset_root = dataset_root_raw
    else:
        candidate_from_yaml_dir = (data_yaml_path.parent / dataset_root_raw).resolve()
        candidate_from_cwd = Path.cwd() / dataset_root_raw
        if candidate_from_yaml_dir.exists():
            dataset_root = candidate_from_yaml_dir
        elif candidate_from_cwd.exists():
            dataset_root = candidate_from_cwd.resolve()
        else:
            dataset_root = data_yaml_path.parent.resolve()

    train_images = (dataset_root / config["train"]).resolve()
    val_images = (dataset_root / config["val"]).resolve()

    train_labels = Path(str(train_images).replace("/images", "/labels"))
    val_labels = Path(str(val_images).replace("/images", "/labels"))

    train_img_count = _count_images(train_images)
    val_img_count = _count_images(val_images)
    train_lbl_count = len(list(train_labels.glob("*.txt")))
    val_lbl_count = len(list(val_labels.glob("*.txt")))

    if train_img_count == 0 or val_img_count == 0:
        raise FileNotFoundError(
            "Dataset images missing. Ensure train/val image folders are populated."
        )

    if train_lbl_count == 0 or val_lbl_count == 0:
        raise FileNotFoundError(
            "Dataset label files missing. YOLO labels are required in train/val labels folders."
        )


def main() -> None:
    data_yaml = Path("beverage_dataset/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError("`beverage_dataset/data.yaml` not found.")
    validate_dataset(data_yaml)

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs/detect",
        name="beverage_detect",
    )

    # Runs validation on the best checkpoint and prints metrics.
    model.val(data=str(data_yaml))


if __name__ == "__main__":
    main()
