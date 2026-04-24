from pathlib import Path

import yaml
from ultralytics import YOLO


def _count_files(path: Path, patterns: tuple[str, ...]) -> int:
    total = 0
    for pattern in patterns:
        total += len(list(path.glob(pattern)))
    return total


def validate_dataset(data_yaml_path: Path) -> None:
    config = yaml.safe_load(data_yaml_path.read_text())
    dataset_root = Path(config.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()

    train_images = (dataset_root / config["train"]).resolve()
    val_images = (dataset_root / config["val"]).resolve()

    train_labels = Path(str(train_images).replace("/images", "/labels"))
    val_labels = Path(str(val_images).replace("/images", "/labels"))

    image_patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    label_patterns = ("*.txt",)

    train_img_count = _count_files(train_images, image_patterns)
    val_img_count = _count_files(val_images, image_patterns)
    train_lbl_count = _count_files(train_labels, label_patterns)
    val_lbl_count = _count_files(val_labels, label_patterns)

    if train_img_count == 0 or val_img_count == 0:
        raise FileNotFoundError(
            "Dataset images missing. Ensure train/val image folders are populated.\n"
            "Tip: use Roboflow download script:\n"
            "python download_roboflow_dataset.py --url '<roboflow-yolov8-url>'"
        )

    if train_lbl_count == 0 or val_lbl_count == 0:
        raise FileNotFoundError(
            "Dataset label files missing. YOLO labels are required in train/val labels folders."
        )


def main() -> None:
    data_yaml = Path("data.yaml")
    validate_dataset(data_yaml)

    # Start from a lightweight pretrained detector for faster fine-tuning.
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        project="runs",
        name="beverage_detect",
    )

    # Runs validation on the best checkpoint and prints metrics.
    model.val(data="data.yaml")


if __name__ == "__main__":
    main()
