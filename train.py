import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

COMMON_DATA_YAML_PATHS = (
    Path("Beverage/data.yaml"),
    Path("dataset/data.yaml"),
    Path("beverage_dataset/data.yaml"),
    Path("data.yaml"),
)

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 beverage detector.")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data.yaml (auto-detects common paths when omitted).",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    return parser.parse_args()


def _first_existing_data_yaml(paths: tuple[Path, ...]) -> Path | None:
    for candidate in paths:
        if candidate.exists():
            return candidate
    return None


def resolve_data_yaml(user_path: str | None) -> Path:
    if user_path:
        data_yaml = Path(user_path)
        if data_yaml.exists():
            return data_yaml

        # If user passed a stale path (e.g., dataset/data.yaml), try common local locations.
        fallback = _first_existing_data_yaml(COMMON_DATA_YAML_PATHS)
        if fallback:
            print(
                f"Warning: provided data path not found ({data_yaml}). "
                f"Using detected config: {fallback}"
            )
            return fallback
        raise FileNotFoundError(f"Provided data.yaml not found: {data_yaml}")

    discovered = _first_existing_data_yaml(COMMON_DATA_YAML_PATHS)
    if discovered:
        return discovered

    raise FileNotFoundError(
        "No data.yaml found. Expected one of: Beverage/data.yaml, dataset/data.yaml, beverage_dataset/data.yaml, or data.yaml."
    )


def main() -> None:
    args = parse_args()
    data_yaml = resolve_data_yaml(args.data)
    validate_dataset(data_yaml)

    model = YOLO("yolov8n.pt")
    print(model)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        # Mild augmentation helps when adding varied shelf/fridge photos.
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project="runs/detect",
        name="beverage_detect",
    )

    model.val(data=str(data_yaml))
# test the model

if __name__ == "__main__":
    main()
