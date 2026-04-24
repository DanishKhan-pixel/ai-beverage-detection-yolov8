import argparse
import os
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YOLOv8 dataset from Roboflow and train YOLOv8."
    )
    parser.add_argument("--workspace", required=True, help="Roboflow workspace ID")
    parser.add_argument("--project", required=True, help="Roboflow project ID")
    parser.add_argument("--version", required=True, type=int, help="Dataset version number")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api-key or set ROBOFLOW_API_KEY.")

    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)

    print("Downloading Roboflow dataset in YOLOv8 format...")
    dataset = version.download("yolov8")
    data_yaml_path = Path(dataset.location) / "data.yaml"

    print(f"Dataset downloaded at: {dataset.location}")
    print(f"Using config: {data_yaml_path}")

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="beverage_detect",
    )
    model.val(data=str(data_yaml_path))


if __name__ == "__main__":
    main()
