import argparse
import os
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YOLOv8 dataset from Roboflow and train YOLOv8."
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("ROBOFLOW_WORKSPACE"),
        help="Roboflow workspace ID (or set ROBOFLOW_WORKSPACE)",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("ROBOFLOW_PROJECT"),
        help="Roboflow project ID (or set ROBOFLOW_PROJECT)",
    )
    parser.add_argument(
        "--version",
        default=int(os.environ.get("ROBOFLOW_VERSION", "1")),
        type=int,
        help="Dataset version number (or set ROBOFLOW_VERSION, default: 1)",
    )
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
    if not args.workspace:
        raise ValueError("Missing workspace. Pass --workspace or set ROBOFLOW_WORKSPACE.")
    if not args.project:
        raise ValueError("Missing project. Pass --project or set ROBOFLOW_PROJECT.")

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
        project="runs/detect",
        name="beverage_detect",
    )
    model.val(data=str(data_yaml_path))


if __name__ == "__main__":
    main()
