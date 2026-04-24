import argparse
import shutil
import zipfile
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Roboflow YOLOv8 zip dataset and extract into local dataset folder."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Roboflow YOLOv8 download URL (from Roboflow export page).",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Local destination directory for extracted dataset.",
    )
    parser.add_argument(
        "--zip-path",
        default="roboflow_yolov8.zip",
        help="Temporary zip file path.",
    )
    return parser.parse_args()


def download_file(url: str, zip_path: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    zip_path.write_bytes(response.content)


def find_extracted_root(extract_dir: Path) -> Path:
    data_yaml_files = list(extract_dir.rglob("data.yaml"))
    if not data_yaml_files:
        raise FileNotFoundError("No data.yaml found in downloaded zip.")
    return data_yaml_files[0].parent


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    extract_root = Path("_tmp_roboflow_extract")
    output_dir = Path(args.output_dir)

    print("Downloading Roboflow YOLOv8 dataset zip...")
    download_file(args.url, zip_path)

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    dataset_root = find_extracted_root(extract_root)
    print(f"Detected dataset root: {dataset_root}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(dataset_root, output_dir)

    zip_path.unlink(missing_ok=True)
    shutil.rmtree(extract_root, ignore_errors=True)

    print(f"Dataset added successfully at: {output_dir.resolve()}")
    print("Now run: python train.py")


if __name__ == "__main__":
    main()
