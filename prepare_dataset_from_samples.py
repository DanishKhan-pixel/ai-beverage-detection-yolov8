from __future__ import annotations

from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    sample_dir = repo_root / "sample_images"
    if not sample_dir.exists():
        raise FileNotFoundError(f"Missing sample_images folder: {sample_dir}")

    dataset_root = repo_root / "beverage_dataset"
    train_images = dataset_root / "train/images"
    train_labels = dataset_root / "train/labels"
    valid_images = dataset_root / "valid/images"
    valid_labels = dataset_root / "valid/labels"

    train_images.mkdir(parents=True, exist_ok=True)
    train_labels.mkdir(parents=True, exist_ok=True)
    valid_images.mkdir(parents=True, exist_ok=True)
    valid_labels.mkdir(parents=True, exist_ok=True)

    samples = sorted(
        [p for p in sample_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )
    if not samples:
        raise FileNotFoundError(f"No images found in: {sample_dir}")

    copied = 0
    for src in samples:
        # Normalize filenames so YOLO label pairing is stable.
        safe_stem = src.stem.replace(" ", "_").replace("(", "").replace(")", "")
        dst_img = train_images / f"sample_{safe_stem}{src.suffix.lower()}"
        dst_lbl = train_labels / f"{dst_img.stem}.txt"

        if not dst_img.exists():
            dst_img.write_bytes(src.read_bytes())
            copied += 1

        # Create empty label file (background image) if missing.
        dst_lbl.touch(exist_ok=True)

    print(f"Copied {copied} sample images into {train_images}")
    print("Created empty labels for sample images (treated as background).")
    print("Now train with:")
    print("  python train.py --data beverage_dataset/data.yaml --epochs 120 --imgsz 640 --batch 8")


if __name__ == "__main__":
    main()

