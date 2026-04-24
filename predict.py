from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def resolve_model_path(model_path: str | None = None) -> Path:
    if model_path:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Model path not found: {model_path}")

    candidates = [
        Path("runs/detect/beverage_detect/weights/best.pt"),
        Path("runs/detect/runs/beverage_detect/weights/best.pt"),
        Path("runs/beverage_detect/weights/best.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    discovered = sorted(Path("runs").glob("**/weights/best.pt"))
    if discovered:
        return discovered[-1]

    raise FileNotFoundError(
        "No trained model weights found. Run `python train.py` or `python train_roboflow.py ...` first."
    )


def predict_and_count(
    image_path: str,
    model_path: str | None = None,
    conf: float = 0.25,
    output_path: str = "runs/predict/output.jpg",
) -> dict:
    model = YOLO(str(resolve_model_path(model_path)))
    thresholds = [conf, 0.1, 0.05, 0.01]
    result = model(image_path, conf=conf)[0]
    used_conf = conf
    if result.boxes is None or len(result.boxes) == 0:
        for threshold in thresholds[1:]:
            result = model(image_path, conf=threshold)[0]
            used_conf = threshold
            if result.boxes is not None and len(result.boxes) > 0:
                break

    names = model.names
    class_ids = result.boxes.cls.tolist() if result.boxes is not None else []
    counts = Counter(names[int(cls_id)] for cls_id in class_ids)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(output_path, annotated)

    return {"counts": dict(counts), "used_conf": used_conf}


def main() -> None:
    image_path = "sample_images/test.jpg"
    output = predict_and_count(image_path=image_path)
    counts = output["counts"]

    print("\nBeverage Count:")
    print(f"Used confidence: {output['used_conf']}")
    if not counts:
        print("No beverages detected.")
        return

    for label, value in sorted(counts.items()):
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
