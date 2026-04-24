from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def predict_and_count(
    image_path: str,
    model_path: str = "runs/beverage_detect/weights/best.pt",
    conf: float = 0.25,
    output_path: str = "runs/predict/output.jpg",
) -> dict:
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    result = results[0]

    names = model.names
    class_ids = result.boxes.cls.tolist() if result.boxes is not None else []
    counts = Counter(names[int(cls_id)] for cls_id in class_ids)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()
    cv2.imwrite(output_path, annotated)

    return dict(counts)


def main() -> None:
    image_path = "sample_images/test.jpg"
    counts = predict_and_count(image_path=image_path)

    print("\nBeverage Count:")
    if not counts:
        print("No beverages detected.")
        return

    for label, value in sorted(counts.items()):
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
