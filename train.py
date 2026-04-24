from ultralytics import YOLO


def main() -> None:
    # Start from a lightweight pretrained detector for faster fine-tuning.
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
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
