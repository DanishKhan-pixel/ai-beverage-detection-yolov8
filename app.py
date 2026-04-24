from collections import Counter
from pathlib import Path
from typing import Optional
from uuid import uuid4

import cv2
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
UPLOAD_DIR = Path("static/uploads")
OUTPUT_DIR = Path("static/outputs")
DEFAULT_MODEL_PATH = Path("runs/detect/beverage_detect/weights/best.pt")
GENERIC_MODEL_ID = "yolov8n.pt"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

_model_cache: Optional[YOLO] = None
_model_cache_key: Optional[str] = None


def resolve_model_path() -> Optional[Path]:
    candidates = [
        DEFAULT_MODEL_PATH,
        Path("runs/beverage_detect/weights/best.pt"),
        Path("runs/detect/runs/detect/beverage_detect/weights/best.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    discovered = sorted(Path("runs").glob("**/weights/best.pt"))
    return discovered[-1] if discovered else None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model(weights: str) -> YOLO:
    global _model_cache, _model_cache_key
    if _model_cache is None or _model_cache_key != weights:
        _model_cache = YOLO(weights)
        _model_cache_key = weights
    return _model_cache


def count_beverage_like_from_generic(result, names: dict | list) -> Counter:
    """
    When using a COCO-pretrained model, we can't classify brands.
    We approximate beverage counting by counting beverage-like classes.
    """
    beverage_like = {"bottle", "cup", "wine glass"}
    counts: Counter = Counter()
    if result.boxes is None or len(result.boxes) == 0:
        return counts
    class_ids = result.boxes.cls.tolist()
    for cls_id in class_ids:
        label = names[int(cls_id)] if isinstance(names, dict) else names[int(cls_id)]
        if label in beverage_like:
            # Count generic categories when no brand model is available.
            if label == "bottle":
                counts["Bottle"] += 1
            elif label == "cup":
                counts["Cup"] += 1
            else:
                counts["Glass"] += 1
    return counts


def _label_from_names(names: dict | list, cls_id: int) -> str:
    return names[int(cls_id)] if isinstance(names, dict) else names[int(cls_id)]


def detect_bottle_boxes(generic_model: YOLO, image_path: str, conf: float):
    result, used_conf = infer_with_fallback(
        generic_model,
        image_path,
        conf,
        imgsz=960,
        iou=0.45,
        min_conf=0.05,
    )
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes, result, used_conf

    for cls_id, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist()):
        label = _label_from_names(generic_model.names, int(cls_id))
        if label == "bottle":
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            boxes.append((x1, y1, x2, y2))
    return boxes, result, used_conf


def classify_bottles_with_trained_model(image_path: str, bottle_boxes: list[tuple[int, int, int, int]], model: YOLO):
    image = cv2.imread(image_path)
    if image is None:
        return Counter(), None

    allowed_labels = {"CocaCola", "Sprite"}
    counts: Counter = Counter()

    for x1, y1, x2, y2 in bottle_boxes:
        h, w = image.shape[:2]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        crop = image[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue

        result = model(crop, conf=0.05, imgsz=320, iou=0.5)[0]
        label = None
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.tolist()
            cls_ids = result.boxes.cls.tolist()
            best_idx = max(range(len(confs)), key=lambda i: float(confs[i]))
            pred_label = _label_from_names(model.names, int(cls_ids[best_idx]))
            if pred_label in allowed_labels:
                label = pred_label

        if label is not None:
            counts[label] += 1
        cv2.rectangle(image, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.putText(
            image,
            label or "Unknown",
            (x1c, max(0, y1c - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return counts, image


def infer_with_fallback(
    model: YOLO,
    image_path: str,
    conf: float,
    *,
    imgsz: int = 640,
    iou: float = 0.5,
    min_conf: float = 0.01,
):
    # Avoid extremely low confidence that can create many noisy boxes.
    thresholds = [conf, 0.1, 0.05, 0.01]
    thresholds = [t for t in thresholds if t >= min_conf]
    if conf not in thresholds:
        thresholds.insert(0, conf)
    used = conf
    result = model(image_path, conf=conf, imgsz=imgsz, iou=iou)[0]
    if result.boxes is not None and len(result.boxes) > 0:
        return result, used

    for threshold in thresholds[1:]:
        result = model(image_path, conf=threshold, imgsz=imgsz, iou=iou)[0]
        used = threshold
        if result.boxes is not None and len(result.boxes) > 0:
            break

    return result, used


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "Please upload an image file."
            return render_template("index.html", error=error, result=result)

        file = request.files["image"]
        if not file or file.filename == "":
            error = "Please choose an image to analyze."
            return render_template("index.html", error=error, result=result)

        if not allowed_file(file.filename):
            error = "Unsupported file type. Use jpg, jpeg, png, or webp."
            return render_template("index.html", error=error, result=result)

        model_path = resolve_model_path()
        using_generic = model_path is None
        weights = str(model_path) if model_path else GENERIC_MODEL_ID
        if using_generic:
            print("[app] Using generic COCO model (no training): yolov8n.pt")
        else:
            print(f"[app] Using trained weights: {model_path}")

        suffix = file.filename.rsplit(".", 1)[1].lower()
        image_id = uuid4().hex

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        upload_path = UPLOAD_DIR / f"{image_id}.{suffix}"
        output_path = OUTPUT_DIR / f"{image_id}.jpg"
        file.save(upload_path)
        inference_input_path = upload_path

        conf = float(request.form.get("conf", 0.25))
        generic_model = get_model(GENERIC_MODEL_ID)
        bottle_boxes, generic_inference, used_conf = detect_bottle_boxes(
            generic_model, str(inference_input_path), conf
        )

        fallback_used = False
        if using_generic:
            counts = Counter()
            annotated = generic_inference.plot()
        else:
            trained_model = get_model(weights)
            counts, annotated = classify_bottles_with_trained_model(
                str(inference_input_path), bottle_boxes, trained_model
            )
            if sum(counts.values()) == 0:
                counts = Counter()
                annotated = generic_inference.plot()
                fallback_used = True

        if annotated is not None:
            cv2.imwrite(str(output_path), annotated)
        else:
            raw_image = cv2.imread(str(inference_input_path))
            if raw_image is not None:
                cv2.imwrite(str(output_path), raw_image)

        result = {
            "counts": dict(sorted(counts.items())),
            "total": int(sum(counts.values())),
            "input_image": url_for("static", filename=f"uploads/{upload_path.name}"),
            "output_image": url_for("static", filename=f"outputs/{output_path.name}"),
            "conf": conf,
            "used_conf": used_conf,
            "mode": "generic" if using_generic else "trained",
            "fallback_used": fallback_used,
            "weights": weights,
            "classes": ["CocaCola", "Sprite"],
        }

    return render_template("index.html", error=error, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
