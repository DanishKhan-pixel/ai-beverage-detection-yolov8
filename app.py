from collections import Counter
from pathlib import Path
from typing import Optional
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from ultralytics import YOLO

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
UPLOAD_DIR = Path("static/uploads")
OUTPUT_DIR = Path("static/outputs")
DEFAULT_MODEL_PATH = Path("runs/beverage_detect/weights/best.pt")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# Lazy-loaded model cache to avoid reloading weights every request.
_model_cache: Optional[YOLO] = None
_generic_model_cache: Optional[YOLO] = None


def resolve_model_path() -> Optional[Path]:
    # Try known locations first, then fallback to any best.pt in runs.
    candidates = [
        DEFAULT_MODEL_PATH,
        Path("runs/detect/beverage_detect/weights/best.pt"),
        Path("runs/detect/runs/beverage_detect/weights/best.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    discovered = sorted(Path("runs").glob("**/weights/best.pt"))
    return discovered[-1] if discovered else None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model(model_path: Path) -> YOLO:
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(str(model_path))
    return _model_cache


def get_generic_model() -> YOLO:
    global _generic_model_cache
    if _generic_model_cache is None:
        _generic_model_cache = YOLO("yolov8n.pt")
    return _generic_model_cache


def infer_with_fallback(model: YOLO, image_path: str, conf: float):
    thresholds = [conf, 0.1, 0.05, 0.01]
    used = conf
    inference = model(image_path, conf=conf)[0]
    count = 0 if inference.boxes is None else len(inference.boxes)
    if count > 0:
        return inference, used

    for threshold in thresholds[1:]:
        inference = model(image_path, conf=threshold)[0]
        count = 0 if inference.boxes is None else len(inference.boxes)
        used = threshold
        if count > 0:
            break

    return inference, used


def is_unreliable_custom_result(inference, used_conf: float) -> bool:
    if inference.boxes is None or len(inference.boxes) == 0:
        return True
    confidences = inference.boxes.conf.tolist()
    max_conf = max(float(score) for score in confidences) if confidences else 0.0
    # Treat very low-score outputs as noise from an underfit model.
    return used_conf <= 0.01 and max_conf < 0.03


def coco_fallback_counts(image_path: str) -> tuple[dict, list]:
    generic_model = get_generic_model()
    inference = generic_model(image_path, conf=0.1)[0]
    names = generic_model.names
    class_ids = inference.boxes.cls.tolist() if inference.boxes is not None else []
    confs = inference.boxes.conf.tolist() if inference.boxes is not None else []

    beverage_like = {"bottle", "cup", "wine glass"}
    other_count = 0
    debug_rows = []
    for cls_id, score in zip(class_ids, confs):
        label = names[int(cls_id)]
        if label in beverage_like:
            other_count += 1
            debug_rows.append(
                {"label": f"Other ({label})", "confidence": round(float(score), 4), "source": "generic"}
            )

    counts = {"Other": other_count} if other_count > 0 else {}
    return counts, debug_rows


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
        if model_path is None:
            error = (
                "Model weights not found. Train first using: "
                "`python train.py` (or `python train_roboflow.py ...`) and ensure best.pt exists in runs."
            )
            return render_template("index.html", error=error, result=result)

        suffix = file.filename.rsplit(".", 1)[1].lower()
        image_id = uuid4().hex

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        upload_path = UPLOAD_DIR / f"{image_id}.{suffix}"
        output_path = OUTPUT_DIR / f"{image_id}.jpg"
        file.save(upload_path)

        model = get_model(model_path)
        conf = float(request.form.get("conf", 0.25))
        debug_mode = request.form.get("debug_mode") == "on"
        inference, used_conf = infer_with_fallback(model, str(upload_path), conf)

        names = model.names
        class_ids = inference.boxes.cls.tolist() if inference.boxes is not None else []
        counts = Counter(names[int(cls_id)] for cls_id in class_ids)
        debug_detections = []
        if debug_mode and inference.boxes is not None:
            for cls_id, score in zip(inference.boxes.cls.tolist(), inference.boxes.conf.tolist()):
                debug_detections.append(
                    {
                        "label": names[int(cls_id)],
                        "confidence": round(float(score), 4),
                        "source": "custom",
                    }
                )

        fallback_used = False
        if sum(counts.values()) == 0 or is_unreliable_custom_result(inference, used_conf):
            fallback_counts, fallback_debug = coco_fallback_counts(str(upload_path))
            if fallback_counts:
                counts = Counter(fallback_counts)
                fallback_used = True
                if debug_mode:
                    debug_detections.extend(fallback_debug)

        annotated = inference.plot()
        from cv2 import imwrite

        imwrite(str(output_path), annotated)

        result = {
            "counts": dict(sorted(counts.items())),
            "total": int(sum(counts.values())),
            "input_image": url_for("static", filename=f"uploads/{upload_path.name}"),
            "output_image": url_for("static", filename=f"outputs/{output_path.name}"),
            "conf": conf,
            "used_conf": used_conf,
            "debug_mode": debug_mode,
            "debug_detections": debug_detections,
            "fallback_used": fallback_used,
        }

    return render_template("index.html", error=error, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
