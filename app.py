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
# Display-only: show CocaCola instead of these count/plot labels (detection unchanged).
SHOW_AS_COCACOLA = frozenset({"Other", "Bottle"})

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
    print(candidates)
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
# python test.py --weights runs/detect/beverage_detect/weights/best.pt --source 0


def count_beverage_like_from_generic(result, names: dict | list) -> Counter:

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



def infer_with_fallback(
    # python test.py --weights runs/detect/beverage_detect/weights/best.pt --source 0
    # python test.py --weights runs/detect/beverage_detect/weights/best.pt --source 0
    # python test.py --weights runs/detect/beverage_detect/weights/best.pt --source 0
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


def _display_label(raw: str) -> str:
    if raw in SHOW_AS_COCACOLA:
        return "CocaCola"
    if raw == "bottle":
        return "CocaCola"
    return raw


def counts_for_display(counts: Counter) -> dict:
    out = Counter()
    for name, n in counts.items():
        out[_display_label(name)] += n
    return dict(sorted(out.items()))


def annotate_with_display_labels(inference, names: dict | list):
    """Same boxes and scores; only label text is remapped for SHOW_AS_COCACOLA / COCO bottle."""
    if inference.boxes is None or len(inference.boxes) == 0:
        return inference.plot()
    base = inference.orig_img
    if base is None:
        return inference.plot()
    out = base.copy()
    for i in range(len(inference.boxes)):
        xyxy = inference.boxes.xyxy[i].cpu().numpy().tolist()
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        score = float(inference.boxes.conf[i])
        cls_id = int(inference.boxes.cls[i])
        raw = names[cls_id] if isinstance(names, dict) else names[cls_id]
        raw = str(raw)
        label = _display_label(raw)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(
            out,
            text,
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return out


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

        model = get_model(weights)
        conf = float(request.form.get("conf", 0.25))
        # Generic COCO model often benefits from a larger inference size on bottle shelves.
        imgsz = 960 if using_generic else 640
        min_conf = 0.05 if using_generic else 0.01
        iou = 0.45 if using_generic else 0.5
        inference, used_conf = infer_with_fallback(
            model,
            str(inference_input_path),
            conf,
            imgsz=imgsz,
            iou=iou,
            min_conf=min_conf,
        )

        names = model.names
        names_for_plot = names
        fallback_used = False
        if using_generic:
            counts = count_beverage_like_from_generic(inference, names)
        else:
            class_ids = inference.boxes.cls.tolist() if inference.boxes is not None else []
            counts = Counter(names[int(cls_id)] for cls_id in class_ids)

            # If trained model misses everything, fallback to generic bottle counting.
            if sum(counts.values()) == 0:
                generic_model = get_model(GENERIC_MODEL_ID)
                generic_inference, _ = infer_with_fallback(
                    generic_model,
                    str(inference_input_path),
                    conf,
                    imgsz=960,
                    iou=0.45,
                    min_conf=0.05,
                )
                generic_counts = count_beverage_like_from_generic(
                    generic_inference, generic_model.names
                )
                if sum(generic_counts.values()) > 0:
                    counts = generic_counts
                    inference = generic_inference
                    names_for_plot = generic_model.names
                    fallback_used = True

        annotated = annotate_with_display_labels(inference, names_for_plot)
        cv2.imwrite(str(output_path), annotated)

        result = {
            "counts": counts_for_display(counts),
            "total": int(sum(counts.values())),
            "input_image": url_for("static", filename=f"uploads/{upload_path.name}"),
            "output_image": url_for("static", filename=f"outputs/{output_path.name}"),
            "conf": conf,
            "used_conf": used_conf,
            "mode": "generic" if using_generic else "trained",
            "fallback_used": fallback_used,
            "weights": weights,
            "classes": list(names.values()) if isinstance(names, dict) else list(names),
        }

    return render_template("index.html", error=error, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
