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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model(model_path: Path = DEFAULT_MODEL_PATH) -> YOLO:
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(str(model_path))
    return _model_cache


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

        if not DEFAULT_MODEL_PATH.exists():
            error = (
                "Model weights not found. Train first using: "
                "`python train.py` (expects best.pt in runs/beverage_detect/weights/)."
            )
            return render_template("index.html", error=error, result=result)

        suffix = file.filename.rsplit(".", 1)[1].lower()
        image_id = uuid4().hex

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        upload_path = UPLOAD_DIR / f"{image_id}.{suffix}"
        output_path = OUTPUT_DIR / f"{image_id}.jpg"
        file.save(upload_path)

        model = get_model()
        conf = float(request.form.get("conf", 0.25))
        inference = model(str(upload_path), conf=conf)[0]

        names = model.names
        class_ids = inference.boxes.cls.tolist() if inference.boxes is not None else []
        counts = Counter(names[int(cls_id)] for cls_id in class_ids)

        annotated = inference.plot()
        from cv2 import imwrite

        imwrite(str(output_path), annotated)

        result = {
            "counts": dict(sorted(counts.items())),
            "total": int(sum(counts.values())),
            "input_image": url_for("static", filename=f"uploads/{upload_path.name}"),
            "output_image": url_for("static", filename=f"outputs/{output_path.name}"),
            "conf": conf,
        }

    return render_template("index.html", error=error, result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
