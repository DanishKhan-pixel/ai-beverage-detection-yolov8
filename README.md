# Beverage Object Recognition and Counting (Flask + YOLOv8)

This project detects and counts beverages (e.g., CocaCola, Sprite, Water, Juice, Other) from fridge/shelf images.
It uses **Ultralytics YOLOv8** for object detection and a **Flask web app** for easy inference demos.

## Why this architecture

- **YOLOv8**: strong baseline for object detection, easy fine-tuning, fast inference.
- **Transfer learning**: starting from `yolov8n.pt` reduces training time and improves results with smaller datasets.
- **Flask app**: simple interface to upload an image, visualize bounding boxes, and show class counts.

## Project structure

```
beverage-recognition/
├── app.py
├── train.py
├── predict.py
├── data.yaml
├── requirements.txt
├── README.md
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── templates/
│   └── index.html
└── static/
    ├── uploads/
    └── outputs/
```

## 1) Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2) Prepare dataset

Use YOLO format labels (`class x_center y_center width height`, normalized).

Place files like this:

- `dataset/train/images/*.jpg`
- `dataset/train/labels/*.txt`
- `dataset/valid/images/*.jpg`
- `dataset/valid/labels/*.txt`
- `dataset/test/images/*.jpg`
- `dataset/test/labels/*.txt`

Class mapping is in `data.yaml`:

1. CocaCola
2. Sprite
3. Water
4. Juice
5. Other

## 3) Train model

```bash
python train.py
```

Best model is saved at:

`runs/beverage_detect/weights/best.pt`

Training/eval artifacts include:

- `runs/beverage_detect/results.png`
- mAP, precision, recall logs in the run directory

## 4) Run CLI inference + counting

Put a test image at `sample_images/test.jpg` (or edit path in `predict.py`), then:

```bash
python predict.py
```

Output image with boxes:

`runs/predict/output.jpg`

Console output example:

```
Beverage Count:
CocaCola: 5
Sprite: 3
Water: 4
Other: 2
```

## 5) Run Flask web app

```bash
python app.py
```

Open:

`http://127.0.0.1:5000`

Upload an unseen fridge/shelf image and get:

- Annotated output image with bounding boxes/labels
- Count summary per beverage class

## Evaluation guidance for submission

Include in your report/demo:

- Dataset source and size (train/val/test split)
- Why YOLOv8 was chosen
- Training settings (epochs, batch size, image size)
- Metrics (mAP, precision, recall)
- At least one unseen demo image with counts

## Deliverables checklist

- [ ] GitHub repo with all code
- [ ] README with run steps
- [ ] Architecture + technology explanation
- [ ] Demo video
- [ ] At least one annotated demo image + count summary

