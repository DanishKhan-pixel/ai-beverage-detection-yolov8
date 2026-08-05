BEVERAGE RECOGNITION - SIMPLE GUIDE
===================================

What it does
------------
Finds bottles/cans in photos and counts them by class (e.g. CocaCola, Sprite).
You train a small model on your images, then use the web page or a script.


Technologies
------------
- Python: main language
- Ultralytics YOLOv8: object detection (draws boxes, assigns class)
- Transfer learning: start from yolov8n.pt, fine-tune on your dataset
- Flask: web upload page (app.py)
- OpenCV: save image with boxes drawn
- PyYAML: read data.yaml for training paths and class names

Method: you label boxes on images (YOLO format) -> train -> model outputs
class + score per box -> we count how many of each class.


How the web app works
---------------------
- If best.pt exists under runs/.../weights/: uses your trained classes.
- If no weights: uses yolov8n.pt (COCO). It does not know brands; only
  rough counts for bottle / cup / wine glass.
- If first detection fails: tries again with lower confidence (down to a limit).
- If trained model finds nothing: may run the generic model once for a fallback.


Main files
----------
app.py          Web UI: upload image, show counts and annotated picture
train.py        Train YOLO; finds data.yaml automatically in common folders
predict.py      Command line: one image, print counts, save output image
prepare_dataset_from_samples.py   Copy sample images into train folder
data.yaml       Example: dataset path and class names
requirements.txt

Folders: templates/, static/uploads/, static/outputs/
Dataset images usually in dataset/ or Beverage/ (see your data.yaml).


Setup
-----
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


Dataset (YOLO)
--------------
train/images/     validation images in valid/images/
train/labels/     one .txt per image, same file name
valid/labels/

Each label line: class_id x_center y_center width height
Numbers are 0 to 1 (normalized).

Example classes (from root data.yaml):
0 CocaCola
1 Sprite
2 Water
3 Juice
4 Other

train.py looks for: Beverage/data.yaml, dataset/data.yaml,
beverage_dataset/data.yaml, or data.yaml


Train
-----
python train.py

Optional:
python train.py --data dataset/data.yaml --epochs 100 --imgsz 640 --batch 8

Weights saved about here:
runs/detect/beverage_detect/weights/best.pt

Clear bad label cache if needed:
rm -f dataset/train/labels.cache dataset/valid/labels.cache


Predict (CLI)
-------------
Edit image path inside predict.py if needed, then:
python predict.py

Browser: http://127.0.0.1:5000


Roboflow
--------
Export as YOLOv8, unzip, train with --data pointing to that data.yaml
