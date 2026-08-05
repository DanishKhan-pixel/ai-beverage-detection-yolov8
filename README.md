# Beverage recognition

This repo is a small project that looks at a fridge or shelf photo and tries to **spot drinks**, draw boxes around them, and **count** how many of each type it thinks it sees (for example CocaCola, Sprite, and so on). Nothing fancy on the website side — you upload a picture and it shows you the result.

---

## How it actually works (in normal words)

1. You collect photos and **label** them (each bottle gets a box and a category). That is your training data.
2. You run **`train.py`**. It uses **YOLOv8** (Ultralytics), which is a common “find objects in an image” model. Training does not start from zero: it begins from **`yolov8n.pt`**, which was already trained on a big public dataset, then **adjusts** the weights on your beverage pictures. That usually works better with less data than training from scratch.
3. Training saves a file called **`best.pt`** under something like `runs/detect/beverage_detect/weights/`.
4. **`app.py`** is a tiny **Flask** website. You open it in the browser, upload an image, and the app loads `best.pt`, runs the model, saves a copy of the image with boxes drawn on it (**OpenCV** helps write that file), and shows you counts.
5. If you do not have a trained `best.pt` yet, the app can still run using the generic **`yolov8n.pt`** model. That one was not trained on your brands, so you only get rough categories like bottle / cup / glass — good enough to see the pipeline working, not good enough for real brand accuracy.

So in short: **labels + train → weights file → upload image → boxes + numbers on the screen.**

---

## What you need installed

- Python 3  
- Packages from **`requirements.txt`**: ultralytics (YOLO), flask, opencv-python, pyyaml  

Install:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `venv\Scripts\activate` instead of `source venv/bin/activate`.

---

## Where things live in this folder

- **`app.py`** — web UI (upload, counts, saved output image).  
- **`train.py`** — training; it tries to find a `data.yaml` in a few usual places (`Beverage/`, `dataset/`, `beverage_dataset/`, or the root `data.yaml`).  
- **`predict.py`** — same idea as the app but from the terminal for one image.  
- **`data.yaml`** — tells training where your train/val images are and what the class names are.  
- **`templates/index.html`** — the HTML for the Flask page.  
- **`static/uploads/`** and **`static/outputs/`** — filled when you use the web app.

Your real dataset is usually **not** all inside this readme; it sits in folders pointed to by `data.yaml` (often `dataset/` or similar).

---

## Dataset (YOLO style, very short)

Put images in `train/images` and `valid/images`. For each image, you need a matching `.txt` in `train/labels` and `valid/labels` with the same base name.



All five numbers are normalized between 0 and 1 (that is just how YOLO wants it).

Example classes from the sample **`data.yaml`** in this repo:

- 0 — CocaCola  
- 1 — Sprite  
- 2 — Water  
- 3 — Juice  
- 4 — Other  

## Train the model

```bash
python train.py
```



---

## Run from the command line

After you have weights, open **`predict.py`**, set the image path inside if needed, then:

```bash
python predict.py


## Run the website

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, upload a jpg/png/webp, and you should see the annotated image plus the counts.