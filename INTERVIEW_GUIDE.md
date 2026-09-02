# Beverage Recognition Interview Guide

## 30-Second Project Pitch
This project is a beverage detection and counting web app. A user uploads an image, the system detects beverage objects, draws bounding boxes, and returns class-wise counts (for example: CocaCola, Sprite, Water).  
I built it with Python, YOLOv8, Flask, and OpenCV. It first tries my trained model; if weights are not available (or detections fail), it falls back to a generic YOLO model for bottle-like object counting.

## Problem Statement
- Retail/shelf images are hard to count manually.
- The goal is to automate beverage detection and counting from photos.

## Solution Overview
- Use object detection (YOLOv8) on uploaded images.
- Return:
  - Class counts
  - Total item count
  - Annotated output image with boxes and confidence scores

## Tech Stack and Why
- Python: fast development for ML + backend.
- Ultralytics YOLOv8: modern, fast object detection framework.
- Transfer Learning: fine-tune `yolov8n.pt` on a custom beverage dataset.
- Flask: lightweight web app for upload/inference/results.
- OpenCV: annotation rendering and output image saving.
- PyYAML: dataset config (`data.yaml`) parsing in training flow.

## Dataset and Labeling
- YOLO format:
  - image folders: `train/images`, `valid/images`
  - label folders: `train/labels`, `valid/labels`
- Label row format:
  - `class_id x_center y_center width height`
  - values are normalized between 0 and 1.

## How the App Works (Runtime Flow)
1. User uploads image from web UI.
2. App validates extension (`jpg`, `jpeg`, `png`, `webp`) and size.
3. App resolves model path:
   - trained weights if found (`runs/.../weights/best.pt`)
   - else generic `yolov8n.pt`.
4. Inference runs with chosen confidence threshold.
5. If no boxes, app retries with lower confidence thresholds.
6. Class IDs are converted to names and counted.
7. Annotated image is saved and shown in UI.
8. User sees counts, total, input image, and output image.

## Key Engineering Decisions (Good Interview Points)
- Model caching in memory to avoid repeated model loading.
- Confidence fallback thresholds to handle difficult images.
- Generic fallback model when custom model is unavailable.
- Input validation and upload/output folder management.

## Current Limitations
- Accuracy depends on data quality and class balance.
- Similar-looking packaging can reduce precision.
- Generic model fallback cannot do brand-specific classification.
- This is a prototype web app, not a full production deployment.

## Future Improvements
- Add more labeled training data and class balancing.
- Use augmentation and hyperparameter tuning.
- Track metrics in UI (precision, recall, mAP).
- Package with Docker and deploy an API service.

## Demo Script (2 Minutes)
1. "I trained YOLOv8 on a custom beverage dataset."
2. "I run the Flask app and open the upload page."
3. "I upload a shelf image."
4. "The app detects items, draws boxes, and shows class-wise counts."
5. "If the trained model is missing or fails, fallback logic still gives useful beverage-like counts."

## Common Interview Questions and Answers

### Why YOLOv8?
YOLOv8 gives strong detection accuracy with fast inference and an easy training pipeline.

### Why Flask instead of Django/FastAPI?
Flask is simple and lightweight for a quick ML inference prototype.

### How did you improve robustness?
By adding confidence fallback and a generic model fallback for edge cases.


