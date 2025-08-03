from ultralytics import YOLO
import os
from PIL import Image
import argparse

# --- Define config ---
EPOCHS = 40
PATIENCE = 6
MOSAIC = 0.5
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False
DATA_YAML = "/content/drive/MyDrive/HackByte_Dataset/yolo_params.yaml"
WEIGHTS_DIR = "/content/drive/MyDrive/HackByte_Dataset/runs/detect/train/weights"
VAL_IMAGES_DIR = "/content/drive/MyDrive/HackByte_Dataset/data/val/images"

# --- Train Model ---
model = YOLO("yolov8m.pt")

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    patience=PATIENCE,
    device=0,
    single_cls=SINGLE_CLS,
    mosaic=MOSAIC,
    optimizer=OPTIMIZER,
    lr0=LR0,
    lrf=LRF,
    momentum=MOMENTUM,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5, shear=0.1,
    flipud=0.5, fliplr=0.5,
    workers=2
)

# --- Validate Model and Show Confusion Matrix ---
val_results = model.val(data=DATA_YAML, plots=True)

# --- Run Inference on 10 Val Images ---
val_images = sorted(os.listdir(VAL_IMAGES_DIR))[:10]

print("\n--- Running Inference on 10 Validation Images ---")
for img_file in val_images:
    img_path = os.path.join(VAL_IMAGES_DIR, img_file)
    model.predict(img_path, save=True, show=True)

# --- Export Model to ONNX ---
print("\n--- Exporting Model to ONNX ---")
model.export(format="onnx")
