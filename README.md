# Akatsuki13
AI object detection model

# 🚀 Space Station Object Detection using YOLOv8

This project implements an object detection system using the **YOLOv8** model to detect objects (e.g., fire extinguishers, toolboxes, oxygen tanks) in synthetic images of a **space station**. The dataset was generated using **FalconEditor**, simulating complex environments with varying lighting, occlusion, overlapping objects, and different camera angles/distances.

## 🗂️ Dataset

The dataset was synthetically generated using **FalconEditor** and follows the YOLO format:
/HackByte_Dataset/

├── data/

│ ├── train/

│ │ ├── images/

│ │ └── labels/
│ ├── val/
│ │ ├── images/
│ │ └── labels/
├── yolo_params.yaml


- **Classes**: `FireExtinguisher`, `ToolBox`, `OxygenTank`
- **Image Count**: 846 images total across train and val
- **Annotation Format**: YOLO (`class_id x_center y_center width height`)

## ⚙️ Model & Training

We used **YOLOv8m** from the [Ultralytics](https://github.com/ultralytics/ultralytics) library, which provides a balanced trade-off between speed and accuracy.

### ✅ YOLOv8 Training Parameters:


EPOCHS = 40
PATIENCE = 6
MOSAIC = 0.5
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False

Data Augmentations Applied:
hsv_h=0.015: Random change in image hue

hsv_s=0.7: Random saturation variation (70%)

hsv_v=0.4: Brightness variation (40%)

degrees=10.0: Rotate image up to ±10 degrees

translate=0.1: Shift image up to 10%

scale=0.5: Random zoom in/out up to 50%

shear=0.1: Distort image by 10% shearing

flipud=0.5: 50% chance to vertically flip

fliplr=0.5: 50% chance to horizontally flip


