# Akatsuki13

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

### Woriking in Colab
Click below to acces a Colab notebook for training model.

<a href="https://colab.research.google.com/drive/1uqmggmGkMwa2eB0H0kID77_l6EOKP-H2?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### 🚀 Why Training Models in Google Colab Is Easy:

#### ✅ 1. Free Access to Powerful Hardware (GPU & TPU)
Google Colab gives you **free access to NVIDIA GPUs** (like Tesla T4 or A100) — without needing your own high-end machine.

### ✅ 2. Easy Integration with Google Drive
You can load datasets and save model checkpoints directly to/from **Google Drive**.

#### ✅ 5. Pre-installed ML Libraries
Libraries like **PyTorch, TensorFlow, OpenCV**, etc. are already installed.  
You can install anything else with just:

```python
!pip install your-library-name
```



# 🧠 YOLOv8 Training Script Parameters Explained

---

# 🚀 YOLOv8 Training Configuration and Parameters

This document explains every training and augmentation parameter used to train your YOLOv8m model on a custom dataset.

---

## 🧠 Model and Dataset

- **Model Used**: `yolov8m.pt` (YOLOv8 Medium)
- **Dataset YAML**: `/content/drive/MyDrive/HackByte_Dataset/yolo_params.yaml`
- **Output Weights Dir**: `/content/drive/MyDrive/HackByte_Dataset/runs/detect/train/weights`
- **Validation Images**: `/content/drive/MyDrive/HackByte_Dataset/data/val/images`

---

## 📌 Training Parameters

| Parameter     | Value     | Description |
|---------------|-----------|-------------|
| `epochs`      | `40`      | Number of full passes through the dataset. |
| `patience`    | `6`       | Stops training if no improvement is seen for 6 epochs (early stopping). |
| `device`      | `0`       | Uses GPU with index 0. |
| `single_cls`  | `False`   | Enables multi-class training. |
| `optimizer`   | `'AdamW'` | Adam optimizer with weight decay regularization. |
| `lr0`         | `0.001`   | Initial learning rate. |
| `lrf`         | `0.01`    | Final LR multiplier (final LR = lr0 × lrf). |
| `momentum`    | `0.937`   | Momentum factor (mainly affects SGD, but supported in AdamW). |
| `workers`     | `2`       | Number of data loading worker processes. |

---

## 🎨 Data Augmentation Parameters



| Parameter     | Value  | Description |
|---------------|--------|-------------|
| `mosaic`      | `0.5`  | 50% chance to apply Mosaic augmentation (combines 4 images). |
| `hsv_h`       | `0.015`| Jitter on image hue. |
| `hsv_s`       | `0.7`  | Jitter on saturation. |
| `hsv_v`       | `0.4`  | Jitter on brightness. |
| `degrees`     | `10.0` | Random rotation of ±10 degrees. |
| `translate`   | `0.1`  | Translate images by 10% in any direction. |
| `scale`       | `0.5`  | Random zoom between 50% to 150%. |
| `shear`       | `0.1`  | Applies a shear (diagonal transformation). |
| `flipud`      | `0.5`  | 50% chance to vertically flip image. |
| `fliplr`      | `0.5`  | 50% chance to horizontally flip image. |

---

## 📁 Paths

| Parameter        | Value |
|------------------|-------|
| `DATA_YAML`      | `/content/drive/MyDrive/HackByte_Dataset/yolo_params.yaml` |
| `WEIGHTS_DIR`    | `/content/drive/MyDrive/HackByte_Dataset/runs/detect/train/weights` |
| `VAL_IMAGES_DIR` | `/content/drive/MyDrive/HackByte_Dataset/data/val/images` |

---

## ✅ Summary

This setup uses:
- **YOLOv8m** with balanced augmentation and optimizer settings.
- **Early stopping** after 6 epochs without improvement.
- **Strong color/shape transformations** like HSV jitter, shear, rotation, and flips.
- **Efficient loading and saving** via Google Drive in Google Colab with GPU (`device=0`).

This configuration is well-suited for diverse, synthetic datasets with object occlusion, lighting variation, and complex spatial relationships.

---




