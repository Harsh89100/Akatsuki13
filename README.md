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

## Train and Augmentaion properties

# 🧠 YOLOv8 Training Script Parameters Explained

This document explains each training and augmentation parameter used while training the YOLOv8 model using the Ultralytics Python API.

---

## 📌 Training Parameters

| Parameter       | Description |
|------------------|-------------|
| `data`           | Path to the `.yaml` file describing dataset structure. It includes train, val directories, and class names. |
| `epochs`         | Number of full passes through the training dataset. More epochs allow the model to learn better but can lead to overfitting. |
| `patience`       | Number of epochs to wait before early stopping if validation performance doesn't improve. Helps prevent unnecessary training. |
| `device`         | Index of the device to train on (e.g., `0` for the first GPU, `cpu` for CPU). |
| `single_cls`     | If `True`, treats all classes as one (useful for binary detection tasks). If `False`, handles multi-class detection. |
| `optimizer`      | The optimization algorithm. YOLOv8 supports `SGD`, `Adam`, and `AdamW`. You’ve selected `AdamW` (Adam with weight decay). |
| `lr0`            | Initial learning rate. Controls how much the model updates per batch initially. |
| `lrf`            | Final learning rate multiplier. Final LR = `lr0 × lrf`. A lower value means the LR decreases more over time. |
| `momentum`       | Momentum factor used by optimizers (mainly affects `SGD`, but applicable to `AdamW` too). It helps accelerate gradients in the right direction. |
| `workers`        | Number of data loading workers (parallel processes to load data). More workers speed up data loading on multi-core CPUs. |

---

## 🎨 Data Augmentation Parameters

These improve model generalization by slightly altering training images in diverse ways.

| Parameter     | Description |
|---------------|-------------|
| `mosaic`      | Probability of using Mosaic augmentation. Mosaic combines 4 training images into one — helping with context and object variety. |
| `hsv_h`       | Image hue (color shade) augmentation factor. Small jitter in hue simulates lighting changes. |
| `hsv_s`       | Saturation jitter. Randomly changes image saturation. Affects how vivid colors appear. |
| `hsv_v`       | Brightness jitter. Randomly alters image brightness. |
| `degrees`     | Random rotation (in degrees) for training images. Helps the model become rotation invariant. |
| `translate`   | Image shift amount. Moves the image slightly in X or Y direction — useful for detecting off-centered objects. |
| `scale`       | Random zoom in/out. Changes image scale during training. |
| `shear`       | Shearing transformation — shifts pixels diagonally to skew the image. Helps recognize distorted or angled objects. |
| `flipud`      | Vertical flip probability (0.5 means 50% chance of flipping image upside down). |
| `fliplr`      | Horizontal flip probability (flips left-to-right). Useful for symmetry in images. |

---

## 📁 File and Directory Parameters

| Parameter        | Description |
|------------------|-------------|
| `DATA_YAML`      | Path to the YAML file that defines training, validation paths, and class names. |
| `WEIGHTS_DIR`    | Directory where the trained model weights will be saved after training. |
| `VAL_IMAGES_DIR` | Path to validation images used to evaluate performance during training. |

---

## ✅ Summary

This configuration uses:
- **YOLOv8m model** (medium version)
- **AdamW optimizer** for weight regularization
- A **balanced set of augmentations** to make the model robust
- **Early stopping** to avoid overfitting

Perfect for training on synthetic datasets with varied lighting, occlusions, and perspectives.

---

Let me know if you want a similar explanation for inference or evaluation parameters.



