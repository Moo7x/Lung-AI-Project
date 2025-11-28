# 🫁 Lung Abnormality Detection System (YOLOv8)

## Project Overview

This project uses Deep Learning (YOLOv8) to detect 14 different lung abnormalities from Chest X-ray images. We compare three different model architectures (Nano, Small, Medium) to find the optimal balance between accuracy and speed for clinical deployment.

## 📂 Dataset

- **Source:** NIH Chest X-rays / VinDr-CXR
- **Classes:** 14 (Atelectasis, Cardiomegaly, Pneumonia, etc.)
- **Size:** ~5,000 images split into Train/Test/Validation.

## 🛠️ Tech Stack

- **Model:** YOLOv8 (Ultralytics)
- **Training:** PyTorch with CUDA (GPU Acceleration)
- **Experimentation:** Data Augmentation (Mosaic, Scaling, Flipping)

## 🚀 How to Run Training

1. Install dependencies:
   ```bash
   pip install ultralytics
   ```
