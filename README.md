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

## 📊 Dataset Analysis

### Data Distribution

![Class Distribution](dist.png)
_Figure 1: Class distribution across the training set showing significant imbalance._

### Key Insights

- **Data Split:** Our dataset contains **4,934 images**, split into **80% Train, 10% Validation, and 10% Test**.
- **Class Imbalance:** The dataset is imbalanced. The most common class, **Emphysema**, appears **552** times, while the rarest class,**Hernia**,appears only **162** times, a ratio of **3.4x**.

### Action Plan

Due to this imbalance, our training strategy will incorporate **data augmentation** (specifically distinct geometric transformations) and we will closely monitor **per-class mAP** rather than global accuracy. Future work could involve weighted loss functions to penalize misclassifications of the minority classes (Hernia, Pneumothorax).

## Error Analysis: Failure Modes

An analysis of the model's false negatives on the test set revealed three primary reasons for detection failure:

1.  **Poor Contrast & Image Quality:** The model struggled with images exhibiting significant blur or a "washed-out" appearance. The lack of sharp edge definition made it difficult to differentiate pathology from healthy tissue.
2.  **Patient Positioning & Rotation:** False negatives were frequent in scans where the patient was rotated or tilted. This distortion alters expected anatomical landmarks, confusing the model's spatial priors.
3.  **Medical Artifacts & Occlusion:** The presence of external objects (tubes, wires, pacemakers) or overlapping anatomy (elevated diaphragm) often obscured the lung fields, leading to missed detections in those specific regions.

**Action Plan:** Future iterations will include augmentation for rotation (+/- 15 degrees) and contrast adjustment (CLAHE) to make the model robust to these specific failures.

**experiment**
Experiment_ID Model_Size Augmentation? Epochs mAP50 Observation
Baseline Medium No 30 0.115 Learned basic features quickly.
Advanced Medium Yes (Rot/Scale) 40 0.107 Slight drop; likely under-trained due to data complexity.
