# 🫁 Lung Abnormality Detection System (YOLOv8)

![Project Status](https://img.shields.io/badge/Status-Prototype_Complete-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Model-YOLOv8-orange)
![Backend](https://img.shields.io/badge/API-FastAPI-teal)

## 📖 Project Overview

This project implements an end-to-end AI pipeline for detecting **14 distinct lung abnormalities** (e.g., Pneumonia, Cardiomegaly, Nodule) from Chest X-rays.

Unlike standard classification tasks, we tackle **Object Detection** to localize specific disease regions. The system includes a full engineering workflow: from Exploratory Data Analysis (EDA) and Model Training to Error Analysis and a deployable REST API.

---

## 📂 1. Dataset & Preparation

We utilize a subset of the **NIH Chest X-ray Dataset** consisting of **4,934 images**.

- **Split Strategy:** 80% Train / 10% Validation / 10% Test.
- **Format:** Standard YOLO format (normalized bounding box coordinates).

### Data Analysis (EDA).

Before training, we conducted a rigorous analysis of the dataset structure (`notebooks/01_EDA.ipynb`).

- **Class Imbalance:** The data shows a **3.4x imbalance ratio**, which is significant but manageable.
  - _Most Common:_ **Emphysema** (552 instances)
  - _Rarest:_ **Hernia** (162 instances)
- **Action Taken:** To combat this, we implemented **Data Augmentation** (Mosaic, Scaling, Flipping) during the advanced training phase to ensure the model sees enough variations of the rare classes.

_(Insert your dist.png image here)_

> _Figure 1: Class distribution showing the prevalence of Emphysema vs. rare classes._

---

## 🛠️ 2. Technical Workflow

We followed a modular engineering structure to ensure reproducibility.

| Component    | Tech Stack               | Description                                                |
| :----------- | :----------------------- | :--------------------------------------------------------- |
| **Model**    | YOLOv8 (Ultralytics)     | Selected `yolov8m` (Medium) for balance of speed/accuracy. |
| **Training** | PyTorch + CUDA           | Training on local GPU (RTX 3050 Ti).                       |
| **Backend**  | FastAPI                  | Asynchronous REST API for model serving.                   |
| **Tracking** | Weights & Biases / Local | Loss curves and mAP monitoring.                            |

---

## 📊 3. Model Experiments & Results

We compared a Baseline training run against an Advanced run with hyperparameter tuning.

| Experiment ID | Model Size | Augmentation         | Epochs | mAP50     | Observation                                                                                                        |
| :------------ | :--------- | :------------------- | :----- | :-------- | :----------------------------------------------------------------------------------------------------------------- |
| **Baseline**  | Medium     | None                 | 30     | **0.115** | Converged quickly; learned structural features.                                                                    |
| **Advanced**  | Medium     | **Rot/Scale/Mosaic** | 40     | 0.107     | Slight drop. Hypothesis: The augmented data is significantly harder; model requires 100+ epochs to fully converge. |

### Error Analysis (Failure Modes)

Post-training analysis revealed three primary reasons for false negatives:

1.  **Low Contrast:** The model misses findings in "washed-out" X-rays where tissue density is unclear.
2.  **Small Lesions:** Tiny Nodules (<5% image area) are frequently missed due to 640x640 resizing.
3.  **Occlusion:** Diseases hidden behind the heart or diaphragm are harder to detect.

---

## 🚀 4. How to Run This Project

This project is set up for easy reproduction.

### A. Environment Setup

```bash
# 1. Clone the repo
git clone https://github.com/Moo7x/Lung-AI-Project.git

# 2. Install dependencies
pip install -r requirements.txt
B. Run the API (Backend)

We have separated the inference logic into a modular API.

# Start the server
uvicorn src.api.main:app --reload

Docs: Go to http://127.0.0.1:8000/docs to test the POST /predict endpoint.

C. Run Training (Reproducibility)

To retrain the model from scratch using our settings:
python train_advanced.py

## 🔮Future Improvements

1.Weighted Loss: Implement cls weights in YOLO to explicitly penalize missing rare classes like Hernia.
2.Higher Resolution: Train at 1024x1024 to improve small Nodule detection.
3.Ensembling: Combine predictions from YOLOv8 and Faster R-CNN.
```
