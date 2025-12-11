from ultralytics import YOLO
import cv2
import os
import random

# --- CONFIGURATION ---
MY_MODEL_PATH = os.path.join("src", "models", "lung_model.pt")
TEAMMATE_MODEL_PATH = os.path.join("src", "models", "teammate_model.pt")
TEST_IMAGES_DIR = os.path.join("datasets", "lung_data", "test", "images")
OUTPUT_DIR = "ensemble_results"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_ensemble():
    # 1. Load Both Models
    print(f"Loading My Model: {MY_MODEL_PATH}...")
    model_a = YOLO(MY_MODEL_PATH)
    
    print(f"Loading Teammate Model: {TEAMMATE_MODEL_PATH}...")
    model_b = YOLO(TEAMMATE_MODEL_PATH)

    # 2. Pick 10 Random Images from Test Set
    all_images = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
    selected_images = random.sample(all_images, 10)

    print(f"Running Ensemble on {len(selected_images)} images...")

    for img_name in selected_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        
        # Read Image
        img = cv2.imread(img_path)
        
        # --- PREDICT WITH MY MODEL (BLUE) ---
        results_a = model_a.predict(img_path, conf=0.15)[0]
        for box in results_a.boxes:
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            label = model_a.names[int(box.cls[0])]
            
            # Draw Blue Box (BGR format: 255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"Me: {label}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # --- PREDICT WITH TEAMMATE MODEL (RED) ---
        results_b = model_b.predict(img_path, conf=0.15)[0]
        for box in results_b.boxes:
            x1, y1, x2, y2 = [int(x) for x in box.xyxy[0]]
            label = model_b.names[int(box.cls[0])]
            
            # Draw Red Box (BGR format: 0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Team: {label}", (x1, y1-25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Save Result
        save_path = os.path.join(OUTPUT_DIR, f"ensemble_{img_name}")
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    run_ensemble()