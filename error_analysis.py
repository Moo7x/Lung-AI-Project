import os
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = os.path.join('runs', 'detect', 'yolo_medium_run2', 'weights', 'best.pt')
TEST_IMAGE_DIR = os.path.join('datasets', 'lung_data', 'test', 'images')
OUTPUT_DIR = 'error_analysis_results'
CONFIDENCE_THRESHOLD = 0.1 # We are looking for things the model was very unsure about

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load your trained model
print(f"Loading model from {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# 2. Run prediction on the entire test set
print(f"Running inference on images in {TEST_IMAGE_DIR}...")
results = model.predict(source=TEST_IMAGE_DIR)

# 3. Analyze results to find "bad" predictions
print(f"Analyzing {len(results)} predictions to find potential errors...")
false_negatives_count = 0

for r in results:
    image_path = r.path
    image_name = os.path.basename(image_path)
    
    # Check if the model found ANY boxes with low confidence
    # This is a simple way to find images where the model struggled
    found_low_confidence_box = False
    if len(r.boxes) > 0:
        for box in r.boxes:
            if box.conf < CONFIDENCE_THRESHOLD:
                found_low_confidence_box = True
                break

    # If the model found nothing, or only low-confidence boxes, it's a potential "miss" (False Negative)
    # A more complex analysis would compare this to ground truth labels
    if len(r.boxes) == 0 or found_low_confidence_box:
        false_negatives_count += 1
        print(f"Potential miss detected in: {image_name}")
        
        # Save a copy of this image for manual review
        img = cv2.imread(image_path)
        save_path = os.path.join(OUTPUT_DIR, f"potential_miss_{image_name}")
        cv2.imwrite(save_path, img)

print(f"\nAnalysis complete. Found {false_negatives_count} potential misses.")
print(f"Check the '{OUTPUT_DIR}' folder to see the images the model struggled with.")