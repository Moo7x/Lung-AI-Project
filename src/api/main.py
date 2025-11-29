from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI(title="Lung AI Detection API (Real)")

# --- 1. CONFIGURATION ---
# We define the path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This gets us to 'src' folder
MODEL_PATH = os.path.join(BASE_DIR, "models", "lung_model.pt")

# --- 2. LOAD MODEL ---
print(f"Loading model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    print("SUCCESS: Model loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. {e}")
    model = None

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Sanity Check
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "File type not supported. Please upload a JPEG or PNG image."}

    if model is None:
        return {"error": "Model failed to load. Check server logs."}

    # 1. Read Image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return {"error": "Invalid image file."}

    # 2. Run Inference
    # conf=0.15 -> Lower confidence threshold to catch more diseases (since medical data is hard)
    results = model.predict(image, conf=0.15,augment=True) 

    # 3. Format Results
    detections = []
    for result in results:
        for box in result.boxes:
            # Get coordinates (x1, y1, x2, y2)
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(x) for x in coords]
            
            # Get Class Name
            class_id = int(box.cls[0])
            label = model.names[class_id]
            
            # Get Confidence
            conf = float(box.conf[0])

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

    return {
        "filename": file.filename,
        "message": "Prediction successful",
        "detections_count": len(detections),
        "detections": detections
    }

# To run: uvicorn src.api.main:app --reload